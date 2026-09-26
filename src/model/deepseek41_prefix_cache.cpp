// src/model/deepseek41_prefix_cache.cpp — Phase 46 (docs/deepseek41/86): the V4.1 prefix cache. See
// include/ie/deepseek41_forward.hpp (PrefixCacheOptions) for what a checkpoint is and why it is enough.
//
// Every device transfer goes through memory pinned in THAT card's context (the pool blocks, the per-card bounce):
// a pageable or foreign-context host pointer turns a SYCL memcpy into the multi-minute stall of 2026-09-01.
#include "ie/deepseek41_forward.hpp"
#include "ie/deepseek41_prefix_key.hpp"
#include "ie/ops.hpp"                        // onednn_runtime_version
#include "deepseek41_numerics_manifest.h"    // generated at build time (#48): kDs41NumericsManifest

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>   // getenv/atof for IE_DS41_CACHE_KEEP_FREE_GIB
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <sys/statvfs.h>

namespace ie {
namespace {
uint64_t ds41pc_mem_available() {
    std::FILE* f = std::fopen("/proc/meminfo", "r"); if (!f) return 0;
    char line[256]; unsigned long long v = 0; uint64_t out = 0;
    while (std::fgets(line, sizeof line, f)) if (std::sscanf(line, "MemAvailable: %llu kB", &v) == 1) { out = uint64_t(v) * 1024ull; break; }
    std::fclose(f); return out;
}
size_t ds41pc_lcp(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    const size_t n = std::min(a.size(), b.size()); size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}
double ds41pc_ms(const std::chrono::steady_clock::time_point& t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); }
constexpr uint64_t kBounceFloats = 16ull << 20;   // 64 MiB per card
constexpr char kDiskMagic[8] = {'D', 'S', '4', '1', 'P', 'F', 'X', '1'};
uint64_t ds41pc_fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* b = static_cast<const uint8_t*>(p); for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull; return h;
}
uint64_t ds41pc_fs_free(const std::string& dir) { struct statvfs v {}; return statvfs(dir.c_str(), &v) == 0 ? uint64_t(v.f_bavail) * v.f_frsize : 0; }
struct Ds41pcFile {
    std::FILE* f = nullptr;
    explicit Ds41pcFile(const std::string& p, const char* mode) : f(std::fopen(p.c_str(), mode)) {}
    ~Ds41pcFile() { if (f) std::fclose(f); }
    bool w(const void* p, size_t n) { return std::fwrite(p, 1, n, f) == n; }
    bool r(void* p, size_t n) { return std::fread(p, 1, n, f) == n; }
};
}  // namespace

std::string Ds41Forward::set_prefix_cache(bool on, const PrefixCacheOptions& o) {
    // tear down whatever exists first (also the path free_resident takes)
    pc_join_writer();
    pc_disk_.clear();
    {
        std::lock_guard<std::mutex> lk(pc_mu_);
        for (size_t ci = 0; ci < pc_pool_.size() && ci < cards_.size(); ++ci) {
            if (cards_[ci]->q) cards_[ci]->q->wait();
            for (float* b : pc_pool_[ci]) if (b) sycl::free(b, *cards_[ci]->q);
            if (ci < pc_bounce_.size() && pc_bounce_[ci]) sycl::free(pc_bounce_[ci], *cards_[ci]->q);
        }
        pc_pool_.clear(); pc_bounce_.clear(); pc_block_.clear(); pc_off_.clear(); pc_ckpts_.clear();
        pc_slots_.clear(); pc_slot_bytes_ = 0; pc_bounce_n_ = 0; pc_on_ = false;
    }
    if (!on) return {};
    if (!resident_ || cards_.empty() || !m_) return "set_prefix_cache: resident mode only";
    if (o.checkpoints == 0) return "set_prefix_cache: at least one checkpoint";
    const auto& c = m_->config();
    const uint64_t WIN = c.window_size, HD = c.head_dim;
    std::unique_lock<std::mutex> lk(pc_mu_);
    pc_opt_ = o;
    // The headroom this guard insists on before it will keep a conversation's state
    // in a host slot. 24 GiB is deliberately generous -- a forward with pinned banks
    // needs room, and an OOM is worse than a slow turn. But when the server's own
    // memory grows the guard starts refusing EVERY slot (live 2026-09-21: MemAvailable
    // sat at ~17-20 GiB, every 0.4-0.6 GiB slot was refused, and each resume or
    // post-compaction turn then re-prefilled tens of thousands of tokens: 178 s, 104 s,
    // 77 s). Tunable so that trade can be made without a rebuild.
    if (const char* v = std::getenv("IE_DS41_CACHE_KEEP_FREE_GIB"))
        if (const double g = std::atof(v); g >= 0.0)
            pc_opt_.keep_free = uint64_t(g * 1073741824.0);
    pc_pool_.assign(cards_.size(), {}); pc_off_.assign(cards_.size(), std::vector<uint64_t>(c.n_layers, 0));
    pc_block_.assign(cards_.size(), 0); pc_bounce_.assign(cards_.size(), nullptr);
    for (size_t ci = 0; ci < cards_.size(); ++ci) {
        Card& card = *cards_[ci];
        uint64_t off = 0;
        for (uint32_t L = card.L0; L < card.L1; ++L) {
            pc_off_[ci][L] = off; off += WIN * HD;
            const auto& k = m_->layers()[L].kind;
            if (k.is_kv_source && k.compress_ratio > 1) off += 2ull * k.compress_ratio * HD;
        }
        pc_block_[ci] = off;
        for (uint32_t b = 0; b < o.checkpoints; ++b) {
            float* p = sycl::malloc_host<float>(off, *card.q);
            if (!p) { pc_on_ = true; return "set_prefix_cache: pinned checkpoint block alloc failed"; }
            pc_pool_[ci].push_back(p);
        }
        pc_bounce_[ci] = sycl::malloc_host<float>(kBounceFloats, *card.q);
        if (!pc_bounce_[ci]) return "set_prefix_cache: pinned bounce alloc failed";
    }
    pc_bounce_n_ = kBounceFloats;
    pc_ckpts_.assign(o.checkpoints, PcCkpt{});
    pc_on_ = true;
    uint32_t disk_other = 0;
    if (!o.disk_dir.empty()) {
        // #48 (docs/deepseek41/103): the key -- the model (its shape and a fingerprint of its embedding bytes), the entry
        // format, this build's numerics manifest (the sources whose code can change a cached value, the compiler, the
        // flags) and the runtime that still generates arithmetic (each card's GPU and driver, the oneDNN library loaded).
        // Not this executable's identity: a rebuild that changes no arithmetic keeps the entries.
        std::string model = "L" + std::to_string(c.n_layers) + " H" + std::to_string(c.dim) + " HD" + std::to_string(c.head_dim) +
                            " IHD" + std::to_string(c.index_head_dim) + " W" + std::to_string(c.window_size) + " R";
        for (uint32_t L = 0; L < c.n_layers; ++L) model += std::to_string(m_->layers()[L].kind.compress_ratio) + (m_->layers()[L].kind.is_kv_source ? "s" : "");
        if (m_->embed.w && m_->embed.w->data) model += " E" + std::to_string(ds41pc_fnv(m_->embed.w->data, std::min<size_t>(m_->embed.w->nbytes, 1u << 20)));
        std::string runtime = onednn_runtime_version();
        for (size_t ci = 0; ci < cards_.size(); ++ci) {
            const sycl::device dev = cards_[ci]->q->get_device();
            runtime += (ci ? "; " : " | ") + dev.get_info<sycl::info::device::name>() + " driver " + dev.get_info<sycl::info::device::driver_version>();
        }
        pc_disk_key_ = ds41_prefix_disk_key(model, kDs41PrefixDiskFormat, kDs41NumericsManifest, runtime);
        std::error_code ec; std::filesystem::create_directories(o.disk_dir, ec);
        lk.unlock(); disk_other = pc_disk_scan(); lk.lock();
    }
    uint64_t pinned = 0; for (size_t ci = 0; ci < cards_.size(); ++ci) pinned += pc_block_[ci] * o.checkpoints * 4 + kBounceFloats * 4;
    std::fprintf(stderr, "[ds41 prefix cache] on: %u checkpoints (%.1f MiB pinned with the bounces), host slots up to %.1f GiB, disk %s\n",
                 o.checkpoints, double(pinned) / 1048576.0, double(o.host_budget) / 1073741824.0,
                 o.disk_dir.empty() ? "off" : (o.disk_dir + " (" + std::to_string(pc_disk_.size()) + " entries for numerics " +
                                               ds41_numerics_fingerprint(kDs41NumericsManifest) + "; " + std::to_string(disk_other) +
                                               " written under another key ignored)").c_str());
    return {};
}

void Ds41Forward::pc_clear_live() {
    std::lock_guard<std::mutex> lk(pc_mu_);
    for (auto& e : pc_ckpts_) e = PcCkpt{};
}

void Ds41Forward::pc_capture(size_t ci, uint32_t pos) {
    std::lock_guard<std::mutex> lk(pc_mu_);
    if (!pc_on_ || ci >= cards_.size() || pc_ckpts_.empty()) return;
    const uint32_t all = (1u << cards_.size()) - 1u;
    int idx = -1;
    for (size_t i = 0; i < pc_ckpts_.size(); ++i) if (pc_ckpts_[i].used && pc_ckpts_[i].pos == pos) { idx = int(i); break; }
    if (ci == 0) {
        // card 0 opens the entry (it runs every chunk first); a re-capture at the same position starts over
        if (idx < 0) for (size_t i = 0; i < pc_ckpts_.size(); ++i) if (!pc_ckpts_[i].used) { idx = int(i); break; }
        if (idx < 0) {   // evict the LOWEST complete position: the checkpoints near the head are the ones a next turn needs
            for (size_t i = 0; i < pc_ckpts_.size(); ++i)
                if (pc_ckpts_[i].cards_done == all && (idx < 0 || pc_ckpts_[i].pos < pc_ckpts_[size_t(idx)].pos)) idx = int(i);
            if (idx < 0) return;   // every entry still being written by a lagging card: skip this one
        }
        auto& e = pc_ckpts_[size_t(idx)];
        e = PcCkpt{}; e.used = true; e.pos = pos; e.tick = ++pc_tick_;
        e.nc.assign(m_->config().n_layers, 0); e.part_valid.assign(m_->config().n_layers, 0);
    } else if (idx < 0) return;   // card 0 never opened it (evicted, or the cache was cleared meanwhile)
    auto& e = pc_ckpts_[size_t(idx)];
    Card& card = *cards_[ci]; sycl::queue& q = *card.q;
    const auto& c = m_->config(); const uint64_t WIN = c.window_size, HD = c.head_dim;
    float* blk = pc_pool_[ci][size_t(idx)];
    for (uint32_t L = card.L0; L < card.L1; ++L) {
        const auto& st = card.state[L]; const auto& k = m_->layers()[L].kind;
        e.nc[L] = st.nc; e.part_valid[L] = st.part_valid ? 1 : 0;
        const uint64_t off = pc_off_[ci][L];
        q.memcpy(blk + off, st.win_kv, WIN * HD * 4);
        if (k.is_kv_source && k.compress_ratio > 1) {
            const uint64_t R = k.compress_ratio;
            q.memcpy(blk + off + WIN * HD, st.part_kv, R * HD * 4);
            q.memcpy(blk + off + WIN * HD + R * HD, st.part_gate, R * HD * 4);
        }
    }
    e.cards_done |= 1u << ci;
}

std::string Ds41Forward::prefix_checkpoint() {
    if (!pc_on_) return {};
    for (size_t ci = 0; ci < cards_.size(); ++ci) pc_capture(ci, n_pos_);
    for (auto& cp : cards_) cp->q->wait_and_throw();
    return {};
}

std::string Ds41Forward::pc_restore_live(const PcCkpt& e, size_t index) {
    const auto& c = m_->config(); const uint64_t WIN = c.window_size, HD = c.head_dim;
    for (size_t ci = 0; ci < cards_.size(); ++ci) {
        Card& card = *cards_[ci]; sycl::queue& q = *card.q;
        const float* blk = pc_pool_[ci][index];
        for (uint32_t L = card.L0; L < card.L1; ++L) {
            auto& st = card.state[L]; const auto& k = m_->layers()[L].kind;
            const uint64_t off = pc_off_[ci][L];
            q.memcpy(st.win_kv, blk + off, WIN * HD * 4);
            if (k.is_kv_source && k.compress_ratio > 1) {
                const uint64_t R = k.compress_ratio;
                q.memcpy(st.part_kv, blk + off + WIN * HD, R * HD * 4);
                q.memcpy(st.part_gate, blk + off + WIN * HD + R * HD, R * HD * 4);
            }
            st.nc = e.nc[L]; st.part_valid = e.part_valid[L] != 0;
        }
    }
    for (auto& cp : cards_) cp->q->wait_and_throw();
    return {};
}

std::string Ds41Forward::pc_save_live_to_slot(bool& saved) {
    saved = false;
    if (n_pos_ < pc_opt_.min_slot_tokens || all_ids_.size() < n_pos_) return {};
    const auto t0 = std::chrono::steady_clock::now();
    for (auto& cp : cards_) cp->q->wait_and_throw();
    const auto& c = m_->config(); const uint64_t WIN = c.window_size, HD = c.head_dim, IHD = c.index_head_dim;
    const uint32_t all = (1u << cards_.size()) - 1u;
    std::vector<size_t> keep;   // the live checkpoints worth carrying, ascending
    for (size_t i = 0; i < pc_ckpts_.size(); ++i) if (pc_ckpts_[i].used && pc_ckpts_[i].cards_done == all && pc_ckpts_[i].pos < n_pos_) keep.push_back(i);
    std::sort(keep.begin(), keep.end(), [&](size_t a, size_t b) { return pc_ckpts_[a].pos < pc_ckpts_[b].pos; });
    uint64_t block_all = 0; for (uint64_t b : pc_block_) block_all += b;
    uint64_t lat = 0;
    for (size_t ci = 0; ci < cards_.size(); ++ci)
        for (uint32_t L = cards_[ci]->L0; L < cards_[ci]->L1; ++L)
            if (m_->layers()[L].kind.is_kv_source) lat += uint64_t(cards_[ci]->state[L].nc) * (HD + IHD);
    const uint64_t bytes = (uint64_t(keep.size() + 1) * block_all + lat) * 4 + uint64_t(n_pos_) * 4;
    if (bytes > pc_opt_.host_budget) return {};
    while (!pc_slots_.empty() && pc_slot_bytes_ + bytes > pc_opt_.host_budget) {   // LRU
        auto it = std::min_element(pc_slots_.begin(), pc_slots_.end(), [](const PcSlot& a, const PcSlot& b) { return a.tick < b.tick; });
        pc_slot_bytes_ -= it->bytes; pc_slots_.erase(it);
    }
    if (ds41pc_mem_available() < pc_opt_.keep_free + bytes) {
        // Say what the refusal COSTS, not just that it happened: without the slot this
        // conversation re-prefills from the disk entry on its next resume.
        std::fprintf(stderr, "[ds41 prefix cache] not keeping a %.2f GiB host slot: MemAvailable %.1f GiB would fall "
                             "under %.0f GiB — this conversation will re-prefill when it resumes "
                             "(IE_DS41_CACHE_KEEP_FREE_GIB lowers the bar)\n",
                     double(bytes) / 1073741824.0, double(ds41pc_mem_available()) / 1073741824.0,
                     double(pc_opt_.keep_free) / 1073741824.0);
        return {};
    }
    PcSlot s; s.ids.assign(all_ids_.begin(), all_ids_.begin() + n_pos_);
    for (size_t i : keep) {
        s.ckpts.push_back(pc_ckpts_[i]);
        s.ring.emplace_back(cards_.size());
        for (size_t ci = 0; ci < cards_.size(); ++ci) s.ring.back()[ci].assign(pc_pool_[ci][i], pc_pool_[ci][i] + pc_block_[ci]);
    }
    // the live position itself: its rings exist only on the devices right now
    PcCkpt live; live.used = true; live.pos = n_pos_; live.cards_done = all; live.nc.assign(c.n_layers, 0); live.part_valid.assign(c.n_layers, 0);
    s.ring.emplace_back(cards_.size()); s.lat.assign(cards_.size(), std::vector<std::vector<float>>(c.n_layers));
    for (size_t ci = 0; ci < cards_.size(); ++ci) {
        Card& card = *cards_[ci]; sycl::queue& q = *card.q; float* bb = pc_bounce_[ci];
        auto& ring = s.ring.back()[ci]; ring.assign(pc_block_[ci], 0.f);
        for (uint32_t L = card.L0; L < card.L1; ++L) {
            const auto& st = card.state[L]; const auto& k = m_->layers()[L].kind;
            live.nc[L] = st.nc; live.part_valid[L] = st.part_valid ? 1 : 0;
            const uint64_t off = pc_off_[ci][L];
            q.memcpy(bb, st.win_kv, WIN * HD * 4).wait(); std::memcpy(ring.data() + off, bb, WIN * HD * 4);
            if (k.is_kv_source && k.compress_ratio > 1) {
                const uint64_t R = k.compress_ratio;
                q.memcpy(bb, st.part_kv, R * HD * 4); q.memcpy(bb + R * HD, st.part_gate, R * HD * 4).wait();
                std::memcpy(ring.data() + off + WIN * HD, bb, 2 * R * HD * 4);
            }
            if (k.is_kv_source && st.nc) {   // latents then index keys, through the bounce in pieces
                auto& v = s.lat[ci][L]; const uint64_t n = st.nc; v.resize(n * (HD + IHD));
                for (const auto& [src, width, base] : {std::tuple<const float*, uint64_t, uint64_t>{st.comp_kv, HD, 0}, {st.idx_k, IHD, n * HD}})
                    for (uint64_t r0 = 0; r0 < n; ) {
                        const uint64_t rn = std::min<uint64_t>(n - r0, pc_bounce_n_ / width);
                        q.memcpy(bb, src + r0 * width, rn * width * 4).wait();
                        std::memcpy(v.data() + base + r0 * width, bb, rn * width * 4);
                        r0 += rn;
                    }
            }
        }
    }
    s.ckpts.push_back(std::move(live));
    s.bytes = bytes; s.tick = ++pc_tick_;
    pc_slot_bytes_ += bytes; pc_slots_.push_back(std::move(s));
    pc_save_ms_ = ds41pc_ms(t0); saved = true;
    std::fprintf(stderr, "[ds41 prefix cache] kept the live conversation (%u tokens, %zu checkpoints, %.1f MiB) in a host slot in %.0f ms; %zu slots, %.2f GiB\n",
                 n_pos_, pc_slots_.back().ckpts.size(), double(bytes) / 1048576.0, pc_save_ms_, pc_slots_.size(), double(pc_slot_bytes_) / 1073741824.0);
    return {};
}

std::string Ds41Forward::pc_load_slot(PcSlot& s, size_t k) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto& c = m_->config(); const uint64_t WIN = c.window_size, HD = c.head_dim, IHD = c.index_head_dim;
    const PcCkpt& e = s.ckpts[k]; const PcCkpt& last = s.ckpts.back();
    for (size_t ci = 0; ci < cards_.size(); ++ci) {
        Card& card = *cards_[ci]; sycl::queue& q = *card.q; float* bb = pc_bounce_[ci];
        q.wait_and_throw();
        const auto& ring = s.ring[k][ci];
        for (uint32_t L = card.L0; L < card.L1; ++L) {
            auto& st = card.state[L]; const auto& kind = m_->layers()[L].kind;
            const uint64_t off = pc_off_[ci][L];
            std::memcpy(bb, ring.data() + off, WIN * HD * 4); q.memcpy(st.win_kv, bb, WIN * HD * 4).wait();
            if (kind.is_kv_source && kind.compress_ratio > 1) {
                const uint64_t R = kind.compress_ratio;
                std::memcpy(bb, ring.data() + off + WIN * HD, 2 * R * HD * 4);
                q.memcpy(st.part_kv, bb, R * HD * 4); q.memcpy(st.part_gate, bb + R * HD, R * HD * 4).wait();
            }
            if (kind.is_kv_source && e.nc[L]) {
                const uint64_t n = e.nc[L], nlast = last.nc[L];
                const auto& v = s.lat[ci][L];
                for (const auto& [dst, width, base] : {std::tuple<float*, uint64_t, uint64_t>{st.comp_kv, HD, 0}, {st.idx_k, IHD, nlast * HD}})
                    for (uint64_t r0 = 0; r0 < n; ) {
                        const uint64_t rn = std::min<uint64_t>(n - r0, pc_bounce_n_ / width);
                        std::memcpy(bb, v.data() + base + r0 * width, rn * width * 4);
                        q.memcpy(dst + r0 * width, bb, rn * width * 4).wait();
                        r0 += rn;
                    }
            }
            st.nc = e.nc[L]; st.part_valid = e.part_valid[L] != 0;
        }
    }
    n_pos_ = e.pos; all_ids_.assign(s.ids.begin(), s.ids.begin() + e.pos);
    // the slot's checkpoints at or below the restored position become the live ones again (the highest first)
    {
        std::lock_guard<std::mutex> lk(pc_mu_);
        for (auto& x : pc_ckpts_) x = PcCkpt{};
        size_t b = 0;
        for (size_t j = k + 1; j-- > 0 && b < pc_ckpts_.size(); ) {
            pc_ckpts_[b] = s.ckpts[j]; pc_ckpts_[b].tick = ++pc_tick_;
            for (size_t ci = 0; ci < cards_.size(); ++ci) std::memcpy(pc_pool_[ci][b], s.ring[j][ci].data(), pc_block_[ci] * 4);
            ++b;
        }
    }
    pc_restore_ms_ = ds41pc_ms(t0);
    return {};
}

std::string Ds41Forward::prefix_prepare(const std::vector<int32_t>& ids, uint32_t& reused, std::string* source) {
    reused = 0; if (source) *source = "none";
    if (!pc_on_ || ids.size() < 2) return {};
    const auto t0 = std::chrono::steady_clock::now();
    for (auto& cp : cards_) cp->q->wait_and_throw();       // in-flight checkpoint copies land first
    const uint32_t all = (1u << cards_.size()) - 1u;
    const size_t limit = ids.size() - 1;                    // the last prompt token is always run: its logits are needed
    // the live state: its own position, or its best checkpoint inside the shared prefix
    const size_t L_live = ds41pc_lcp(all_ids_, ids);
    uint32_t best_live = 0; int live_idx = -2;
    if (n_pos_ > 0 && all_ids_.size() == n_pos_ && n_pos_ <= L_live && n_pos_ <= limit) { best_live = n_pos_; live_idx = -1; }
    for (size_t i = 0; i < pc_ckpts_.size(); ++i) {
        const auto& e = pc_ckpts_[i];
        if (e.used && e.cards_done == all && e.pos <= n_pos_ && e.pos <= L_live && e.pos <= limit && e.pos > best_live) { best_live = e.pos; live_idx = int(i); }
    }
    // the host slots
    uint32_t best_slot = 0; size_t si = 0, sk = 0;
    for (size_t s = 0; s < pc_slots_.size(); ++s) {
        const size_t Ls = ds41pc_lcp(pc_slots_[s].ids, ids);
        for (size_t k = 0; k < pc_slots_[s].ckpts.size(); ++k) {
            const uint32_t p = pc_slots_[s].ckpts[k].pos;
            if (p <= Ls && p <= limit && p > best_slot) { best_slot = p; si = s; sk = k; }
        }
    }
    // the disk entries: usable when the prompt holds an entry's whole prefix
    PcDisk d;   // a copy: the writer thread may change the index meanwhile
    {
        std::lock_guard<std::mutex> lk(pc_mu_);
        for (const auto& x : pc_disk_)
            if (x.ids.size() <= limit && x.ids.size() > d.ids.size() && ds41pc_lcp(x.ids, ids) == x.ids.size()) d = x;
    }
    const uint32_t disk_pos = uint32_t(d.ids.size());
    bool saved = false;
    if (disk_pos > std::max(best_live, best_slot) + 32) {
        if (auto e = pc_save_live_to_slot(saved); !e.empty()) return e;
        if (auto e = pc_load_disk(d); !e.empty()) {
            std::fprintf(stderr, "[ds41 prefix cache] disk entry %s unusable (%s): recomputing\n", d.path.c_str(), e.c_str());
            reset_state();
            return {};
        }
        reused = disk_pos; if (source) *source = "disk";
    } else if (best_slot > best_live + 32) {
        PcSlot slot = std::move(pc_slots_[si]);
        pc_slot_bytes_ -= slot.bytes; pc_slots_.erase(pc_slots_.begin() + std::ptrdiff_t(si));
        if (auto e = pc_save_live_to_slot(saved); !e.empty()) return e;
        if (auto e = pc_load_slot(slot, sk); !e.empty()) return e;
        reused = best_slot; if (source) *source = "host slot";
    } else if (best_live > 0) {
        // truncating the live conversation well below its head drops work a later request may want back: keep it first --
        // but only when PROMPT positions would be lost (a checkpoint min_slot_tokens above the restore point). An agent
        // client that does not echo reasoning_content truncates the generated reasoning every turn; saving that tail
        // would copy the whole conversation to host each turn and evict other conversations' slots for nothing.
        bool prompt_above = false;
        for (const auto& x : pc_ckpts_) if (x.used && x.cards_done == all && x.pos >= best_live + pc_opt_.min_slot_tokens) prompt_above = true;
        if (prompt_above) if (auto e = pc_save_live_to_slot(saved); !e.empty()) return e;
        if (live_idx >= 0) {
            const PcCkpt e = pc_ckpts_[size_t(live_idx)];
            if (auto err = pc_restore_live(e, size_t(live_idx)); !err.empty()) return err;
            n_pos_ = e.pos;
        }
        all_ids_.resize(best_live);
        reused = best_live; if (source) *source = live_idx == -1 ? "live" : "checkpoint";
        std::lock_guard<std::mutex> lk(pc_mu_);
        for (auto& x : pc_ckpts_) if (x.used && x.pos > best_live) x = PcCkpt{};   // their latents are about to be rewritten
        pc_restore_ms_ = ds41pc_ms(t0);
    } else {
        // nothing shared: the caller's pos0 = 0 prefill will reset the state -- keep this conversation if it is worth it
        if (auto e = pc_save_live_to_slot(saved); !e.empty()) return e;
    }
    snap_valid_ = false;
    return {};
}

// ---- Phase 47: disk entries -------------------------------------------------------------------------------------------
// file: magic, u32 version, u32 key length + key, u32 n + ids, u32 n_layers, nc[n_layers] u32, part_valid[n_layers] u8,
// then per model layer: ring [WIN, HD], the ratio-2 halves [2 R HD], the latents [nc HD] and index keys [nc IHD] (fp32)
uint32_t Ds41Forward::pc_disk_scan() {
    std::vector<PcDisk> found;
    uint32_t other = 0;   // entries written under another key: another numerics manifest, runtime, format or model
    std::error_code ec;
    for (const auto& de : std::filesystem::directory_iterator(pc_opt_.disk_dir, ec)) {
        if (!de.is_regular_file() || de.path().extension() != ".ds41pfx") continue;
        Ds41pcFile f(de.path().string(), "rb"); if (!f.f) continue;
        char magic[8]; uint32_t ver = 0, klen = 0, n = 0;
        if (!f.r(magic, 8) || std::memcmp(magic, kDiskMagic, 8) != 0 || !f.r(&ver, 4)) continue;
        if (ver != kDs41PrefixDiskFormat || !f.r(&klen, 4) || klen > 65536) { ++other; continue; }
        std::string key(klen, '\0'); if (!f.r(key.data(), klen)) continue;
        if (key != pc_disk_key_) { ++other; continue; }   // the LRU budget ages these out (prefix_persist's writer)
        if (!f.r(&n, 4) || n == 0 || n > (1u << 24)) continue;
        PcDisk d; d.path = de.path().string(); d.ids.resize(n);
        if (!f.r(d.ids.data(), size_t(n) * 4)) continue;
        d.bytes = uint64_t(de.file_size(ec));
        found.push_back(std::move(d));
    }
    std::lock_guard<std::mutex> lk(pc_mu_);
    pc_disk_ = std::move(found);
    return other;
}

std::string Ds41Forward::prefix_persist(uint32_t pos) {
    if (!pc_on_ || pc_opt_.disk_dir.empty() || pos < pc_opt_.min_disk_tokens || all_ids_.size() < pos) return {};
    std::vector<int32_t> ids(all_ids_.begin(), all_ids_.begin() + pos);
    const uint32_t all = (1u << cards_.size()) - 1u;
    int idx = -1;
    {
        std::lock_guard<std::mutex> lk(pc_mu_);
        for (const auto& d : pc_disk_) if (d.ids == ids) { std::error_code ec; std::filesystem::last_write_time(d.path, std::filesystem::file_time_type::clock::now(), ec); return {}; }
        for (size_t i = 0; i < pc_ckpts_.size(); ++i) if (pc_ckpts_[i].used && pc_ckpts_[i].cards_done == all && pc_ckpts_[i].pos == pos) { idx = int(i); break; }
    }
    if (idx < 0) return {};
    const auto t0 = std::chrono::steady_clock::now();
    for (auto& cp : cards_) cp->q->wait_and_throw();
    const PcCkpt e = pc_ckpts_[size_t(idx)];
    const auto& c = m_->config(); const uint64_t WIN = c.window_size, HD = c.head_dim, IHD = c.index_head_dim;
    // the payload, built in memory now: the rings from the pinned checkpoint block, the latents from the devices (valid
    // only until the state rolls below pos)
    std::vector<float> body;
    uint64_t floats = 0;
    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const auto& k = m_->layers()[L].kind;
        floats += WIN * HD + (k.is_kv_source && k.compress_ratio > 1 ? 2ull * k.compress_ratio * HD : 0) + (k.is_kv_source ? uint64_t(e.nc[L]) * (HD + IHD) : 0);
    }
    const uint64_t bytes = floats * 4 + uint64_t(pos) * 4 + uint64_t(c.n_layers) * 5 + pc_disk_key_.size() + 64;
    if (ds41pc_fs_free(pc_opt_.disk_dir) < pc_opt_.disk_keep_free + bytes) return {};
    if (bytes > pc_opt_.disk_budget) return {};
    body.resize(floats);
    uint64_t at = 0;
    for (size_t ci = 0; ci < cards_.size(); ++ci) {
        Card& card = *cards_[ci]; sycl::queue& q = *card.q; float* bb = pc_bounce_[ci];
        const float* blk = pc_pool_[ci][size_t(idx)];
        for (uint32_t L = card.L0; L < card.L1; ++L) {
            const auto& k = m_->layers()[L].kind; const auto& st = card.state[L];
            const uint64_t off = pc_off_[ci][L];
            std::memcpy(body.data() + at, blk + off, WIN * HD * 4); at += WIN * HD;
            if (k.is_kv_source && k.compress_ratio > 1) { std::memcpy(body.data() + at, blk + off + WIN * HD, 2 * k.compress_ratio * HD * 4); at += 2ull * k.compress_ratio * HD; }
            if (k.is_kv_source) {
                const uint64_t n = e.nc[L];
                // the latents [n, HD] at `at`, then the index keys [n, IHD] right after them
                for (const auto& [src, width, base] : {std::tuple<const float*, uint64_t, uint64_t>{st.comp_kv, HD, at}, {st.idx_k, IHD, at + n * HD}})
                    for (uint64_t r0 = 0; r0 < n; ) {
                        const uint64_t rn = std::min<uint64_t>(n - r0, pc_bounce_n_ / width);
                        q.memcpy(bb, src + r0 * width, rn * width * 4).wait();
                        std::memcpy(body.data() + base + r0 * width, bb, rn * width * 4);
                        r0 += rn;
                    }
                at += n * (HD + IHD);
            }
        }
    }
    const double gather_ms = ds41pc_ms(t0);
    // the file, on a thread: tmp + rename, so a reader never sees a partial entry; then the budget, oldest first
    pc_join_writer();
    const std::string dir = pc_opt_.disk_dir, key = pc_disk_key_;
    const uint64_t budget = pc_opt_.disk_budget;
    char name[64]; std::snprintf(name, sizeof name, "%016llx.ds41pfx", (unsigned long long)ds41pc_fnv(ids.data(), ids.size() * 4, ds41pc_fnv(key.data(), key.size())));
    const std::string path = dir + "/" + name;
    const uint32_t n_tok = pos;
    pc_writer_ = std::thread([this, dir, key, budget, path, ids = std::move(ids), e, body = std::move(body), gather_ms, n_tok, n_layers = c.n_layers]() mutable {
        const auto tw = std::chrono::steady_clock::now();
        const std::string tmp = path + ".tmp";
        bool ok = false;
        {
            Ds41pcFile f(tmp, "wb");
            if (f.f) {
                const uint32_t ver = kDs41PrefixDiskFormat, klen = uint32_t(key.size()), n = uint32_t(ids.size());
                ok = f.w(kDiskMagic, 8) && f.w(&ver, 4) && f.w(&klen, 4) && f.w(key.data(), klen) && f.w(&n, 4) && f.w(ids.data(), size_t(n) * 4) &&
                     f.w(&n_layers, 4) && f.w(e.nc.data(), size_t(n_layers) * 4) && f.w(e.part_valid.data(), n_layers) && f.w(body.data(), body.size() * 4);
                ok = ok && std::fflush(f.f) == 0;
            }
        }
        std::error_code ec;
        if (!ok) { std::filesystem::remove(tmp, ec); std::fprintf(stderr, "[ds41 prefix cache] disk entry write failed: %s\n", path.c_str()); return; }
        std::filesystem::rename(tmp, path, ec);
        if (ec) { std::filesystem::remove(tmp, ec); return; }
        const uint64_t fbytes = uint64_t(std::filesystem::file_size(path, ec));
        // the budget over every entry in the directory (other builds' too), least recently used first
        std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> files; uint64_t total = 0;
        for (const auto& de : std::filesystem::directory_iterator(dir, ec))
            if (de.is_regular_file() && de.path().extension() == ".ds41pfx") { files.push_back({de.last_write_time(ec), de.path()}); total += uint64_t(de.file_size(ec)); }
        std::sort(files.begin(), files.end());
        std::vector<std::string> removed;
        for (const auto& [t, p] : files) { if (total <= budget) break; if (p.string() == path) continue; total -= uint64_t(std::filesystem::file_size(p, ec)); std::filesystem::remove(p, ec); removed.push_back(p.string()); }
        {
            std::lock_guard<std::mutex> lk(pc_mu_);
            pc_disk_.erase(std::remove_if(pc_disk_.begin(), pc_disk_.end(), [&](const PcDisk& d) { return std::find(removed.begin(), removed.end(), d.path) != removed.end() || d.path == path; }), pc_disk_.end());
            PcDisk d; d.path = path; d.ids = std::move(ids); d.bytes = fbytes; pc_disk_.push_back(std::move(d));
        }
        std::fprintf(stderr, "[ds41 prefix cache] wrote a disk entry: %u tokens, %.1f MiB (gathered in %.0f ms, written in %.0f ms)\n",
                     n_tok, double(fbytes) / 1048576.0, gather_ms, ds41pc_ms(tw));
    });
    return {};
}

std::string Ds41Forward::pc_load_disk(const PcDisk& d) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto& c = m_->config(); const uint64_t WIN = c.window_size, HD = c.head_dim, IHD = c.index_head_dim;
    Ds41pcFile f(d.path, "rb"); if (!f.f) return "cannot open";
    char magic[8]; uint32_t ver = 0, klen = 0, n = 0, nl = 0;
    if (!f.r(magic, 8) || std::memcmp(magic, kDiskMagic, 8) != 0 || !f.r(&ver, 4) || ver != kDs41PrefixDiskFormat || !f.r(&klen, 4) || klen > 65536) return "bad header";
    std::string key(klen, '\0'); if (!f.r(key.data(), klen) || key != pc_disk_key_) return "key mismatch";
    if (!f.r(&n, 4) || n != d.ids.size()) return "id count mismatch";
    std::vector<int32_t> ids(n); if (!f.r(ids.data(), size_t(n) * 4) || ids != d.ids) return "ids mismatch";
    if (!f.r(&nl, 4) || nl != c.n_layers) return "layer count mismatch";
    PcCkpt e; e.used = true; e.pos = n; e.cards_done = (1u << cards_.size()) - 1u; e.nc.resize(nl); e.part_valid.resize(nl);
    if (!f.r(e.nc.data(), size_t(nl) * 4) || !f.r(e.part_valid.data(), nl)) return "truncated counts";
    for (uint32_t L = 0; L < nl; ++L) if (e.nc[L] > cap_pos_) return "a latent count exceeds this runtime's capacity";
    for (auto& cp : cards_) cp->q->wait_and_throw();
    // the one checkpoint the entry holds becomes pool block 0 on every card
    {
        std::lock_guard<std::mutex> lk(pc_mu_);
        for (auto& x : pc_ckpts_) x = PcCkpt{};
    }
    for (size_t ci = 0; ci < cards_.size(); ++ci) {
        Card& card = *cards_[ci]; sycl::queue& q = *card.q; float* bb = pc_bounce_[ci]; float* blk = pc_pool_[ci][0];
        // the file holds the layers in model order, and the cards' ranges are contiguous and ascending
        for (uint32_t L = card.L0; L < card.L1; ++L) {
            auto& st = card.state[L]; const auto& k = m_->layers()[L].kind;
            const uint64_t off = pc_off_[ci][L];
            if (!f.r(blk + off, WIN * HD * 4)) return "truncated ring";
            q.memcpy(st.win_kv, blk + off, WIN * HD * 4);
            if (k.is_kv_source && k.compress_ratio > 1) {
                const uint64_t R = k.compress_ratio;
                if (!f.r(blk + off + WIN * HD, 2 * R * HD * 4)) return "truncated half";
                q.memcpy(st.part_kv, blk + off + WIN * HD, R * HD * 4); q.memcpy(st.part_gate, blk + off + WIN * HD + R * HD, R * HD * 4);
            }
            if (k.is_kv_source) {
                const uint64_t nr = e.nc[L];
                for (const auto& [dst, width] : {std::pair<float*, uint64_t>{st.comp_kv, HD}, {st.idx_k, IHD}})
                    for (uint64_t r0 = 0; r0 < nr; ) {
                        const uint64_t rn = std::min<uint64_t>(nr - r0, pc_bounce_n_ / width);
                        q.wait();                                           // the bounce is reused: the previous copy must have landed
                        if (!f.r(bb, rn * width * 4)) return "truncated latents";
                        q.memcpy(dst + r0 * width, bb, rn * width * 4);
                        r0 += rn;
                    }
            }
            st.nc = e.nc[L]; st.part_valid = e.part_valid[L] != 0;
        }
        q.wait_and_throw();
    }
    n_pos_ = n; all_ids_ = ids;
    {
        std::lock_guard<std::mutex> lk(pc_mu_);
        pc_ckpts_[0] = e; pc_ckpts_[0].tick = ++pc_tick_;
    }
    std::error_code ec; std::filesystem::last_write_time(d.path, std::filesystem::file_time_type::clock::now(), ec);
    pc_restore_ms_ = ds41pc_ms(t0);
    std::fprintf(stderr, "[ds41 prefix cache] loaded a disk entry: %u tokens in %.0f ms\n", n, pc_restore_ms_);
    return {};
}

Ds41Forward::PrefixCacheStats Ds41Forward::prefix_cache_stats() const {
    PrefixCacheStats s;
    const uint32_t all = cards_.empty() ? 0u : (1u << cards_.size()) - 1u;
    for (const auto& e : pc_ckpts_) if (e.used && e.cards_done == all) ++s.checkpoints;
    s.host_slots = uint32_t(pc_slots_.size()); s.host_bytes = pc_slot_bytes_;
    s.last_restore_ms = pc_restore_ms_; s.last_save_ms = pc_save_ms_;
    return s;
}

}  // namespace ie
