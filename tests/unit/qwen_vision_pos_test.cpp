// tests/unit/qwen_vision_pos_test.cpp -- host-only (no GPU, no model): the Qwen3.8 vision position builder and
// the splice arithmetic the 27B split shares with Flash-Next (include/ie/qwen4_vision.hpp; P4 B45,
// ~/ds41_work/p60/vision-splits/STUDY.md step 1).
//
// (1) Text-only identity: with no image qwen4_build_mrope3 gives every stream the token position and delta 0,
//     and qwen4_mrope3_slice hands every piece (prompt or decode, any start) pos3 = [p, p, p]: rope_imrope3 then
//     reads positions[(r % 3) * T + t] == p for every rotary pair r -- the position rope_partial reads
//     (tests/unit/rope_imrope3_test.cpp is the kernel-level bit identity on equal streams; this is the table side).
// (2) The HF get_rope_index contract (modeling_qwen3_5.py:1386-1475, quoted in the study): text linear on 3 streams,
//     an image's rows T = start, H = start + row, W = start + col over its merged grid, the text after it at
//     max + 1, delta = max + 1 - n so a decode row at token position p ropes at p + delta.
// (3) The splice ranges per piece: inside, straddling either end, covering, two images in one piece, no overlap,
//     and a 1-row piece.
#undef NDEBUG
#include "ie/qwen4_vision.hpp"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {
int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

// What rope_imrope3 reads for rotary pair r of token t from a [3, T] stream-major table (elementwise.cpp:776).
int32_t imrope_pos(const std::vector<int32_t>& p3, uint32_t T, uint32_t r, uint32_t t) {
    return p3[size_t(r % 3u) * T + t];
}

void test_text_identity() {
    for (uint32_t n : {1u, 7u, 513u}) {
        std::vector<int32_t> pos3; int32_t delta = 99;
        ie::qwen4_build_mrope3(n, {}, pos3, delta);
        check(pos3.size() == size_t(3) * n && delta == 0, "text-only table: [3, n], delta 0");
        for (uint32_t s = 0; s < 3; ++s)
            for (uint32_t t = 0; t < n; ++t) check(pos3[size_t(s) * n + t] == int32_t(t), "text-only table: every stream = the position");
        // every piece of a text-only request: prompt pieces inside the table, decode rows past it
        for (auto [start, T] : std::vector<std::pair<uint32_t, uint32_t>>{{0, n}, {0, 1}, {n / 2, n - n / 2}, {n, 1}, {n + 37, 1}, {n - 1, 3}}) {
            std::vector<int32_t> p3(size_t(3) * T);
            ie::qwen4_mrope3_slice(pos3.data(), n, delta, start, T, p3.data());
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t r = 0; r < 32; ++r)   // the 32 rotary pairs of n_rotary 64
                    check(imrope_pos(p3, T, r, t) == int32_t(start + t), "text-only piece: rope_imrope3 reads the token position for every pair");
        }
    }
}

void test_image_contract() {
    // 5 text tokens, <|vision_start|>, a 2 x 3 merged grid (6 pads), <|vision_end|>, 4 text tokens: n = 17
    const uint32_t n = 17;
    std::vector<int32_t> pos3; int32_t delta = 0;
    ie::qwen4_build_mrope3(n, {{6, 2, 3}}, pos3, delta);
    auto P = [&](uint32_t s, uint32_t t) { return pos3[size_t(s) * n + t]; };
    for (uint32_t t = 0; t < 6; ++t) for (uint32_t s = 0; s < 3; ++s) check(P(s, t) == int32_t(t), "text before the image: linear on 3 streams");
    for (uint32_t r = 0; r < 2; ++r)
        for (uint32_t c = 0; c < 3; ++c) {
            const uint32_t t = 6 + r * 3 + c;
            check(P(0, t) == 6, "image T stream = the start position");
            check(P(1, t) == int32_t(6 + r), "image H stream = start + row");
            check(P(2, t) == int32_t(6 + c), "image W stream = start + col");
        }
    // after the image: cur = 6 + max(2, 3) = 9 for <|vision_end|>, then 10..13
    for (uint32_t t = 12; t < n; ++t) for (uint32_t s = 0; s < 3; ++s) check(P(s, t) == int32_t(9 + (t - 12)), "text after the image continues at max + 1");
    check(delta == 13 + 1 - int32_t(n), "delta = max + 1 - n (-3 here)");
    // the decode rows: position n ropes at n + delta = max + 1, all streams equal
    std::vector<int32_t> p3(3);
    ie::qwen4_mrope3_slice(pos3.data(), n, delta, n, 1, p3.data());
    check(p3[0] == 14 && p3[1] == 14 && p3[2] == 14, "first decode row ropes at max + 1 on every stream");
    ie::qwen4_mrope3_slice(pos3.data(), n, delta, n + 5, 1, p3.data());
    check(p3[0] == 19 && p3[1] == 19 && p3[2] == 19, "a later decode row ropes at position + delta");
    // a prompt piece straddling the image end takes the table rows, then the linear tail: [10, 20) = 7 prompt + 3 decode-style
    std::vector<int32_t> q3(30);
    ie::qwen4_mrope3_slice(pos3.data(), n, delta, 10, 10, q3.data());
    check(q3[0] == P(0, 10) && q3[10] == P(1, 10) && q3[20] == P(2, 10), "slice: a prompt row from the table (stream-major, stride T)");
    check(q3[6] == P(0, 16) && q3[7] == 14 && q3[9] == 16, "slice: the last prompt row, then position + delta");
    // two images: the second starts where the first's cur left off; its rows do not restart at 0
    ie::qwen4_build_mrope3(20, {{1, 2, 2}, {8, 3, 2}}, pos3, delta);
    check(pos3[1] == 1 && pos3[size_t(20) + 4] == 2 && pos3[5] == 3, "two images: first at 1 (H up to 2), the token after at 1 + 2 = 3");
    // (text tokens 5, 6, 7 take 3, 4, 5; the second image starts at 6: T 6, H 6 + row, W 6 + col)
    check(pos3[8] == 6 && pos3[size_t(20) + 8 + 2 * 2] == 6 + 2 && pos3[size_t(2) * 20 + 8 + 1] == 6 + 1, "two images: the second's streams start at the running position");
    check(delta == 14 + 1 - 20, "two images: delta from the last position (14)");
}

void test_splice_ranges() {
    using S = ie::Qwen4VisSpan;
    const std::vector<S> spans = {{10, 6, 0}, {30, 4, 6}};   // image rows [10,16) <- staged 0..5, [30,34) <- staged 6..9
    auto R = [&](uint32_t start, uint32_t T) { return ie::qwen4_vis_splice_ranges(spans, start, T); };
    { auto r = R(0, 8); check(r.empty(), "no overlap: nothing"); }
    { auto r = R(0, 40); check(r.size() == 2 && r[0].dst == 10 && r[0].src == 0 && r[0].n == 6 && r[1].dst == 30 && r[1].src == 6 && r[1].n == 4, "covering piece: both spans whole"); }
    { auto r = R(8, 4);  check(r.size() == 1 && r[0].dst == 2 && r[0].src == 0 && r[0].n == 2, "piece straddling the start: the span's first rows"); }
    { auto r = R(13, 10); check(r.size() == 1 && r[0].dst == 0 && r[0].src == 3 && r[0].n == 3, "piece straddling the end: the span's last rows"); }
    { auto r = R(12, 2); check(r.size() == 1 && r[0].dst == 0 && r[0].src == 2 && r[0].n == 2, "piece inside the span"); }
    { auto r = R(15, 1); check(r.size() == 1 && r[0].dst == 0 && r[0].src == 5 && r[0].n == 1, "1-row piece inside the span: one row"); }
    { auto r = R(16, 1); check(r.empty(), "1-row piece right after the span: nothing"); }
    { auto r = R(14, 18); check(r.size() == 2 && r[0].dst == 0 && r[0].src == 4 && r[0].n == 2 && r[1].dst == 16 && r[1].src == 6 && r[1].n == 2, "piece meeting both spans partially"); }
    // the row arithmetic: a copied row's source is its position's offset into the span, from the span's staged row
    for (uint32_t start = 0; start < 40; ++start)
        for (uint32_t T = 1; T <= 12; ++T)
            for (const auto& c : R(start, T)) {
                check(c.dst + c.n <= T, "a range stays inside the piece");
                bool in_span = false;
                for (const S& vs : spans)
                    if (start + c.dst >= vs.t0 && start + c.dst + c.n <= vs.t0 + vs.n && c.src == vs.row0 + (start + c.dst - vs.t0)) in_span = true;
                check(in_span, "a range copies the span's rows at the piece position's offset");
            }
}
}  // namespace

int main() {
    test_text_identity();
    test_image_contract();
    test_splice_ranges();
    std::printf("qwen_vision_pos_test: %d checks PASS\n", checks);
    return 0;
}
