#pragma once
// --cards (docs/serve_config.md): which physical GPUs one engine process may use.
//
// "Card N" is the N-th DISCRETE Level Zero GPU in PCI address order (the integrated GPU is never a card).
// On this box: card 0 = 0000:04:00.0, card 1 = 0000:09:00.0 -- the same numbers xpu-smi prints. `ie cards` lists them.
//
// Mechanism: before the SYCL runtime starts, the engine enumerates Level Zero itself (card_select.cpp), maps the
// requested cards to the runtime's Level Zero indices and sets ONEAPI_DEVICE_SELECTOR to exactly those, so every
// device list in the engine (the allocator, the planner, the DeepSeek/MiMo runtimes, preflight) sees only the chosen
// cards and the other cards are never opened. After the runtime starts, verify_card_selection() checks by PCI address
// that the runtime exposes exactly the chosen cards and refuses the launch otherwise.
// No --cards = nothing here runs: the device lists are today's.
#include <cstdint>
#include <string>
#include <vector>

namespace ie {

struct LevelZeroGpu {
    uint32_t    index = 0;          // position in the Level Zero device list = the ONEAPI_DEVICE_SELECTOR level_zero:<index>
    std::string pci;                // "0000:04:00.0"
    std::string name;
    bool        integrated = false;
};

// The cards (discrete GPUs), numbered: result[N] is card N (PCI address order).
inline std::vector<LevelZeroGpu> cards_of(std::vector<LevelZeroGpu> gpus) {
    std::vector<LevelZeroGpu> cards;
    for (auto& g : gpus) if (!g.integrated) cards.push_back(std::move(g));
    for (size_t i = 1; i < cards.size(); ++i)          // insertion sort by PCI address (a handful of entries)
        for (size_t j = i; j > 0 && cards[j].pci < cards[j - 1].pci; --j) std::swap(cards[j], cards[j - 1]);
    return cards;
}

// The ONEAPI_DEVICE_SELECTOR value that exposes exactly `wanted` (card numbers); "" + err when a card does not exist.
inline std::string card_selector_for(const std::vector<LevelZeroGpu>& gpus, const std::vector<uint32_t>& wanted,
                                     std::string& err, std::vector<std::string>* pcis = nullptr) {
    const auto cards = cards_of(gpus);
    std::string sel = "level_zero:";
    for (size_t i = 0; i < wanted.size(); ++i) {
        if (wanted[i] >= cards.size()) {
            err = "--cards " + std::to_string(wanted[i]) + ": this machine has " + std::to_string(cards.size()) +
                  " card(s) (numbered 0.." + std::to_string(cards.size() ? cards.size() - 1 : 0) + "; `ie cards` lists them)";
            if (cards.empty()) err = "--cards: no discrete Level Zero GPU found";
            return {};
        }
        sel += (i ? "," : "") + std::to_string(cards[wanted[i]].index);
        if (pcis) pcis->push_back(cards[wanted[i]].pci);
    }
    return sel;
}

// Level Zero enumeration (no SYCL). "" = ok.
std::string enumerate_level_zero_gpus(std::vector<LevelZeroGpu>& out);
// Set ONEAPI_DEVICE_SELECTOR for `cards`; must run before anything touches SYCL. Fills the chosen PCI addresses. "" = ok.
std::string apply_card_selection(const std::vector<uint32_t>& cards, std::vector<std::string>& pcis);
// After the runtime starts: the Level Zero GPUs SYCL exposes must be exactly `pcis`. "" = ok.
std::string verify_card_selection(const std::vector<std::string>& pcis);

}  // namespace ie
