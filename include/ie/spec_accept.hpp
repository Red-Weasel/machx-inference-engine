// include/ie/spec_accept.hpp — one step of speculative sampling (Leviathan et al. 2023, Chen et al. 2023). Host only.
//
// A drafter proposed `draft`, drawn from its own distribution q. The model's sampling distribution for the same position
// is p (the candidates the sampler would draw from and their probabilities, summing to 1). The step keeps the draft with
// probability min(1, p(draft) / q(draft)); otherwise it returns a token drawn from max(0, p - q), normalised. Whatever q
// is, the token returned is distributed exactly as p -- the drafter changes the speed, never the distribution.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace ie {

struct SpecStep { bool accepted; int32_t token; };

// q_of(id): the drafter's probability of a token (asked for the draft and, after a rejection, for p's candidates).
// uniform(): a draw from [0, 1). The draft must have been DRAWN from q (a draft chosen any other way breaks the guarantee).
template <class QOf, class Uniform>
SpecStep spec_accept(const std::vector<int32_t>& p_ids, const std::vector<double>& p_probs, int32_t draft, QOf&& q_of, Uniform&& uniform) {
    double pd = 0;
    for (size_t i = 0; i < p_ids.size(); ++i) if (p_ids[i] == draft) { pd = p_probs[i]; break; }
    if (uniform() * q_of(draft) < pd) return {true, draft};
    // rejected: the residual over p's candidates (q's mass on a token p excludes leaves nothing there)
    std::vector<double> w(p_ids.size()); double tot = 0;
    for (size_t i = 0; i < p_ids.size(); ++i) { w[i] = std::max(0.0, p_probs[i] - q_of(p_ids[i])); tot += w[i]; }
    if (tot <= 0) { w = p_probs; tot = 0; for (double x : w) tot += x; }      // p <= q on all of p cannot reject; rounding: draw from p
    const double u = uniform() * tot; double c = 0; int32_t next = p_ids.empty() ? draft : p_ids.back();
    for (size_t i = 0; i < p_ids.size(); ++i) { c += w[i]; if (u < c) { next = p_ids[i]; break; } }
    return {false, next};
}

}  // namespace ie
