#include "solution.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "check failed line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)
using namespace std::chrono_literals;

int main(int argc, char** argv) {
    bool threw = false; try { BoundedQueue<int> q(0); } catch (const std::invalid_argument&) { threw = true; } CHECK(threw);
    { BoundedQueue<int> q(2); CHECK(q.try_push(1) && q.try_push(2) && !q.try_push(3) && q.size() == 2);
      CHECK(q.try_pop() == 1); CHECK(q.pop() == 2); CHECK(!q.try_pop()); q.close(); q.close(); CHECK(!q.push(4) && !q.try_push(5) && !q.pop()); }
    { BoundedQueue<int> q(3); q.push(2); q.push(1); q.push(3); CHECK(q.pop() == 2 && q.pop() == 1 && q.pop() == 3); }
    { BoundedQueue<int> q(3); q.push(1); q.push(2); q.close(); CHECK(q.pop() == 1 && q.try_pop() == 2 && !q.pop()); }
    { BoundedQueue<std::unique_ptr<int>> q(1); CHECK(q.push(std::make_unique<int>(7))); auto p = q.pop(); CHECK(p && **p == 7); }
    // a pop waiting on an empty queue wakes on close
    { BoundedQueue<int> q(1); std::optional<int> got = 5; std::thread t([&] { got = q.pop(); }); std::this_thread::sleep_for(200ms); q.close(); t.join(); CHECK(!got); }
    // a push waiting on a full queue returns false on close; the queued item is still delivered
    { BoundedQueue<int> q(1); q.push(1); bool r = true; std::thread t([&] { r = q.push(2); }); std::this_thread::sleep_for(200ms); q.close(); t.join();
      CHECK(!r); CHECK(q.pop() == 1); CHECK(!q.pop()); }
    // a blocked push proceeds once a slot frees up
    { BoundedQueue<int> q(1); q.push(1); std::atomic<bool> done = false; std::thread t([&] { q.push(2); done = true; });
      std::this_thread::sleep_for(200ms); CHECK(!done); CHECK(q.pop() == 1); t.join(); CHECK(done && q.pop() == 2); }
    // 4 producers x 20,000 items, 4 consumers, capacity 8
    const int P = 4, N = 20000, C = 4; const size_t CAP = 8;
    BoundedQueue<int> q(CAP); std::atomic<bool> over_cap = false;
    std::vector<std::vector<int>> got(C);
    std::vector<std::thread> ts;
    for (int c = 0; c < C; ++c) ts.emplace_back([&, c] {
        while (auto v = (c % 2 ? q.pop() : q.try_pop())) { got[c].push_back(*v); if (q.size() > CAP) over_cap = true; }
        while (auto v = q.pop()) got[c].push_back(*v);   // try_pop consumers keep going with pop once empty
    });
    std::vector<std::thread> ps;
    for (int p = 0; p < P; ++p) ps.emplace_back([&, p] { for (int i = 0; i < N; ++i) { int v = p * N + i; if (i % 3) CHECK(q.push(v)); else while (!q.try_push(v)) std::this_thread::yield(); } });
    for (auto& t : ps) t.join();
    q.close();
    for (auto& t : ts) t.join();
    CHECK(!over_cap);
    std::vector<int> seen(P * N, 0);
    for (int c = 0; c < C; ++c) {
        std::vector<int> last(P, -1);
        for (int v : got[c]) { CHECK(v >= 0 && v < P * N); seen[v]++; int p = v / N; CHECK(v > last[p]); last[p] = v; }
    }
    for (int v = 0; v < P * N; ++v) CHECK(seen[v] == 1);
    std::printf("%s\n", argc > 1 ? argv[1] : "");
}
