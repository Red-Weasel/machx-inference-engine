#undef NDEBUG
#include "ie/stop_sequences.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
int main() {
 std::vector<std::string> stops={"END","STOP"};
 assert(ie::first_stop_match("aSTOPbEND",stops)==1);
 assert(ie::first_stop_match("aENDbSTOP",stops)==1);
 assert(ie::first_stop_match("thinking STOP</think>xEND",stops,21)==22);
 assert(ie::first_stop_match("aSTOP<tool_call>END",stops,0,5)==1);
 assert(ie::first_stop_match("<tool_call>STOP",stops,0,0)==std::string::npos);
 assert(ie::first_stop_match("ST",stops)==std::string::npos);
 std::puts("stop_sequences_test: OK");
}
