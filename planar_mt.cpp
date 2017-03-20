// https://github.com/boris-dimitrov/z4_planar_langford
//
// Copyright 2017 Boris Dimitrov, Portola Valley, CA 94028.
// Questions? Contact http://www.facebook.com/boris
//
// This program counts the permutations of the sequence 1, 1, 2, 2, 3, 3, ..., n, n
// in which the two occurrences of each m are separated by precisely m other numbers,
// and lines connecting all (m, m) pairs can be drawn on the page without crossing.
//
// See http://www.dialectrix.com/langford.html ("Planar Solutions") or Knuth volume 4a
// page 3.  Todo: Provide better Knuth reference.
//
// Compiling on Mac or Linux:
//
//     g++ -O3 -std=c++11 -DNDEBUG -o planar_mt planar_mt.cpp -lpthread
//
// The resulting executable will be named "planar_mt".
//
//
// THE ALGORITHM
//
// This program runs a depth-first-search aka backtracking algorithm which chooses
// to "open" or "close" a pair at each position, starting with position 0, and
// whether that pair would be connected from "below" or "above".  There are 4 choices
// for each of the 2*n positions, making it O(4^(2n)) with maximum stack depth 6*n.
//
// When choosing to "close" at position k, it locates the matching "open" at k',
// and computes the distance m = k - k' + 1.  If this m has already been placed,
// closing at position k is not possible.
//
// When the number of open pairs reaches n, opening new pairs is no longer possible.
// Observing this constraint greatly prunes the search tree.
//
// The matching "open" at k' is very easy to find using two auxiliary stacks of
// currently open pairs, one for "below" and one for "above".
//
//
// DEDUPLICATION
//
// To dedup the Left <-> Right reversal symmetry, (1, 1) is placed in pos <= n.
//
// Many, but not all, top <-> bottom twins are deduped by forcing the pair in
// position 0 to be connected from below.
//
// Remaining duplicates are eliminated by storing all solutions in memory,
// with a final sort and count.  Fortunately, the number of solutions to the
// planar Langford problem is quite small, so this is feasible.
//
//
// IMPLEMENTATION TRICKS
//
// A nice boost in performance is realized through the use of a single 64-bit
// integer to encode the positions of *all* currently open pairs;  in this compact
// representation, we can quickly "pop" the position of the most recently open pair
// by using the operations
//
//     ffsl(x)       if x is non-zero, return one plus the index of the least
//                   significant bit of x;  if x is zero, return zero;
//                   result range 0..64
//
//     x &= (x-1)    clear the least signifficant 1-bit of x
//
//
// EXAMPLE OUTPUT
//
//     1488184396210 Solving Planar Langford for n = 3
//     1488184396268 Result 1 for n = 3 MATCHES previously published result and took 58 milliseconds.
//     ...
//     1488184397165 Solving Planar Langford for n = 19
//     1488184414041 Result 2384 for n = 19 MATCHES previously published result and took 16876 milliseconds.
//     ...
//
// The first number on each output line is a unix timestamp, i.e., milliseconds elapsed
// since Jan 1, 1970 GMT.  You may convert it to human-readable datetime using python,
// as follows.
//
// 1) Start "python"
// 2) Type "import time" and press enter
// 3) Type "time.localtime(1488184414041 / 1000.0)" and press enter
//
// The result is a decoding of unix timestamp 1488184414041 in your local time zone:
//
//     time.struct_time(tm_year=2017, tm_mon=2, tm_mday=27, tm_hour=0, tm_min=33,
//         tm_sec=34, tm_wday=0, tm_yday=58, tm_isdst=0)
//
//
// PRINTING ALL SOLUTION SEQUENCES
//
// If you wish all solutions sequences printed, change 'kPrint' below to 'true',
// and recompile.
//
//
// ACHIEVEMENTS
//
// On March 2, 2017 at 11:15pm PST this program computed PL(2, 27) after ~91.5 hours
// of work on a 22-core Xeon E5-2699v4 (14nm, early 2016).
//
//     1488195648532 Solving Planar Langford for n = 27
//     1488525305820 Result 426502 for n = 27 MATCHES previously published result and took 329657288 milliseconds.
//
// As of 2017-03-19 this algorithm is outperformed by the technique in
//
//     https://github.com/boris-dimitrov/z5_langford/blob/master/langford.cpp
// 
// which appears to produce results in 30% less time.  Both algorithms have
// exponential complexity O(3.38^n).
//
// SEE ALSO
//
// There is a variant of this program that runs on GPUs.  Performance is comparable between a single
// Titan X Pascal GPU (16nm process, mid 2016) and a 22-core Xeon E5-2699v4 (14nm, early 2016).
//
//     https://github.com/boris-dimitrov/z4_planar_langford_multigpu


#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
using namespace std;

// 2^n-1 works best for the silly modulus hash thingy
constexpr int kMaxThreads = 511;

// to avoid integer overflow, n should not exceed this constant
constexpr int kMaxN = 31;

// set to "true" if you want each solution printed
constexpr bool kPrint = false;

static_assert(sizeof(int64_t) == 8, "int64_t is not 8 bytes");
static_assert(sizeof(int32_t) == 4, "int32_t is not 4 bytes");
static_assert(sizeof(int8_t) == 1, "int64_t is not 1 byte");

constexpr int64_t lsb = 1;
constexpr int32_t lsb32 = 1;

template <int n>
using Positions = array<int8_t, n>;

template <int n>
using Results = vector<Positions<n>>;

template <int n>
void print(const Positions<n>& pos);

template <int n>
void dfs(Results<n>& results, const int num_threads, const int thread_id, mutex& mtx) {
    constexpr int two_n = 2 * n;
    constexpr int two_n_less_1 = 2 * n - 1;
    constexpr int64_t msb = lsb << (int64_t)(n - 1);
    constexpr int64_t nn1 = lsb << (two_n - 1);
    // initially none of the numbers 1, 2, ..., n have been placed;
    // this is represented by setting bits 0..n-1 to 1 in avail
    int32_t availability[2 * n + 1];
    availability[0] = msb | (msb - 1);
    // let pos[m] be the position of the closing m+1, for each m
    Positions<n> pos;
    // open[2*k+2] represents the nested open-from-above pairs in positions 0...k
    // open[2*k+3] is same for below
    int64_t open[4*n + 2];
    open[0] = 0;
    open[1] = 0;
    int8_t k, m, d, num_open;
    // there are 2*n positions with 3 decisions per position and 4 bytes per decision on the stack
    int8_t stack[24 * n];
    // the size of the arrays above add up to ~2KB for n=32
    // this bodes well for fitting tousands of threads inside on-chip memory
    int top = 0;
    // lambdas look better but... macros still outperform them
    #define push(k, m, d, num_open) do { \
        stack[top++] = k; \
        stack[top++] = m; \
        stack[top++] = d; \
        stack[top++] = num_open; \
    } while (0)
    #define pop(k, m, d, num_open) do { \
        num_open = stack[--top]; \
        d = stack[--top]; \
        m = stack[--top]; \
        k = stack[--top]; \
    } while (0)
    // every solution starts out by opening a below-pair at position 0
    push(0, -1, 0, 0);
    while (top) {
        pop(k, m, d, num_open);
        int64_t* openings = open + 2 * k + 2;
        openings[0] = openings[-2];
        openings[1] = openings[-1];
        int32_t avail = availability[k];
        // On CPU, this macro trick improves perf over 10% by letting the compiler
        // take advantage of the fact that d can only be 0 or 1.
        // Makes no difference on GPU.
        #define place_macro(d) do { \
            if (m>=0) { \
                pos[m] = k; \
                avail ^= (lsb32 << m); \
                openings[d] &= (openings[d] - 1); \
            } else { \
                openings[d] |= nn1 >> k; \
                ++num_open; \
            } \
        } while (0)
        if (d) {
            place_macro(1);
        } else {
            place_macro(0);
        }
        ++k;
        availability[k] = avail;
        if (k == two_n) {
            mtx.lock();
            results.push_back(pos);
            mtx.unlock();
        } else {
            // A super-naive way to divide the work across threads.  A hash of the current state at k_limit
            // determines whether the current thread should be pursuing a completion from this state or not.
            // The depth k_limit is chosen empirically to be both shallow enough so it's quick to reach and
            // deep enough to allow plenty of concurrency. This seems to work remarkably well in practice.
            constexpr int8_t k_limit = (n > 19 ? (8 + (n / 3)) : (n - 5));
            if (kMaxThreads > 1 &&
                k == k_limit &&
                // multiply by a nice Mersenne prime to divide the work evenly across the threads... it works well...
                uint64_t(131071 * (openings[1] - openings[0]) + avail) % kMaxThreads != thread_id) {
                // some other thread will work on this
                continue;
            }
            // Now push on the stack the the children of the current node in the search tree.
            int8_t offset = k - two_n - 2;
            for (d=0; d<2; ++d) {
                if (openings[d]) { // if there is an opening, try closing it
                    // let m be the distance to the opening, less 1
                    m = offset + __builtin_ffsll(openings[d]);
                    // m could be -1 when the opening was at k-1
                    // only m from 0..n-1 are worth pursuing
                    if (((unsigned)m < n) && ((avail >> m) & 1)) {
                        if (m || k <= n) { // this dedups L <==> R reversal twins
                            push(k, m, d, num_open);
                        }
                    }
                }
            }
            if (num_open < n) {
                push(k, -1, 1, num_open);
                push(k, -1, 0, num_open);
            }
        }
    }
}

// Sort the vector of solution sequences and count the unique ones.
// Optionally print each unique one.
template <int n>
int64_t unique_count(Results<n> &results) {
    int64_t total = results.size();
    int64_t unique = total;
    sort(results.begin(), results.end());
    if (kPrint && total) {
        print<n>(results[0]);
    }
    for (int i=1; i<total; ++i) {
        if (results[i] == results[i-1]) {
            --unique;
        } else if (kPrint) {
            print<n>(results[i]);
        }
    }
    return unique;
}

// This is the main function of the sequential algorithm.
template <int n>
int solve() {
    if (n <= 0 || n > kMaxN || n % 4 == 1 || n % 4 == 2) {
        return 0;
    }
    Results<n> results;
    int num_running = kMaxThreads;
    mutex mtx;
    for (int thread_id=0;  thread_id < kMaxThreads;  ++thread_id) {
        auto thread_func = [&](int thread_id) {
            dfs<n>(results, kMaxThreads, thread_id, mtx);
            mtx.lock();
            --num_running;
            mtx.unlock();
        };
        thread(thread_func, thread_id).detach();
    }
    bool done = false;
    while (!done) {
        this_thread::sleep_for(chrono::milliseconds(50));
        mtx.lock();
        done = (num_running == 0);
        mtx.unlock();
    }
    return unique_count<n>(results);
}


// ----------------------------- crux of solution ends here -------------------------------
// The rest is boring utilities for pretty printing, argument parsing, validation, etc.
// ----------------------------------------------------------------------------------------

// Return number of milliseconds elapsed since Jan 1, 1970 00:00 GMT.
long unixtime() {
    // There is the unix way, the navy way, and the C++11 way... apparently.
    using namespace chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    // Using steady_clock instead of system_clock above produces comically incorrect results.
    // Probably steady_clock has the wrong epoch start.
}

void init_known_results(int64_t (&known_results)[64]) {
    for (int i=0;  i<64; ++i) {
        known_results[i] = 0;
    }
    // There are no published results for n >= 29
    for (int i = 29;  i<64;  ++i) {
        if (i % 4 == 3 || i % 4 == 0) {
            known_results[i] = -1;
        }
    }
    known_results[3]  = 1;
    known_results[4]  = 0;
    known_results[7]  = 0;
    known_results[8]  = 4;
    known_results[11] = 16;
    known_results[12] = 40;
    known_results[15] = 194;
    known_results[16] = 274;
    known_results[19] = 2384;
    known_results[20] = 4719;
    known_results[23] = 31856;
    known_results[24] = 62124;
    known_results[27] = 426502;
    known_results[28] = 817717;
}

template <int n>
void print(const Positions<n>& pos) {
    cout << unixtime() << " Sequence ";
    int s[2 * n];
    for (int i=0; i<2*n; ++i) {
        s[i] = -1;
    }
    for (int m=1;  m<=n;  ++m) {
        int k2 = pos[m-1];
        int k1 = k2 - m - 1;
        assert(0 <= k1);
        assert(k2 < 2*n);
        assert(s[k1] == -1);
        assert(s[k2] == -1);
        s[k1] = s[k2] = m;
    }
    for (int i=0;  i<2*n;  ++i) {
        const int64_t m = s[i];
        assert(0 <= m);
        assert(m <= n);
        cout << setw(3) << m;
    }
    cout << "\n";
}

template <int n>
void run(const int64_t* known_results) {
    auto t_start = unixtime();
    cout << t_start << " Solving Planar Langford for n = " << n << "\n";
    cout << flush;
    int cnt = solve<n>();
    auto t_end = unixtime();
    cout << t_end << " Result " << cnt << " for n = " << n;
    if (n < 0 || n >= 64 || known_results[n] == -1) {
        cout << " is NEW";
    } else if (known_results[n] == cnt) {
        cout << " MATCHES previously published result";
    } else {
        cout << " MISMATCHES previously published result " << known_results[n];
    }
    cout << " and took " << (t_end - t_start) << " milliseconds to compute.\n";
    cout << flush;
}

int main(int argc, char **argv) {
    int64_t known_results[64];
    init_known_results(known_results);
    run<3>(known_results);
    run<4>(known_results);
    run<7>(known_results);
    run<8>(known_results);
    run<11>(known_results);
    run<12>(known_results);
    run<15>(known_results);
    run<16>(known_results);
    run<19>(known_results);
    run<20>(known_results);
    run<23>(known_results);
    run<24>(known_results);
    run<27>(known_results);
    run<28>(known_results);
    return 0;
}
