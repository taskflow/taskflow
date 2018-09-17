#pragma once

#include <iostream>
#include <mutex>
#include <deque>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <list>
#include <forward_list>
#include <numeric>
#include <iomanip>
#include <cassert>

#include "simple_threadpool.hpp"
#include "proactive_threadpool.hpp"
#include "speculative_threadpool.hpp"
#include "privatized_threadpool.hpp"


namespace tf {

using ProactiveThreadpool = proactive_threadpool::BasicProactiveThreadpool<std::function>;
using SpeculativeThreadpool = speculative_threadpool::BasicSpeculativeThreadpool<std::function>;
using PrivatizedThreadpool = privatized_threadpool::BasicPrivatizedThreadpool<std::function>;

};  // namespace tf. ----------------------------------------------------------

