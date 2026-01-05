module;

#include <taskflow/taskflow.hpp>

export module tf;

export import :algorithm;
export import :core;
export import :cuda;
export import :utility;

export namespace tf {
    using tf::version;
}
