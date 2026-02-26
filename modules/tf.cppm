module;

#include <taskflow/taskflow.hpp>

export module tf;

export import :algorithm;
export import :core;
#ifdef TF_BUILD_CUDA
export import :cuda;
#endif
export import :utility;

export namespace tf {
    using tf::version;
}
