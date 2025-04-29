#pragma once

#include "../taskflow.hpp"
#include "cuda_graph.hpp"
#include "cuda_graph_exec.hpp"
#include "algorithm/single_task.hpp"

/**
@file taskflow/cuda/cudaflow.hpp
@brief cudaFlow include file
*/

namespace tf {

/**
@brief default smart pointer type to manage a `cudaGraph_t` object with unique ownership
*/
using cudaGraph = cudaGraphBase<cudaGraphCreator, cudaGraphDeleter>;

/**
@brief default smart pointer type to manage a `cudaGraphExec_t` object with unique ownership
*/
using cudaGraphExec = cudaGraphExecBase<cudaGraphExecCreator, cudaGraphExecDeleter>;

}  // end of namespace tf -----------------------------------------------------


