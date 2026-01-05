module;

#include <taskflow/algorithm/algorithm.hpp>
#include <taskflow/algorithm/data_pipeline.hpp>
#include <taskflow/algorithm/find.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/algorithm/module.hpp>
#include <taskflow/algorithm/partitioner.hpp>
#include <taskflow/algorithm/pipeline.hpp>
#include <taskflow/algorithm/reduce.hpp>
#include <taskflow/algorithm/scan.hpp>
#include <taskflow/algorithm/sort.hpp>
#include <taskflow/algorithm/transform.hpp>

export module tf:algorithm;

export namespace tf {
    using tf::Algorithm;
    using tf::DataPipe;
    using tf::DataPipeline;
    using tf::PartitionerType;
    using tf::DefaultClosureWrapper;
    using tf::IsPartitioner;
    using tf::PartitionerBase;
    using tf::GuidedPartitioner;
    using tf::DynamicPartitioner;
    using tf::StaticPartitioner;
    using tf::RandomPartitioner;
    using tf::DefaultPartitioner;
    using tf::Pipeflow;
    using tf::PipeType;
    using tf::Pipe;
    using tf::Pipeline;
    using tf::ScalablePipeline;

    using tf::is_partitioner_v;

    using tf::make_data_pipe;
    using tf::make_find_if_task;
    using tf::make_for_each_task;
    using tf::make_module_task;
    using tf::make_reduce_task;
    using tf::make_transform_reduce_task;
    using tf::make_reduce_by_index_task;
    using tf::make_inclusive_scan_task;
    using tf::make_transform_inclusive_scan_task;
    using tf::make_exclusive_scan_task;
    using tf::make_transform_exclusive_scan_task;
    using tf::make_sort_task;
    using tf::make_transform_task;
}

export {
    using ::make_find_if_not_task;
    using ::make_min_element_task;
    using ::make_max_element_task;
    using ::make_for_each_index_task;
    using ::make_for_each_by_index_task;
}
