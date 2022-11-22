#ifndef __RPR_DPE
#define __RPR_DPE

#include <string>
#include <ff/node.hpp>

namespace rpr {

	/* Global functions */
	int setup(const std::string &program_name);
	void cleanup();

	/* Control functions */
	void register_kernel(ff::ff_node &kernel, int kernel_id);
	void deregister_kernel(int kernel_id);
	int register_pipeline(ff::ff_node &pipeline, int pipeline_id);
	void deregister_pipeline(int pipeline_id);
	int register_farm(ff::ff_node &farm, int farm_id);
	void deregister_farm(int farm_id);
	int register_async(ff::ff_node &async, int async_id);
	void deregister_async(int async_id);

	/* Scheduling functions */
        std::string schedule_kernel(int kernel_id, const size_t=0);
        std::string schedule_pipeline(int pipeline_id, const size_t=0);
	std::string schedule_farm(int farm_id, const size_t=0);
	std::string schedule_async(int async_id, const size_t=0);
}
#endif
