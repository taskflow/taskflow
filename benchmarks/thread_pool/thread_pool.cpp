#include <taskflow/taskflow.hpp>
#include "ThreadPool.hpp"

ThreadPool* ThreadPool::singleton = nullptr;
std::mutex ThreadPool::singleton_mutex;
tf::Executor executor;

class ChronoTimer {
public:
	ChronoTimer(void) {
	}
	void start(void){
		startTime = std::chrono::high_resolution_clock::now();
	}
	void finish(std::string msg){
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - startTime;
		printf("%s : %f ms\n",msg.c_str(), elapsed.count() * 1000);
	}
private:
	std::chrono::high_resolution_clock::time_point startTime;
};

float benchFunc(uint64_t loopLen){
	float acc = 0;
	for (uint64_t k = 0; k < loopLen; ++k)
		acc += k;
  return acc;
}

void bench(uint32_t iter){
	printf("Benchmark with %d iterations\n",iter);
	const uint64_t num_blocks = 1000;
	const uint64_t loopLen = 100;
	ChronoTimer timer;
	ThreadPool *pool = ThreadPool::get();

	timer.start();
	for (uint64_t it = 0; it < iter; ++it) {
		tf::Taskflow taskflow;
		tf::Task node[num_blocks];
		for (uint64_t i = 0; i < num_blocks; i++)
			node[i] = taskflow.placeholder();
		for (uint64_t i = 0; i < num_blocks; i++) {
			node[i].work([=]() {
				benchFunc(loopLen);
			});
		}
		executor.run(taskflow).wait();
	}
	timer.finish("taskflow: time in ms: ");

	timer.start();
	for (uint64_t it = 0; it < iter; ++it) {
		std::vector<std::future<int>> results;
		for (uint64_t i = 0; i < num_blocks; i++) {
			results.emplace_back(pool->enqueue([=]() {
				benchFunc(loopLen);

				return 0;
			}));
		}
		for(auto& result : results)
		{
			result.get();
		}
	}
	timer.finish("threadpool: time in ms: ");
}

int main() {
	for (uint32_t i = 0; i < 5; ++i)
		bench(100);
	for (uint32_t i = 0; i < 5; ++i)
		bench(50);
	for (uint32_t i = 0; i < 5; ++i)
		bench(20);
	for (uint32_t i = 0; i < 5; ++i)
		bench(10);
	for (uint32_t i = 0; i < 5; ++i)
		bench(5);
}
