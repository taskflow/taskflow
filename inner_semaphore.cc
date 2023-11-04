#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

#include <taskflow/taskflow.hpp>

constexpr int count_limit = 10000;

std::mutex mtx;
int32_t count_mtx;

void ping_lock() {
    while (count_mtx <= count_limit) {
        std::lock_guard<std::mutex> lk(mtx);
        ++count_mtx;
    }
}

void pong_lock() {
    while (count_mtx < count_limit) {
        std::lock_guard<std::mutex> lk(mtx);
        ++count_mtx;
    }
}

int main() {
    {
        tf::Semaphore se(1);
        tf::Executor executor(2);
        tf::Taskflow taskflow;

        int32_t count = 0;

        taskflow.emplace(
                [&](tf::Runtime& rt) {
                    while (count < count_limit) {
                        rt.acquire(se);
                        ++count;
                        rt.release(se);
                    }
                },
                [&](tf::Runtime& rt) {
                    while (count <= count_limit) {
                        rt.acquire(se);
                        ++count;
                        rt.release(se);
                    }
                });

        auto beg = std::chrono::high_resolution_clock::now();
        executor.run(taskflow).wait();
        executor.wait_for_all();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "taskflow semaphore time cost : "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() << "us." << std::endl;
    }
    {
        auto beg = std::chrono::high_resolution_clock::now();
        std::thread t1(ping_lock);
        std::thread t2(pong_lock);
        t1.join();
        t2.join();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "mutex lock guard time cost : "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() << "us." << std::endl;
    }
}
