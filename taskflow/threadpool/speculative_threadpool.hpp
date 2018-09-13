// 2018/09/12 - created by Tsung-Wei Huang and Chun-Xun Lin
//
// Speculative threadpool is similar to proactive threadpool except
// each thread will speculatively move a new task to its local worker
// data structure to reduce extract hit to the task queue.
// This can save time from locking the mutex during dynamic tasking.

#pragma once

#include <iostream>
#include <functional>
#include <vector>
#include <mutex>
#include <deque>
#include <thread>
#include <stdexcept>
#include <condition_variable>
#include <memory>
#include <future>
#include <unordered_set>
#include <unordered_map>


namespace speculative_threadpool {

template <typename T>
struct MoC {

  MoC(T&& rhs): object(std::move(rhs)) {}
  MoC(const MoC& other) : object(std::move(other.object)) {}

  T& get() {return object; }
  
  mutable T object;
};

// Class: BasicSpeculativeThreadpool
template < template<typename...> class Func >
class BasicSpeculativeThreadpool {

  using TaskType = Func<void()>;

  struct Worker{
    std::condition_variable cv;
    TaskType task;
    bool ready {false};
  };

  public:

    BasicSpeculativeThreadpool(unsigned);
    ~BasicSpeculativeThreadpool();

    size_t num_tasks() const;
    size_t num_workers() const;

    bool is_owner() const;

    void shutdown();
    void spawn(unsigned);
    void wait_for_all();

    template <typename C>
    void silent_async(C&&);

    template <typename C>
    auto async(C&&);

  private:

    mutable std::mutex _mutex;

    std::condition_variable _empty_cv;

    std::deque<TaskType> _task_queue;
    std::vector<std::thread> _threads;
    std::vector<Worker*> _idlers; 
    std::unordered_map<std::thread::id, Worker*> _worker_maps;    
    
    const std::thread::id _owner {std::this_thread::get_id()};

    bool _exiting      {false};
    bool _wait_for_all {false};

    std::vector<std::unique_ptr<Worker>> _works;

    auto _this_worker() const;

};  // class BasicSpeculativeThreadpool. --------------------------------------

// Constructor
template < template<typename...> class Func >
BasicSpeculativeThreadpool<Func>::BasicSpeculativeThreadpool(unsigned N){
  spawn(N);
}

// Destructor
template < template<typename...> class Func >
BasicSpeculativeThreadpool<Func>::~BasicSpeculativeThreadpool(){
  shutdown();
}

// Function: is_owner
template < template<typename...> class Func >
bool BasicSpeculativeThreadpool<Func>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_tasks
template < template<typename...> class Func >
size_t BasicSpeculativeThreadpool<Func>::num_tasks() const { 
  return _task_queue.size(); 
}

// Function: num_workers
template < template<typename...> class Func >
size_t BasicSpeculativeThreadpool<Func>::num_workers() const { 
  return _threads.size();  
}
    
// Function: _this_worker
template < template<typename...> class Func >
auto BasicSpeculativeThreadpool<Func>::_this_worker() const {
  auto id = std::this_thread::get_id();
  return _worker_maps.find(id);
}

// Function: shutdown
template < template<typename...> class Func >
void BasicSpeculativeThreadpool<Func>::shutdown(){

  if(!is_owner()){
    throw std::runtime_error("Worker thread cannot shut down the pool");
  }

  if(_threads.empty()) {
    return;
  }

  { 
    std::unique_lock lock(_mutex);
    _wait_for_all = true;
    while(_idlers.size() != num_workers()) {
      _empty_cv.wait(lock);
    }
    _exiting = true;
    
    for(auto w : _idlers){
      w->ready = true;
      w->task = nullptr;
      w->cv.notify_one();
    }
    _idlers.clear();
  }

  for(auto& t : _threads){
    t.join();
  } 
  _threads.clear();  

  _works.clear();
  _worker_maps.clear();
  
  _wait_for_all = false;
  _exiting = false;
  
  // task queue might have tasks added by threads outside this pool...
  //while(not _task_queue.empty()) {
  //  std::invoke(_task_queue.front());
  //  _task_queue.pop_front();
  //}
}

// Function: spawn 
template < template<typename...> class Func >
void BasicSpeculativeThreadpool<Func>::spawn(unsigned N) {

  if(! is_owner()){
    throw std::runtime_error("Worker thread cannot spawn threads");
  }

  // Wait untill all workers become idle if any
  if(!_threads.empty()){
    wait_for_all();
  }

  for(size_t i=0; i<N; ++i){
    _works.push_back(std::make_unique<Worker>());
  }
  
  const size_t sz = _threads.size();

  // Lock to synchronize all workers before creating _worker_mapss
  std::scoped_lock<std::mutex> lock(_mutex);

  for(size_t i=0; i<N; ++i){

    _threads.emplace_back([this, i=i+sz]() -> void {

       TaskType t {nullptr};
       Worker& w = *(_works[i]);

       std::unique_lock<std::mutex> lock(_mutex);

       while(!_exiting){
         if(_task_queue.empty()){
           w.ready = false;
           _idlers.push_back(&w);

           if(_wait_for_all && _idlers.size() == num_workers()){
             _empty_cv.notify_one();
           } 

           while(!w.ready) {
             w.cv.wait(lock);
           }

           t = std::move(w.task);
         }
         else{
           t = std::move(_task_queue.front());
           _task_queue.pop_front();
         } 

         if(t){
           _mutex.unlock();
           // speculation loop
           while(t) {
             t();
             t = std::move(w.task);
           }
           _mutex.lock();
         }
       } // End of while ------------------------------------------------------
    });     

    _worker_maps.insert({_threads.back().get_id(), _works[i+sz].get()});
  } // End of For ---------------------------------------------------------------------------------

}


// Function: async
template < template<typename...> class Func >
template <typename C>
auto BasicSpeculativeThreadpool<Func>::async(C&& c){

  using R = std::invoke_result_t<C>;

  std::promise<R> p;
  auto fu = p.get_future();
  
  // master thread
  if(num_workers() == 0){
    if constexpr(std::is_same_v<void, R>){
      c();
      p.set_value();
    }
    else{
      p.set_value(c());
    } 
  }
  // have worker(s)
  else{
    if constexpr(std::is_same_v<void, R>){
      
      // speculation
      if(std::this_thread::get_id() != _owner){
        auto iter = _this_worker();
        if(iter != _worker_maps.end() && iter->second->task == nullptr){
          iter->second->task =
            [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
              c();
              p.get().set_value(); 
            };
          return fu;
        }
      }

      std::scoped_lock lock(_mutex);     
      if(_idlers.empty()){
        _task_queue.emplace_back(
          [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
            c();
            p.get().set_value(); 
          }
        );
      }
      // Got an idle work
      else{
        Worker* w = _idlers.back();
        _idlers.pop_back();
        w->ready = true;
        w->task = [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
          c();
          p.get().set_value(); 
        };
        w->cv.notify_one(); 
      }
    }
    else{

      // speculation
      if(std::this_thread::get_id() != _owner){
        auto iter = _this_worker();
        if(iter != _worker_maps.end() && iter->second->task == nullptr){
          iter->second->task = 
            [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
              p.get().set_value(c());
            };
          return fu;
        }
      }

      std::scoped_lock lock(_mutex);     
      if(_idlers.empty()){
        _task_queue.emplace_back(
          [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
            p.get().set_value(c());
          }
        );
      }
      else{
        Worker* w = _idlers.back();
        _idlers.pop_back();
        w->ready = true;
        w->task = [p = MoC(std::move(p)), c = std::forward<C>(c)]() mutable {
          p.get().set_value(c()); 
        };
        w->cv.notify_one(); 
      }
    }
  }

  return fu;
}

template < template<typename...> class Func >
template <typename C>
void BasicSpeculativeThreadpool<Func>::silent_async(C&& c){

  TaskType t {std::forward<C>(c)};

  //no worker thread available
  if(num_workers() == 0){
    t();
    return;
  }

  // speculation
  if(std::this_thread::get_id() != _owner){
    auto iter = _this_worker();
    if(iter != _worker_maps.end() && iter->second->task == nullptr){
      iter->second->task = std::move(t);
      return ;
    }
  }

  std::scoped_lock lock(_mutex);
  if(_idlers.empty()){
    _task_queue.push_back(std::move(t));
  } 
  else{
    Worker* w = _idlers.back();
    _idlers.pop_back();
    w->ready = true;
    w->task = std::move(t);
    w->cv.notify_one();   
  }
}


// Function: wait_for_all
template < template<typename...> class Func >
void BasicSpeculativeThreadpool<Func>::wait_for_all() {

  if(!is_owner()){
    throw std::runtime_error("Worker thread cannot wait for all");
  }

  std::unique_lock lock(_mutex);
  _wait_for_all = true;
  while(_idlers.size() != num_workers()) {
    _empty_cv.wait(lock);
  }
  _wait_for_all = false;
}



};  // namespace speculative_threadpool. --------------------------------------





