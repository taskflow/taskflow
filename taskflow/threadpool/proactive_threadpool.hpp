// 
// contributed by Guannan
// 
// ProactiveThreadpool schedules independent jobs in a greedy manner.
// Whenever a job is inserted into the threadpool, the threadpool will check if there
// are any spare threads available. The spare thread will be woken through its local 
// condition variable. The new job will be directly moved into
// this thread instead of pushed at the back of the pending queue.

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

namespace tf {


// template < template<typename...> class FuncType >

class ProactiveThreadpool {

  using UnitTask = std::function<void()>;

  public:

    ProactiveThreadpool(unsigned N){
      spawn(N);
    }

    ~ProactiveThreadpool(){
      shutdown();
    }

    size_t num_tasks() const{ return _task_queue.size(); }

    size_t num_workers() const { return _threads.size();  }

    void shutdown(){

      { 
        std::unique_lock<std::mutex> lock(_mutex);
        _shutdown = true;
        _empty.wait(lock, [this](){ return _task_queue.empty(); });
        _exiting = true;
        
        for(auto w : _workers){
          w->ready = true;
          w->task = nullptr;
          w->cv.notify_one();
        }
      }
    
      for(auto& t : _threads){
        t.join();
      } 
      _workers.clear();
      _threads.clear();  
      
      _shutdown = false;
      _exiting = false;
    }
    
    bool is_worker() const{
      std::scoped_lock<std::mutex> lock(_mutex);
      return _worker_ids.find(std::this_thread::get_id()) != _worker_ids.end();
    }

    void spawn(unsigned N){
      if(is_worker()){
        throw std::runtime_error("Worker thread cannot spawn threads");
      }

      for(size_t i=0; i<N; ++i){
      
        _threads.emplace_back([this]()->void{
          
          {
            std::scoped_lock<std::mutex> lock(_mutex);
            _worker_ids.insert(std::this_thread::get_id());
          }

          Worker w;
          UnitTask t; 
          std::unique_lock<std::mutex> lock(_mutex);
          while(!_exiting){
            if(_task_queue.empty()){
              w.ready = false;
              _workers.push_back(&w);

              if(_workers.size() == num_workers()){
                _complete.notify_one();
              }

              w.cv.wait(lock, [&w](){ return w.ready; });              
              t = std::move(w.task);
            }
            else{
              t= std::move(_task_queue.front());
              _task_queue.pop_front();
              
              if(_task_queue.empty() && _shutdown) {
                _empty.notify_one();
              }  
            } 

            if(t){
              _mutex.unlock();
              t(); // run task in parallel
              t = nullptr;
              _mutex.lock();
            }
          }
    
        });     

      } 
    }

  template <typename C>
  void silent_async(C&& c){

    UnitTask t {std::forward<C>(c)};

    //no worker thread available
    if(num_workers() == 0){
      t();
      return;
    }

    std::unique_lock<std::mutex> lock(_mutex);
    if(_workers.empty()){
      _task_queue.push_back(std::move(t));
    } 
    else{
      Worker* w = _workers.back();
      _workers.pop_back();
      w->ready = true;
      w->task = std::move(t);
      w->cv.notify_one();   
    }

  }

  template <typename C>
  auto async(C&& c){

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
      std::unique_lock<std::mutex> lock(_mutex);     
      if constexpr(std::is_same_v<void, R>){
        // all workers are busy.
        if(_workers.empty()){
          _task_queue.emplace_back(
            [p = MoveOnCopy(std::move(p)), c = std::forward<C>(c)]() mutable {
              c();
              p.get().set_value(); 
            }
          );
        }
        // Got an idle work
        else{
          Worker* w = _workers.back();
          _workers.pop_back();
          w->ready = true;
          w->task = [p = MoveOnCopy(std::move(p)), c = std::forward<C>(c)]() mutable {
            c();
            p.get().set_value(); 
          };
          w->cv.notify_one(); 
        }
      }
      else{

        if(_workers.empty()){
          _task_queue.emplace_back(
            [p = MoveOnCopy(std::move(p)), c = std::forward<C>(c)]() mutable {
              p.get().set_value(c());
              return; 
            }
          );
        }
        else{
          Worker* w = _workers.back();
          _workers.pop_back();
          w->ready = true;
          w->task = [p = MoveOnCopy(std::move(p)), c = std::forward<C>(c)]() mutable {
            p.get().set_value(c()); 
            return;
          };
          w->cv.notify_one(); 
        }
      }
    }
  
    return fu;
  
  }

  void wait_for_all(){

    if(is_worker()){
      throw std::runtime_error("Worker thread cannot wait for all");
    }

    std::unique_lock<std::mutex> lock(_mutex);
    _complete.wait(lock, [this](){ return _workers.size() == num_workers(); }); 

  }


  private:
    
    template <typename T>
    struct MoveOnCopy{
    
      MoveOnCopy(T&& rhs): object(std::move(rhs)) {}
      MoveOnCopy(const MoveOnCopy& other) : object(std::move(other.object)) {}
    
      T& get() {return object; }
      
      mutable T object;
    };


    struct Worker{
      std::condition_variable cv;
      UnitTask task;
      bool ready;
    };

    mutable std::mutex _mutex;

    std::condition_variable _empty;
    std::condition_variable _complete;

    std::deque<UnitTask> _task_queue;
    std::vector<std::thread> _threads;
    std::vector<Worker*> _workers; 
    std::unordered_set<std::thread::id> _worker_ids;    

    bool _exiting {false};
    bool _shutdown{false};

};



};



