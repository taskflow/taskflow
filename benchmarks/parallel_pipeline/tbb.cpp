#include "parallel_pipeline.hpp"
#include <tbb/pipeline.h>
#include <tbb/tick_count.h>
#include <tbb/tbb_allocator.h>
#include <tbb/global_control.h>
#include <fstream>
#include <iostream>
#include <vector>
//#include "../../3rd-party/tbb/examples/common/utility/utility.h"
//#include "../../3rd-party/tbb/examples/common/utility/get_default_num_threads.h"

//std::vector<double> result;
size_t i = 0;

// Filter for one filter only 
class MyFunc {
public:
  MyFunc(size_t size) : s(size) {}
    
  ~MyFunc(){}

  size_t s;
  
  void operator()(tbb::flow_control& fc) const {
    int retval = 0;
      
    if (i++ == s) {
      fc.stop();
    }
    else {
      retval = retval + 1;
    }
  }
};

// Filter 1
class MyInputFunc {
public:
  MyInputFunc(size_t size) : s(size) {}
    
  ~MyInputFunc(){}

  size_t s;
  
  int operator()(tbb::flow_control& fc) const {
    int retval = 0;
          
    if (i++ == s) {
      fc.stop();
      return -1;
    }
    else {
      return retval + 1;
    }
  }
};

// Filter 2 
class MyTransformFunc1 {
public:
  int operator()(int input) const {
    int retval = 0;
    retval = input << 1;
    return retval;  
  }
};

// Filter 3
class MyTransformFunc2 {
public:
  int operator()(int input) const {
    int retval = 0;
    retval = input + 999;
    return retval;
  }
};

// Filter 4 
class MyTransformFunc3 {
public:
  int operator()(int input) const {
    int retval = 0;
    retval = input << 1;
    return retval;
  }
};

// Filter 5 
class MyTransformFunc4 {
public:
  int operator()(int input) const {
    int retval = 0;
    retval = input - 792;
    return retval;
  }
};

// Filter 6
class MyTransformFunc5 {
public:
  int operator()(int input) const {
    int retval = 0;
    retval = input * 35;
    return retval;
  }
};

// Filter 7 
class MyTransformFunc6 {
public:
  int operator()(int input) const {
    int retval = 0;
    retval = input >> 1;
    return retval;
  }
};

// Filter 8
class MyTransformFunc7 {
public:
  int operator()(int input) const {
    int retval = 0;
    retval = input * input * input;
    return retval;
  }
};

// Filter 9 
class MyTransformFunc8 {
public:
  int operator()(int input) const {
    int retval = 0;
    retval = input >> 2;
    return retval;
  }
};

// Filter last 
class MyOutputFunc {
public:
  MyOutputFunc(){}
  void operator()(int input) const {
    int retval = 0;
    retval = input + 99999;
    //result.emplace_back(retval); 
  }
};


// parallel_pipeline_tbb_1_pipe
void parallel_pipeline_tbb_1_pipe(std::string pipes, unsigned num_lines, size_t size) {
    tbb::parallel_pipeline(
      num_lines,
      tbb::make_filter<void, void>(
        tbb::filter::serial_in_order, MyFunc(size))  
    );
}

// parallel_pipeline_tbb_2_pipes
void parallel_pipeline_tbb_2_pipes(std::string pipes, unsigned num_lines, size_t size) {
    tbb::parallel_pipeline(
      num_lines,
      tbb::make_filter<void, int>(
        tbb::filter::serial_in_order, MyInputFunc(size))  &
      tbb::make_filter<int, void>(
        pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyOutputFunc())
    );
}

// parallel_pipeline_tbb_3_pipes
void parallel_pipeline_tbb_3_pipes(std::string pipes, unsigned num_lines, size_t size) {
    tbb::parallel_pipeline(
      num_lines,
      tbb::make_filter<void, int>(
        tbb::filter::serial_in_order, MyInputFunc(size))  &
      tbb::make_filter<int, int>(
        pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc1()) &
      tbb::make_filter<int, void>(
        pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyOutputFunc())
    );
}

// parallel_pipeline_tbb_4_pipes
void parallel_pipeline_tbb_4_pipes(std::string pipes, unsigned num_lines, size_t size) {
    tbb::parallel_pipeline(
      num_lines,
      tbb::make_filter<void, int>(
        tbb::filter::serial_in_order, MyInputFunc(size))  &
      tbb::make_filter<int, int>(
        pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc1()) &
      tbb::make_filter<int, int>(
        pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc2()) &
      tbb::make_filter<int, void>(
        pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyOutputFunc())
    );
}

// parallel_pipeline_tbb_5_pipes
void parallel_pipeline_tbb_5_pipes(std::string pipes, unsigned num_lines, size_t size) {
    tbb::parallel_pipeline(
      num_lines,
      tbb::make_filter<void, int>(
        tbb::filter::serial_in_order, MyInputFunc(size))  &
      tbb::make_filter<int, int>(
        pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc1()) &
      tbb::make_filter<int, int>(
        pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc2()) &
      tbb::make_filter<int, int>(
        pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc3()) &
      tbb::make_filter<int, void>(
        pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyOutputFunc())
    );
}

// parallel_pipeline_tbb_6_pipes
void parallel_pipeline_tbb_6_pipes(std::string pipes, unsigned num_lines, size_t size) {
    tbb::parallel_pipeline(
      num_lines,
      tbb::make_filter<void, int>(
        tbb::filter::serial_in_order, MyInputFunc(size))  &
      tbb::make_filter<int, int>(
        pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc1()) &
      tbb::make_filter<int, int>(
        pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc2()) &
      tbb::make_filter<int, int>(
        pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc3()) &
      tbb::make_filter<int, int>(
        pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc4()) &
      tbb::make_filter<int, void>(
        pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyOutputFunc())
    );
}

// parallel_pipeline_tbb_7_pipes
void parallel_pipeline_tbb_7_pipes(std::string pipes, unsigned num_lines, size_t size) {
    tbb::parallel_pipeline(
      num_lines,
      tbb::make_filter<void, int>(
        tbb::filter::serial_in_order, MyInputFunc(size))  &
      tbb::make_filter<int, int>(
        pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc1()) &
      tbb::make_filter<int, int>(
        pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc2()) &
      tbb::make_filter<int, int>(
        pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc3()) &
      tbb::make_filter<int, int>(
        pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc4()) &
      tbb::make_filter<int, int>(
        pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc5()) &
      tbb::make_filter<int, void>(
        pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyOutputFunc())
    );
}

// parallel_pipeline_tbb_8_pipes
void parallel_pipeline_tbb_8_pipes(std::string pipes, unsigned num_lines, size_t size) {
    tbb::parallel_pipeline(
      num_lines,
      tbb::make_filter<void, int>(
        tbb::filter::serial_in_order, MyInputFunc(size))  &
      tbb::make_filter<int, int>(
        pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc1()) &
      tbb::make_filter<int, int>(
        pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc2()) &
      tbb::make_filter<int, int>(
        pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc3()) &
      tbb::make_filter<int, int>(
        pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc4()) &
      tbb::make_filter<int, int>(
        pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc5()) &
      tbb::make_filter<int, int>(
        pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc6()) &
      tbb::make_filter<int, void>(
        pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyOutputFunc())
    );
}

// parallel_pipeline_tbb_9_pipes
void parallel_pipeline_tbb_9_pipes(std::string pipes, unsigned num_lines, size_t size) {
    tbb::parallel_pipeline(
      num_lines,
      tbb::make_filter<void, int>(
        tbb::filter::serial_in_order, MyInputFunc(size))  &
      tbb::make_filter<int, int>(
        pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc1()) &
      tbb::make_filter<int, int>(
        pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc2()) &
      tbb::make_filter<int, int>(
        pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc3()) &
      tbb::make_filter<int, int>(
        pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc4()) &
      tbb::make_filter<int, int>(
        pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc5()) &
      tbb::make_filter<int, int>(
        pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc6()) &
      tbb::make_filter<int, int>(
        pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc7()) &
      tbb::make_filter<int, void>(
        pipes[8] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyOutputFunc())
    );
}

// parallel_pipeline_tbb_10_pipes
void parallel_pipeline_tbb_10_pipes(std::string pipes, unsigned num_lines, size_t size) {
    tbb::parallel_pipeline(
      num_lines,
      tbb::make_filter<void, int>(
        tbb::filter::serial_in_order, MyInputFunc(size))  &
      tbb::make_filter<int, int>(
        pipes[1] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc1()) &
      tbb::make_filter<int, int>(
        pipes[2] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc2()) &
      tbb::make_filter<int, int>(
        pipes[3] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc3()) &
      tbb::make_filter<int, int>(
        pipes[4] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc4()) &
      tbb::make_filter<int, int>(
        pipes[5] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc5()) &
      tbb::make_filter<int, int>(
        pipes[6] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc6()) &
      tbb::make_filter<int, int>(
        pipes[7] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc7()) &
      tbb::make_filter<int, int>(
        pipes[8] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyTransformFunc8()) &
      tbb::make_filter<int, void>(
        pipes[9] == 's' ? tbb::filter::serial_in_order : tbb::filter::parallel, MyOutputFunc())
    );
}


std::chrono::microseconds measure_time_tbb(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  //result.clear();
  //utility::thread_number_range threads( utility::get_default_num_threads, 0);                                
  tbb::global_control c(tbb::global_control::max_allowed_parallelism, num_threads);
  
  auto beg = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();
  i = 0; 
  switch(pipes.size()) {
    case 1:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_1_pipe(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;
    
    case 2:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_2_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 3:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_3_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;
    
    case 4:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_4_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 5:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_5_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;
    
    case 6:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_6_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;
    
    case 7:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_7_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 8:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_8_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;
    
    case 9:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_9_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;

    case 10:
      beg = std::chrono::high_resolution_clock::now();
      parallel_pipeline_tbb_10_pipes(pipes, num_lines, size);
      end = std::chrono::high_resolution_clock::now();
      break;
  }
  //std::ofstream outputfile;
  //outputfile.open("./tbb_result.txt", std::ofstream::app);
  //for (auto r:result) {
  //  outputfile << r << '\n';
  //}
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
