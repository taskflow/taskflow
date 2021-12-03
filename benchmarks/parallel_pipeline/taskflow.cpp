#include "parallel_pipeline.hpp"
#include <taskflow/taskflow.hpp> 
#include <taskflow/algorithm/pipeline.hpp>

// parallel_pipeline_taskflow_1_pipe
std::chrono::microseconds parallel_pipeline_taskflow_1_pipe(
  unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 1>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
        //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
      }
    }}
  ); 

  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_2_pipes
std::chrono::microseconds parallel_pipeline_taskflow_2_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 2>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2th pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //result.emplace_back(mybuffer[pf.line()][pf.pipe()]);
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }} 
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_3_pipes
std::chrono::microseconds parallel_pipeline_taskflow_3_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 3>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3th pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //result.emplace_back(mybuffer[pf.line()][pf.pipe()]);
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }} 
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_4_pipes
std::chrono::microseconds parallel_pipeline_taskflow_4_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 4>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //result.emplace_back(mybuffer[pf.line()][pf.pipe()]);
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }} 
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_5_pipes
std::chrono::microseconds parallel_pipeline_taskflow_5_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 5>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //result.emplace_back(mybuffer[pf.line()][pf.pipe()]);
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }} 
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_6_pipes
std::chrono::microseconds parallel_pipeline_taskflow_6_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 6>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] - 792;  
    }}, 
  
    // 6th pipe
    tf::Pipe{pipes[5] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //result.emplace_back(mybuffer[pf.line()][pf.pipe()]);
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }} 
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_7_pipes
std::chrono::microseconds parallel_pipeline_taskflow_7_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 7>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] - 792;  
    }}, 
  
    // 6th pipe
    tf::Pipe{pipes[5] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] * 35;   
    }},
     
    // 7th pipe
    tf::Pipe{pipes[6] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //result.emplace_back(mybuffer[pf.line()][pf.pipe()]);
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }} 
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_8_pipes
std::chrono::microseconds parallel_pipeline_taskflow_8_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  //std::ofstream outputfile;
  //outputfile.open("./tf_result.txt", std::ofstream::app);
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 28>> mybuffer(num_lines);
  //std::vector<double> result;

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] - 792;  
    }}, 
  
    // 6th pipe
    tf::Pipe{pipes[5] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] * 35;   
    }},
     
    // 7th pipe
    tf::Pipe{pipes[6] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 1;   
    }}, 

    // 8th pipe
    tf::Pipe{pipes[7] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //result.emplace_back(mybuffer[pf.line()][pf.pipe()]);
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }} 
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  //for(auto r:result) {
  //  outputfile << r << '\n';
  //}
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_9_pipes
std::chrono::microseconds parallel_pipeline_taskflow_9_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 9>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] - 792;  
    }}, 
  
    // 6th pipe
    tf::Pipe{pipes[5] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] * 35;   
    }},
     
    // 7th pipe
    tf::Pipe{pipes[6] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 1;   
    }}, 

    // 8th pipe
    tf::Pipe{pipes[7] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      int input = mybuffer[pf.line()][pf.pipe()-1];
      mybuffer[pf.line()][pf.pipe()] = input * input * input;   
    }}, 

    // 9th pipe
    tf::Pipe{pipes[8] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }}
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_10_pipes
std::chrono::microseconds parallel_pipeline_taskflow_10_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 10>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] - 792;  
    }}, 
  
    // 6th pipe
    tf::Pipe{pipes[5] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] * 35;   
    }},
     
    // 7th pipe
    tf::Pipe{pipes[6] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 1;   
    }}, 

    // 8th pipe
    tf::Pipe{pipes[7] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      int input = mybuffer[pf.line()][pf.pipe()-1];
      mybuffer[pf.line()][pf.pipe()] = input * input * input;   
    }}, 

    // 9th pipe
    tf::Pipe{pipes[8] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 2;   
    }},

    // 10th pipe
    tf::Pipe{pipes[9] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }}
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


// parallel_pipeline_taskflow_11_pipes
std::chrono::microseconds parallel_pipeline_taskflow_11_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 11>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] - 792;  
    }}, 
  
    // 6th pipe
    tf::Pipe{pipes[5] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] * 35;   
    }},
     
    // 7th pipe
    tf::Pipe{pipes[6] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 1;   
    }}, 

    // 8th pipe
    tf::Pipe{pipes[7] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      int input = mybuffer[pf.line()][pf.pipe()-1];
      mybuffer[pf.line()][pf.pipe()] = input * input * input;   
    }}, 

    // 9th pipe
    tf::Pipe{pipes[8] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 2;   
    }},

    // 10th pipe
    tf::Pipe{pipes[9] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(std::sqrt(mybuffer[pf.line()][pf.pipe() - 1]));
    }},

    // 11th pipe
    tf::Pipe{pipes[10] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }}
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_12_pipes
std::chrono::microseconds parallel_pipeline_taskflow_12_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 12>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] - 792;  
    }}, 
  
    // 6th pipe
    tf::Pipe{pipes[5] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] * 35;   
    }},
     
    // 7th pipe
    tf::Pipe{pipes[6] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 1;   
    }}, 

    // 8th pipe
    tf::Pipe{pipes[7] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      int input = mybuffer[pf.line()][pf.pipe()-1];
      mybuffer[pf.line()][pf.pipe()] = input * input * input;   
    }}, 

    // 9th pipe
    tf::Pipe{pipes[8] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 2;   
    }},

    // 10th pipe
    tf::Pipe{pipes[9] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(std::sqrt(mybuffer[pf.line()][pf.pipe() - 1]));
    }},

    // 11th pipe
    tf::Pipe{pipes[10] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(std::log(mybuffer[pf.line()][pf.pipe() - 1]));
    }},

    // 12th pipe
    tf::Pipe{pipes[11] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }}
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_13_pipes
std::chrono::microseconds parallel_pipeline_taskflow_13_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 13>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] - 792;  
    }}, 
  
    // 6th pipe
    tf::Pipe{pipes[5] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] * 35;   
    }},
     
    // 7th pipe
    tf::Pipe{pipes[6] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 1;   
    }}, 

    // 8th pipe
    tf::Pipe{pipes[7] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      int input = mybuffer[pf.line()][pf.pipe()-1];
      mybuffer[pf.line()][pf.pipe()] = input * input * input;   
    }}, 

    // 9th pipe
    tf::Pipe{pipes[8] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 2;   
    }},

    // 10th pipe
    tf::Pipe{pipes[9] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(std::sqrt(mybuffer[pf.line()][pf.pipe() - 1]));
    }},

    // 11th pipe
    tf::Pipe{pipes[10] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(std::log(mybuffer[pf.line()][pf.pipe() - 1]));
    }},

    // 12th pipe
    tf::Pipe{pipes[11] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 3;
    }},

    // 13th pipe
    tf::Pipe{pipes[12] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }}
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_14_pipes
std::chrono::microseconds parallel_pipeline_taskflow_14_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 14>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] - 792;  
    }}, 
  
    // 6th pipe
    tf::Pipe{pipes[5] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] * 35;   
    }},
     
    // 7th pipe
    tf::Pipe{pipes[6] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 1;   
    }}, 

    // 8th pipe
    tf::Pipe{pipes[7] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      int input = mybuffer[pf.line()][pf.pipe()-1];
      mybuffer[pf.line()][pf.pipe()] = input * input * input;   
    }}, 

    // 9th pipe
    tf::Pipe{pipes[8] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 2;   
    }},

    // 10th pipe
    tf::Pipe{pipes[9] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(std::sqrt(mybuffer[pf.line()][pf.pipe() - 1]));
    }},

    // 11th pipe
    tf::Pipe{pipes[10] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(std::log(mybuffer[pf.line()][pf.pipe() - 1]));
    }},

    // 12th pipe
    tf::Pipe{pipes[11] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 3;
    }},

    // 13th pipe
    tf::Pipe{pipes[12] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = 0 - mybuffer[pf.line()][pf.pipe() - 1];
    }},

    // 14th pipe
    tf::Pipe{pipes[13] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }}
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_15_pipes
std::chrono::microseconds parallel_pipeline_taskflow_15_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 15>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] - 792;  
    }}, 
  
    // 6th pipe
    tf::Pipe{pipes[5] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] * 35;   
    }},
     
    // 7th pipe
    tf::Pipe{pipes[6] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 1;   
    }}, 

    // 8th pipe
    tf::Pipe{pipes[7] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      int input = mybuffer[pf.line()][pf.pipe()-1];
      mybuffer[pf.line()][pf.pipe()] = input * input * input;   
    }}, 

    // 9th pipe
    tf::Pipe{pipes[8] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 2;   
    }},

    // 10th pipe
    tf::Pipe{pipes[9] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(std::sqrt(mybuffer[pf.line()][pf.pipe() - 1]));
    }},

    // 11th pipe
    tf::Pipe{pipes[10] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(std::log(mybuffer[pf.line()][pf.pipe() - 1]));
    }},

    // 12th pipe
    tf::Pipe{pipes[11] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 3;
    }},

    // 13th pipe
    tf::Pipe{pipes[12] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = 0 - mybuffer[pf.line()][pf.pipe() - 1];
    }},

    // 14th pipe
    tf::Pipe{pipes[13] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = (mybuffer[pf.line()][pf.pipe() - 1]) * (mybuffer[pf.line()][pf.pipe() - 1]);
    }},

    // 15th pipe
    tf::Pipe{pipes[14] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }}
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

// parallel_pipeline_taskflow_16_pipes
std::chrono::microseconds parallel_pipeline_taskflow_16_pipes(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  std::vector<std::array<int, 16>> mybuffer(num_lines);

  auto beg = std::chrono::high_resolution_clock::now();
  tf::Pipeline pl(num_lines,
    // 1st pipe
    tf::Pipe{tf::PipeType::SERIAL, [i = size_t{0}, size, &mybuffer](auto& pf) mutable {
      if (i++ == size) {
        pf.stop(); 
      }
      else {
        mybuffer[pf.line()][pf.pipe()] = 1; 
      }
    }},
    
    // 2nd pipe
    tf::Pipe{pipes[1] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;  
    }}, 
  
    // 3rd pipe
    tf::Pipe{pipes[2] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 999;   
    }},
     
    // 4th pipe
    tf::Pipe{pipes[3] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 1;   
    }}, 

    // 5th pipe
    tf::Pipe{pipes[4] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] - 792;  
    }}, 
  
    // 6th pipe
    tf::Pipe{pipes[5] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] * 35;   
    }},
     
    // 7th pipe
    tf::Pipe{pipes[6] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 1;   
    }}, 

    // 8th pipe
    tf::Pipe{pipes[7] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      int input = mybuffer[pf.line()][pf.pipe()-1];
      mybuffer[pf.line()][pf.pipe()] = input * input * input;   
    }}, 

    // 9th pipe
    tf::Pipe{pipes[8] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] >> 2;   
    }},

    // 10th pipe
    tf::Pipe{pipes[9] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(std::sqrt(mybuffer[pf.line()][pf.pipe() - 1]));
    }},

    // 11th pipe
    tf::Pipe{pipes[10] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(std::log(mybuffer[pf.line()][pf.pipe() - 1]));
    }},

    // 12th pipe
    tf::Pipe{pipes[11] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] << 3;
    }},

    // 13th pipe
    tf::Pipe{pipes[12] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = 0 - mybuffer[pf.line()][pf.pipe() - 1];
    }},

    // 14th pipe
    tf::Pipe{pipes[13] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = (mybuffer[pf.line()][pf.pipe() - 1]) * (mybuffer[pf.line()][pf.pipe() - 1]);
    }},

    // 15th pipe
    tf::Pipe{pipes[14] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = static_cast<int>(mybuffer[pf.line()][pf.pipe() - 1] / 97);
    }},

    // 16th pipe
    tf::Pipe{pipes[15] == 's' ? tf::PipeType::SERIAL : tf::PipeType::PARALLEL, [&mybuffer](auto& pf) {
      mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 99999;   
      //printf("%d\n", mybuffer[pf.line()][pf.pipe()]);
    }}
  ); 
  
  taskflow.composed_of(pl);
  executor.run(taskflow).wait(); 
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

std::chrono::microseconds measure_time_taskflow(
  std::string pipes, unsigned num_lines, unsigned num_threads, size_t size) {
 
  std::chrono::microseconds elapsed; 

  switch(pipes.size()) {
    case 1:
      elapsed = parallel_pipeline_taskflow_1_pipe(num_lines, num_threads, size);
      break;
    
    case 2:
      elapsed = parallel_pipeline_taskflow_2_pipes(pipes, num_lines, num_threads, size);
      break;

    case 3:
      elapsed = parallel_pipeline_taskflow_3_pipes(pipes, num_lines, num_threads, size);
      break;
    
    case 4:
      elapsed = parallel_pipeline_taskflow_4_pipes(pipes, num_lines, num_threads, size);
      break;

    case 5:
      elapsed = parallel_pipeline_taskflow_5_pipes(pipes, num_lines, num_threads, size);
      break;
    
    case 6:
      elapsed = parallel_pipeline_taskflow_6_pipes(pipes, num_lines, num_threads, size);
      break;
    
    case 7:
      elapsed = parallel_pipeline_taskflow_7_pipes(pipes, num_lines, num_threads, size);
      break;

    case 8:
      elapsed = parallel_pipeline_taskflow_8_pipes(pipes, num_lines, num_threads, size);
      break;
    
    case 9:
      elapsed = parallel_pipeline_taskflow_9_pipes(pipes, num_lines, num_threads, size);
      break;

    case 10:
      elapsed = parallel_pipeline_taskflow_10_pipes(pipes, num_lines, num_threads, size);
      break;
    
    case 11:
      elapsed = parallel_pipeline_taskflow_11_pipes(pipes, num_lines, num_threads, size);
      break;
    
    case 12:
      elapsed = parallel_pipeline_taskflow_12_pipes(pipes, num_lines, num_threads, size);
      break;

    case 13:
      elapsed = parallel_pipeline_taskflow_13_pipes(pipes, num_lines, num_threads, size);
      break;
    
    case 14:
      elapsed = parallel_pipeline_taskflow_14_pipes(pipes, num_lines, num_threads, size);
      break;

    case 15:
      elapsed = parallel_pipeline_taskflow_15_pipes(pipes, num_lines, num_threads, size);
      break;
    
    case 16:
      elapsed = parallel_pipeline_taskflow_16_pipes(pipes, num_lines, num_threads, size);
      break;
  }

  //std::ofstream outputfile;
  //outputfile.open("./build/benchmarks/tf_time.csv", std::ofstream::app);
  //outputfile << num_threads << ','
  //           << num_lines   << ','
  //           << pipes       << ','
  //           << size        << ','
  //           << elapsed.count()/1e3 << '\n';

  //outputfile.close();
  return elapsed;
}


