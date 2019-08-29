/*
    Copyright (c) 2005-2019 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#define VIDEO_WINMAIN_ARGS

#include <iostream>
#include <CLI11.hpp>
#include <utility>
#include "tbb/tick_count.h"

#include "seismic_video.h"
#include "universe.h"

Universe u;

struct RunOptions {
    //! It is used for console mode for test with different number of threads and also has
    //! meaning for GUI: threads.first  - use separate event/updating loop thread (>0) or not (0).
    //!                  threads.second - initialization value for scheduler
    //utility::thread_number_range threads;
    std::pair<unsigned, unsigned> thread_range;
    int numberOfFrames;
    bool silent;
    bool parallel;
    std::string model;
    RunOptions(std::pair<unsigned, unsigned> thread_range, int num_frames , bool silent, bool parallel, std::string model)
        : thread_range(thread_range), numberOfFrames(num_frames), silent(silent), parallel(parallel), model(model)
    {}
};

int main(int argc, char *argv[]) {

  CLI::App app{"Seismic"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};  
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");

  unsigned num_frames {1000};  
  app.add_option("-f,--num_frames", num_frames, "number of frames (default=1000, 0 means unlimited)");

  // Serial mode: in GUI mode start with serial version of algorithm
  std::string model = "tf";
  app.add_option("-m,--model", model, "model name tbb|omp|tf|serial (default=tf)")
     ->check([] (const std::string& m) {
        if(m != "tbb" && m != "omp" && m != "tf" && m != "serial") {
          return "model name should be \"tbb\", \"omp\", or \"tf\"";
        }
        return "";
     });

  CLI11_PARSE(app, argc, argv);
   
  std::cout << "model=" << model << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds=" << num_rounds << ' '
            << "num_frames=" << num_frames << ' '
            << std::endl;

  const bool silent = false;
    try{
        auto start_time = std::chrono::high_resolution_clock::now();
        //tbb::tick_count mainStartTime = tbb::tick_count::now();
        RunOptions options(std::make_pair(0, num_threads), num_frames, silent, model != "serial", model);
        u.set_model(model);
        //RunOptions options = ParseCommandLine(argc,argv);
        SeismicVideo video(u, num_frames, num_threads, model != "serial");
        //SeismicVideo video(u,options.numberOfFrames,options.threads.last,options.parallel);

        // video layer init
        if(video.init_window(u.UniverseWidth, u.UniverseHeight)) {
            video.calc_fps = true;
            video.threaded = options.thread_range.first > 0;
            // video is ok, init Universe
            u.InitializeUniverse(video);
            // main loop
            video.main_loop();
        }
        else if(video.init_console()) {
            // do console mode
            printf("Substituting %u for unlimited frames because not running interactively\n", num_frames);
            //for(int p = options.threads.first;  p <= options.threads.last; p = options.threads.step(p)) {
            for(unsigned p = options.thread_range.first;  p <= options.thread_range.second; p ++) {  
                tbb::tick_count xwayParallelismStartTime = tbb::tick_count::now();
                u.InitializeUniverse(video);
                int numberOfFrames = num_frames;
#if __TBB_MIC_OFFLOAD
                drawing_memory dmem = video.get_drawing_memory();
                char *pMem = dmem.get_address();
                size_t memSize = dmem.get_size();

                #pragma offload target(mic) in(u, numberOfFrames, p, dmem), out(pMem:length(memSize))
                {
                    // It is necessary to update the pointer on mic 
                    // since the address spaces on host and on target are different
                    dmem.set_address(pMem);
                    u.SetDrawingMemory(dmem);
#endif // __TBB_MIC_OFFLOAD
                    if (p==0) {
                      //run a serial version
                      for( int i=0; i<numberOfFrames; ++i ) {
                        u.SerialUpdateUniverse();
                      }
                    } 
                    else {
                      if(model == "tbb") {
                        measure_time_tbb(p, numberOfFrames, u);
                      }
                      else if(model == "tf") {
                        measure_time_taskflow(p, numberOfFrames, u);
                      }
                      else {
                        assert(false);
                      }
                     
                      //tbb::task_scheduler_init init(p);
                      //for( int i=0; i<numberOfFrames; ++i ) {
                      //  //u.ParallelUpdateUniverse();
                      //}
                    }
#if __TBB_MIC_OFFLOAD
                }
#endif // __TBB_MIC_OFFLOAD

                if (!options.silent){
                  double fps =  options.numberOfFrames/((tbb::tick_count::now()-xwayParallelismStartTime).seconds());
                  std::cout<<fps<<" frame per sec with ";
                  if (p==0) {
                    std::cout<<"serial code\n";
                  }
                  else {
                    std::cout<<p<<" way parallelism\n";
                  }
                }

                if(model == "serial") {
                  break;
                }
            }
        }
        video.terminate();
        auto end_time = std::chrono::high_resolution_clock::now();

        std::cout << "elapsed time : " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()/1e3
                  << " seconds"
                  << std::endl;
        return 0;
    }
    catch(std::exception& e){
      std::cerr<<"error occurred. error text is :\"" <<e.what()<<"\"\n";
      return 1;
    }
}
