#include "levelgraph.hpp"
#include <omp.h>
#include <fstream>
#include <thread>
//#include "matrix_calculation.hpp"

void pipe_helper(LevelGraph& graph, const size_t i) {

  int lev = i/graph.level();
  int len = i%graph.length();
  int uid = graph.node_at(lev, len).uid();
  int retval = work(uid);
  graph.node_at(lev, len).set_value(retval);
}

void last_pipe_helper(LevelGraph& graph, const size_t i) {
  int lev = i/graph.level();
  int len = i%graph.length();
  int uid = graph.node_at(lev, len).uid();
  int retval = work(uid);

  if (lev != 0) {
    int value_prev_level = 0;
    for (auto& in_edge : graph.node_at(lev, len)._in_edges) {
      value_prev_level += graph.node_at(lev-1, in_edge.first).get_value();
    }
    graph.node_at(lev, len).set_value(retval + value_prev_level);
  }
  else {
    graph.node_at(lev, len).set_value(retval);
  }
}


// 1 pipe
void graph_pipeline_omp_1_pipe(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();
      for (size_t i = 0; i < total_nodes; ++i){
        #pragma omp task firstprivate(i)
        {
          pipe_helper(graph, i);
        }
        #pragma omp taskwait
      }
    }
  }
}

// 2 pipes
void graph_pipeline_omp_2_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+1; ++i) {

        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            //int lev = i/graph.level();
            //int len = i%graph.length();
            //int uid = graph.node_at(lev, len).uid();
            //int retval = work(uid);
            //graph.node_at(lev, len).set_value(retval);
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            //int lev = (i-1)/graph.level();
            //int len = (i-1)%graph.length();
            //int uid = graph.node_at(lev, len).uid();
            //int retval = work(uid);

            //if (lev != 0) {
            //  int value_prev_level = 0;
            //  for (auto& in_edge : graph.node_at(lev, len)._in_edges) {
            //    value_prev_level += graph.node_at(lev-1, in_edge.first).get_value();
            //  }
            //  graph.node_at(lev, len).set_value(retval + value_prev_level);
            //}
            //else {
            //  graph.node_at(lev, len).set_value(retval);
            //}
            last_pipe_helper(graph, i-1);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 3 pipes
void graph_pipeline_omp_3_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+2; ++i) {

        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            //int lev = i/graph.level();
            //int len = i%graph.length();
            //int uid = graph.node_at(lev, len).uid();
            //int retval = work(uid);
            //graph.node_at(lev, len).set_value(retval);
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            //int lev = (i-1)/graph.level();
            //int len = (i-1)%graph.length();
            //int uid = graph.node_at(lev, len).uid();
            //int retval = work(uid);
            //graph.node_at(lev, len).set_value(retval);
            pipe_helper(graph, i-1);
          }
        }

        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            //int lev = (i-2)/graph.level();
            //int len = (i-2)%graph.length();
            //int uid = graph.node_at(lev, len).uid();
            //int retval = work(uid);

            //if (lev != 0) {
            //  int value_prev_level = 0;
            //  for (auto& in_edge : graph.node_at(lev, len)._in_edges) {
            //    value_prev_level += graph.node_at(lev-1, in_edge.first).get_value();
            //  }
            //  graph.node_at(lev, len).set_value(retval + value_prev_level);
            //}
            //else {
            //  graph.node_at(lev, len).set_value(retval);
            //}
            last_pipe_helper(graph, i-2);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}



// 4 pipes
void graph_pipeline_omp_4_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+3; ++i) {
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            //int lev = (i-3)/graph.level();
            //int len = (i-3)%graph.length();
            //int uid = graph.node_at(lev, len).uid();
            //int retval = work(uid);

            //if (lev != 0) {
            //  int value_prev_level = 0;
            //  for (auto& in_edge : graph.node_at(lev, len)._in_edges) {
            //    value_prev_level += graph.node_at(lev-1, in_edge.first).get_value();
            //  }
            //  graph.node_at(lev, len).set_value(retval + value_prev_level);
            //}
            //else {
            //  graph.node_at(lev, len).set_value(retval);
            //}
            last_pipe_helper(graph, i-3);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 5 pipes
void graph_pipeline_omp_5_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+4; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            last_pipe_helper(graph, i-4);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 6 pipes
void graph_pipeline_omp_6_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+5; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            pipe_helper(graph, i-4);
          }
        }
        // 6th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 5 && i < total_nodes+5) {
            last_pipe_helper(graph, i-5);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 7 pipes
void graph_pipeline_omp_7_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+6; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            pipe_helper(graph, i-4);
          }
        }
        // 6th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 5 && i < total_nodes+5) {
            pipe_helper(graph, i-5);
          }
        }
        // 7th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 6 && i < total_nodes+6) {
            last_pipe_helper(graph, i-6);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 8 pipes
void graph_pipeline_omp_8_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+7; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            pipe_helper(graph, i-4);
          }
        }
        // 6th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 5 && i < total_nodes+5) {
            pipe_helper(graph, i-5);
          }
        }
        // 7th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 6 && i < total_nodes+6) {
            pipe_helper(graph, i-6);
          }
        }
        // 8th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 7 && i < total_nodes+7) {
            last_pipe_helper(graph, i-7);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 9 pipes
void graph_pipeline_omp_9_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+8; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            pipe_helper(graph, i-4);
          }
        }
        // 6th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 5 && i < total_nodes+5) {
            pipe_helper(graph, i-5);
          }
        }
        // 7th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 6 && i < total_nodes+6) {
            pipe_helper(graph, i-6);
          }
        }
        // 8th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 7 && i < total_nodes+7) {
            pipe_helper(graph, i-7);
          }
        }
        // 9th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 8 && i < total_nodes+8) {
            last_pipe_helper(graph, i-8);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 10 pipes
void graph_pipeline_omp_10_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+9; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            pipe_helper(graph, i-4);
          }
        }
        // 6th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 5 && i < total_nodes+5) {
            pipe_helper(graph, i-5);
          }
        }
        // 7th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 6 && i < total_nodes+6) {
            pipe_helper(graph, i-6);
          }
        }
        // 8th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 7 && i < total_nodes+7) {
            pipe_helper(graph, i-7);
          }
        }
        // 9th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 8 && i < total_nodes+8) {
            pipe_helper(graph, i-8);
          }
        }
        // 10th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 9 && i < total_nodes+9) {
            last_pipe_helper(graph, i-9);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 11 pipes
void graph_pipeline_omp_11_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+10; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            pipe_helper(graph, i-4);
          }
        }
        // 6th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 5 && i < total_nodes+5) {
            pipe_helper(graph, i-5);
          }
        }
        // 7th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 6 && i < total_nodes+6) {
            pipe_helper(graph, i-6);
          }
        }
        // 8th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 7 && i < total_nodes+7) {
            pipe_helper(graph, i-7);
          }
        }
        // 9th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 8 && i < total_nodes+8) {
            pipe_helper(graph, i-8);
          }
        }
        // 10th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 9 && i < total_nodes+9) {
            pipe_helper(graph, i-9);
          }
        }
        // 11th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 10 && i < total_nodes+10) {
            last_pipe_helper(graph, i-10);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 12 pipes
void graph_pipeline_omp_12_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+11; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            pipe_helper(graph, i-4);
          }
        }
        // 6th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 5 && i < total_nodes+5) {
            pipe_helper(graph, i-5);
          }
        }
        // 7th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 6 && i < total_nodes+6) {
            pipe_helper(graph, i-6);
          }
        }
        // 8th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 7 && i < total_nodes+7) {
            pipe_helper(graph, i-7);
          }
        }
        // 9th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 8 && i < total_nodes+8) {
            pipe_helper(graph, i-8);
          }
        }
        // 10th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 9 && i < total_nodes+9) {
            pipe_helper(graph, i-9);
          }
        }
        // 11th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 10 && i < total_nodes+10) {
            pipe_helper(graph, i-10);
          }
        }
        // 12th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 11 && i < total_nodes+11) {
            last_pipe_helper(graph, i-11);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 13 pipes
void graph_pipeline_omp_13_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+12; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            pipe_helper(graph, i-4);
          }
        }
        // 6th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 5 && i < total_nodes+5) {
            pipe_helper(graph, i-5);
          }
        }
        // 7th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 6 && i < total_nodes+6) {
            pipe_helper(graph, i-6);
          }
        }
        // 8th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 7 && i < total_nodes+7) {
            pipe_helper(graph, i-7);
          }
        }
        // 9th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 8 && i < total_nodes+8) {
            pipe_helper(graph, i-8);
          }
        }
        // 10th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 9 && i < total_nodes+9) {
            pipe_helper(graph, i-9);
          }
        }
        // 11th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 10 && i < total_nodes+10) {
            pipe_helper(graph, i-10);
          }
        }
        // 12th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 11 && i < total_nodes+11) {
            pipe_helper(graph, i-11);
          }
        }
        // 13th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 12 && i < total_nodes+12) {
            last_pipe_helper(graph, i-12);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 14 pipes
void graph_pipeline_omp_14_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+13; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            pipe_helper(graph, i-4);
          }
        }
        // 6th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 5 && i < total_nodes+5) {
            pipe_helper(graph, i-5);
          }
        }
        // 7th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 6 && i < total_nodes+6) {
            pipe_helper(graph, i-6);
          }
        }
        // 8th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 7 && i < total_nodes+7) {
            pipe_helper(graph, i-7);
          }
        }
        // 9th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 8 && i < total_nodes+8) {
            pipe_helper(graph, i-8);
          }
        }
        // 10th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 9 && i < total_nodes+9) {
            pipe_helper(graph, i-9);
          }
        }
        // 11th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 10 && i < total_nodes+10) {
            pipe_helper(graph, i-10);
          }
        }
        // 12th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 11 && i < total_nodes+11) {
            pipe_helper(graph, i-11);
          }
        }
        // 13th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 12 && i < total_nodes+12) {
            pipe_helper(graph, i-12);
          }
        }
        // 14th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 13 && i < total_nodes+13) {
            last_pipe_helper(graph, i-13);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 15 pipes
void graph_pipeline_omp_15_pipes(LevelGraph& graph) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+14; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            pipe_helper(graph, i-4);
          }
        }
        // 6th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 5 && i < total_nodes+5) {
            pipe_helper(graph, i-5);
          }
        }
        // 7th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 6 && i < total_nodes+6) {
            pipe_helper(graph, i-6);
          }
        }
        // 8th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 7 && i < total_nodes+7) {
            pipe_helper(graph, i-7);
          }
        }
        // 9th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 8 && i < total_nodes+8) {
            pipe_helper(graph, i-8);
          }
        }
        // 10th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 9 && i < total_nodes+9) {
            pipe_helper(graph, i-9);
          }
        }
        // 11th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 10 && i < total_nodes+10) {
            pipe_helper(graph, i-10);
          }
        }
        // 12th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 11 && i < total_nodes+11) {
            pipe_helper(graph, i-11);
          }
        }
        // 13th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 12 && i < total_nodes+12) {
            pipe_helper(graph, i-12);
          }
        }
        // 14th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 13 && i < total_nodes+13) {
            pipe_helper(graph, i-13);
          }
        }
        // 15th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 14 && i < total_nodes+14) {
            last_pipe_helper(graph, i-14);
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

// 16 pipes
void graph_pipeline_omp_16_pipes(LevelGraph& graph) {
  //std::ofstream outputfile;
  //outputfile.open("./omp_16_pipes.txt", std::ofstream::app);

  #pragma omp parallel
  {
    #pragma omp single
    {
      size_t total_nodes = graph.level() * graph.length();

      for (size_t i = 0; i < total_nodes+15; ++i){
        // 1st pipe
        #pragma omp task firstprivate(i)
        {
          if (i < total_nodes) {
            pipe_helper(graph, i);
          }
        }

        // 2nd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 1 && i < total_nodes+1) {
            pipe_helper(graph, i-1);
          }
        }
        // 3rd pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 2 && i < total_nodes+2) {
            pipe_helper(graph, i-2);
          }
        }
        // 4th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 3 && i < total_nodes+3) {
            pipe_helper(graph, i-3);
          }
        }
        // 5th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 4 && i < total_nodes+4) {
            pipe_helper(graph, i-4);
          }
        }
        // 6th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 5 && i < total_nodes+5) {
            pipe_helper(graph, i-5);
          }
        }
        // 7th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 6 && i < total_nodes+6) {
            pipe_helper(graph, i-6);
          }
        }
        // 8th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 7 && i < total_nodes+7) {
            pipe_helper(graph, i-7);
          }
        }
        // 9th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 8 && i < total_nodes+8) {
            pipe_helper(graph, i-8);
          }
        }
        // 10th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 9 && i < total_nodes+9) {
            pipe_helper(graph, i-9);
          }
        }
        // 11th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 10 && i < total_nodes+10) {
            pipe_helper(graph, i-10);
          }
        }
        // 12th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 11 && i < total_nodes+11) {
            pipe_helper(graph, i-11);
          }
        }
        // 13th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 12 && i < total_nodes+12) {
            pipe_helper(graph, i-12);
          }
        }
        // 14th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 13 && i < total_nodes+13) {
            pipe_helper(graph, i-13);
          }
        }
        // 15th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 14 && i < total_nodes+14) {
            pipe_helper(graph, i-14);
          }
        }
        // 16th pipe
        #pragma omp task firstprivate(i)
        {
          if (i >= 15 && i < total_nodes+15) {
            last_pipe_helper(graph, i-15);
          }
        }
        #pragma omp taskwait
      }
    }
  }
  //outputfile.close();
}

std::chrono::microseconds measure_time_omp(
  LevelGraph& graph, size_t pipes, unsigned , unsigned num_threads) {

  omp_set_num_threads(num_threads);


  auto beg = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  switch(pipes) {
    case 1:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_1_pipe(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 2:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_2_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 3:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_3_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 4:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_4_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 5:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_5_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 6:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_6_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 7:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_7_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 8:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_8_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 9:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_9_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 10:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_10_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 11:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_11_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 12:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_12_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 13:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_13_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 14:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_14_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 15:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_15_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 16:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_omp_16_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    default:
      throw std::runtime_error("can support only up to 16 pipes");
    break;
  }
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

