#include "levelgraph.hpp"
#include <fstream>


int pipe_helper(
  LevelGraph& graph,
  const int uid,
  const size_t level,
  const size_t index,
  const int count) {

  int retval = 0;
  for (int i = 0; i < count; ++i) {
    //retval = work(uid + retval);
    retval = work(uid);
    graph.node_at(level, index).set_value(retval);
  }

  return retval;
}


// 1 pipe
void graph_pipeline_gold_1_pipe(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_1_pipe.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {

      // 1st pipe
      int retval = work(graph.node_at(i, j).uid());
      //std::cout << "retval = " << retval << '\n';
      graph.node_at(i, j).set_value(retval);

      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 2 pipes
void graph_pipeline_gold_2_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_2_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe
      int retval = pipe_helper(graph, uid, i, j, 1);

      // 2nd pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }
      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }

  outputfile.close();
}

// 3 pipes
void graph_pipeline_gold_3_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_3_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 2nd pipe
      int retval = pipe_helper(graph, uid, i, j, 2);

      // 3rd pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }
      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }


  outputfile.close();
}

// 4 pipes
void graph_pipeline_gold_4_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_4_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 3th pipe
      int retval = pipe_helper(graph, uid, i, j, 3);

      // 4th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }
      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 5 pipes
void graph_pipeline_gold_5_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_5_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 4th pipe
      int retval = pipe_helper(graph, uid, i, j, 4);

      // 5th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }
      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 6 pipes
void graph_pipeline_gold_6_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_6_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 5th pipe
      int retval = pipe_helper(graph, uid, i, j, 5);

      // 6th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }
      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 7 pipes
void graph_pipeline_gold_7_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_7_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 6th pipe
      int retval = pipe_helper(graph, uid, i, j, 6);

      // 7th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }
      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 8 pipes
void graph_pipeline_gold_8_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_8_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 7th pipe
      int retval = pipe_helper(graph, uid, i, j, 7);

      // 8th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }
      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 9 pipes
void graph_pipeline_gold_9_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_9_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 8th pipe
      int retval = pipe_helper(graph, uid, i, j, 8);

      // 9th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }

      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 10 pipes
void graph_pipeline_gold_10_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_10_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 9th pipe
      int retval = pipe_helper(graph, uid, i, j, 9);

      // 10th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }

      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 11 pipes
void graph_pipeline_gold_11_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_11_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 10th pipe
      int retval = pipe_helper(graph, uid, i, j, 10);

      // 11th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }

      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 12 pipes
void graph_pipeline_gold_12_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_12_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 11th pipe
      int retval = pipe_helper(graph, uid, i, j, 11);

      // 12th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }
      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 13 pipes
void graph_pipeline_gold_13_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_13_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 12th pipe
      int retval = pipe_helper(graph, uid, i, j, 12);

      // 13th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }
      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 14 pipes
void graph_pipeline_gold_14_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_14_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 13th pipe
      int retval = pipe_helper(graph, uid, i, j, 13);

      // 14th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }
      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 15 pipes
void graph_pipeline_gold_15_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_15_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 14th pipe
      int retval = pipe_helper(graph, uid, i, j, 14);

      // 15th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }

      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

// 16 pipes
void graph_pipeline_gold_16_pipes(LevelGraph& graph) {
  std::ofstream outputfile;
  outputfile.open("./gold_16_pipes.txt", std::ofstream::app);

  for (size_t i = 0; i < graph.level(); ++i) {
    for (size_t j = 0; j < graph.length(); ++j) {
      int uid = graph.node_at(i, j).uid();

      // 1st pipe ~ 15th pipe
      int retval = pipe_helper(graph, uid, i, j, 15);

      // 16th pipe
      retval = work(uid);

      if (i != 0) {
        int value_prev_level = 0;
        for (auto& in_edge : graph.node_at(i, j)._in_edges) {
          value_prev_level += graph.node_at(i-1, in_edge.first).get_value();
        }
        graph.node_at(i, j).set_value(retval + value_prev_level);
      }
      else {
        graph.node_at(i, j).set_value(retval);
      }
      outputfile << graph.node_at(i, j).get_value() << '\n';
    }
  }
  outputfile.close();
}

std::chrono::microseconds measure_time_gold(
  LevelGraph& graph, size_t pipes) {

  auto beg = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  switch(pipes) {
    case 1:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_1_pipe(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 2:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_2_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 3:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_3_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 4:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_4_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 5:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_5_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 6:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_6_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 7:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_7_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 8:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_8_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 9:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_9_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 10:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_10_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 11:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_11_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 12:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_12_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 13:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_13_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 14:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_14_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 15:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_15_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    case 16:
      beg = std::chrono::high_resolution_clock::now();
      graph_pipeline_gold_16_pipes(graph);
      end = std::chrono::high_resolution_clock::now();
    break;

    default:
      throw std::runtime_error("can support only up to 16 pipes");
    break;
  }

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

