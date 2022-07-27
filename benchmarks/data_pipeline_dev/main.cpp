#include "data_pipeline.hpp"
#include <CLI11.hpp>

int main(int argc, char* argv[]) {

  CLI::App app{"Parallel DataPipeline"};

  unsigned num_threads {1};
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");

  std::string model = "normal";
  app.add_option("-m,--model", model, "model name normal|efficient (default=normal)")
     ->check([] (const std::string& m) {
        if(m != "normal" && m != "efficient") {
          return "model name should be \"normal\" or \"efficient\"";
        }
        return "";
     });

  unsigned num_lines {8};
  app.add_option("-l,--num_lines", num_lines, "num of lines (default=8)");

  std::string pipes = "ssssssss";
  app.add_option("-p,--pipes", pipes, "the chain of pipes (default=ssssssss)")
      ->check([pipes] (const std::string& p) {
        if (p[0] == 'p') {
          return "the first pipe should be \"s\" (serial)";
        }
        else if (p.size() > 16) {
          return "no more than 16 pipes";
        }
        else if (p.size() == 0) {
          return "at least one pipe is required";
        }
        return "";
      });

  std::string datatype = "int";
  app.add_option("-d,--datatype", datatype, "datatype name int|string (default=int)")
     ->check([] (const std::string& d) {
        if(d != "int" && d != "string") {
          return "model name should be \"int\" or \"string\"";
        }
        return "";
     });

  CLI11_PARSE(app, argc, argv);

  std::cout << "model="       << model       << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds="  << num_rounds  << ' '
            << "num_lines="   << num_lines   << ' '
            << "pipes="       << pipes       << ' '
            << "datatype="    << datatype    << ' '
            << std::endl;

  std::cout << std::setw(12) << "size"
            << std::setw(12) << "Runtime"
            << '\n';

  size_t  log_length = 23;

  for(size_t i = 1; i <= log_length; ++i) {

    size_t L = 1 << i;

    double runtime {0.0};

    for(unsigned j = 0; j < num_rounds; ++j) {
      if(model == "normal") {
        runtime += measure_time_normal(pipes, num_lines, num_threads, L, datatype).count();
      } else if (model == "efficient") {
        runtime += measure_time_efficient(pipes, num_lines, num_threads, L, datatype).count();
      }
    }

    std::cout << std::setw(12) << L
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;

    /*if (model == "tf") {
      std::ofstream outputfile;
      outputfile.open("./tf_time.csv", std::ofstream::app);
      outputfile << num_threads << ','
                 << num_lines   << ','
                 << pipes       << ','
                 << L           << ','
                 << runtime / num_rounds / 1e3 << '\n';

      outputfile.close();
    }
    else if (model == "tbb") {
      std::ofstream outputfile;
      outputfile.open("./tbb_time.csv", std::ofstream::app);
      outputfile << num_threads << ','
                 << num_lines   << ','
                 << pipes       << ','
                 << L           << ','
                 << runtime / num_rounds / 1e3 << '\n';

      outputfile.close();
    }*/
  }
}
