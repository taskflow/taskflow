#include <taskflow/taskflow.hpp>
#include <httplib/httplib.hpp>
#include <CLI11/CLI11.hpp>

// TODO

int main(int argc, char* argv[]) {
  
  // parse arguments
  CLI::App app{"tfprof"};

  int port {8080};  
  app.add_option("-p,--port", port, "port to listen (default=8080)");

  std::string input;
  app.add_option("-i,--input", input, "input profiling file")
     ->required();

  CLI11_PARSE(app, argc, argv);
  
  // launc the server
  httplib::Server svr;

  auto ret = svr.set_mount_point("/", "../");
  if (!ret) {
    std::cout << "folder doesn't exist ...\n";
  }
  //svr.Get(R"/zoomX=[", [](const httplib::Request &req, httplib::Response &res) {
  //  std::cout << "get /gg\n";
  //  std::cout << req.method << '\n';
  //  std::cout << req.body << '\n';
  //  res.set_content("Hello World!", "text/plain");
  //});
  svr.Post("/query", [](const httplib::Request& req, httplib::Response& res){
    std::cout << req.method << '\n';
    std::cout << req.body << '\n';
    res.set_content("{\"a\": 123, \"b\": 456}", "application/json");
  });

  svr.listen("0.0.0.0", 8080);

  return 0;
}


