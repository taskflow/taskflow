#include <taskflow/taskflow.hpp>
#include <filesystem>
#include "httplib.hpp"

// TODO

int main() {

  std::filesystem::path path;

  std::cout << std::filesystem::current_path() << std::endl;

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


