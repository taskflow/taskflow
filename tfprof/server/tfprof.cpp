#include <taskflow/taskflow.hpp>
#include <httplib/httplib.hpp>
#include <CLI11/CLI11.hpp>

// TODO
namespace tf {

class Database {

  public:

  struct ZoomX {
    size_t beg;
    size_t end;
  };

  struct ZoomY {
    std::vector<std::tuple<size_t, size_t, size_t>> workers;
  };

  struct TaskData {

    std::string name;
    TaskType type;
    size_t beg;     // beg time
    size_t end;     // end time

    TaskData(std::string s, TaskType t, size_t b, size_t e) :
      name{std::move(s)}, type{t}, beg{b}, end{e} {
    }

    TaskData(const TaskData&) = delete;
    TaskData(TaskData&&) = default;

    TaskData& operator = (const TaskData&) = delete;
    TaskData& operator = (TaskData&&) = default;
  };

  struct WorkerData {
    size_t eid;
    size_t wid;
    size_t lid;
    std::vector<TaskData> tasks; 

    WorkerData(size_t e, size_t w, size_t l) : 
      eid{e}, wid{w}, lid{l} {
    }

    WorkerData(const WorkerData&) = delete;
    WorkerData(WorkerData&&) = default;

    WorkerData& operator = (const WorkerData&) = delete;
    WorkerData& operator = (WorkerData&&) = default; 
  };

  struct Cluster {
    size_t w;
    size_t f;  // from task
    size_t t;  // to task   (inclusive)
    size_t g;

    Cluster(size_t in_w, size_t in_f, size_t in_t, size_t in_g) :
      w{in_w}, f{in_f}, t{in_t}, g{in_g} {
    }

    using iterator_t = std::list<Cluster>::iterator;
  };

  struct ClusterComparator {
    bool operator () (Cluster::iterator_t a, Cluster::iterator_t b) {
      return a->g > b->g;
    }
  };

  using ClusterHeap = std::priority_queue<
    Cluster::iterator_t, std::vector<Cluster::iterator_t>, ClusterComparator
  >;

  public:

  Database(const std::string fpath) {

    std::ifstream ifs(fpath);

    if(!ifs) {
      TF_THROW("failed to open profile data ", fpath);
    }
    
    ProfileData pd;
    tf::Deserializer deserializer(ifs);
    deserializer(pd);
    
    // find the minimum starting point
    auto origin = tf::observer_stamp_t::max();
    for(auto& timeline : pd.timelines) {
      if(timeline.origin < origin) {
        origin = timeline.origin; 
      }
    }
    
    // conver to flat data
    size_t idx = 0;

    _ewl2wd.resize(pd.timelines.size());
    for(size_t e=0; e<pd.timelines.size(); e++) {
      _ewl2wd[e].resize(pd.timelines[e].segments.size());
      for(size_t w=0; w<pd.timelines[e].segments.size(); w++) {
        _ewl2wd[e][w].resize(pd.timelines[e].segments[w].size());
        for(size_t l=0; l<pd.timelines[e].segments[w].size(); l++) {

          _ewl2wd[e][w][l] = idx++;
          _num_tasks += pd.timelines[e].segments[w][l].size();

          // a new worker data
          WorkerData wd(e, w, l);
          for(size_t s=0; s<pd.timelines[e].segments[w][l].size(); s++) {
            auto& t = wd.tasks.emplace_back(
              std::move(pd.timelines[e].segments[w][l][s].name),
              pd.timelines[e].segments[w][l][s].type,
              std::chrono::duration_cast<std::chrono::microseconds>(
                pd.timelines[e].segments[w][l][s].beg - origin
              ).count(),
              std::chrono::duration_cast<std::chrono::microseconds>(
                pd.timelines[e].segments[w][l][s].end - origin
              ).count()
            );
            
            if(t.beg < _minX) _minX = t.beg;
            if(t.end > _maxX) _maxX = t.end;
          }
          _wd.push_back(std::move(wd));
        }
      }
    }
  }

  void query(
    std::ostream& os, 
    std::optional<ZoomX> zoomx, 
    std::optional<ZoomY> zoomy
  ) const {

    // Acquire the range of worker id
    std::vector<size_t> w;
    
    if(zoomy) {
      w.resize(zoomy->workers.size());
      for(size_t i=0; i<zoomy->workers.size(); i++) {
        const auto& [eid, wid, lid] = zoomy->workers[i];
        w[i] = _ewl2wd[eid][wid][lid];
      }
    }
    else {
      w.resize(_wd.size());
      for(size_t i=0; i<_wd.size(); i++) {
        w[i] = i;
      }
    }

    if(!zoomx) {
      zoomx = ZoomX{_minX, _maxX};
    }

    std::vector<std::list<Cluster>> clusters{w.size()};
    ClusterHeap heap;
    
    // bsearch the range of segments for each worker data
    // TODO: parallel_for?
    for(size_t i=0; i<w.size(); i++) {

      size_t slen = _wd[w[i]].tasks.size();
      size_t beg, end, mid;
      std::optional<size_t> l, r;
      
      // r = maxArg {span[0] <= zoomX[1]}
      beg = 0, end = slen;
      while(beg < end) {
        mid = (beg + end) >> 1;
        if(_wd[w[i]].tasks[mid].beg <= zoomx->end) {
          beg = mid + 1;
          r = (r == std::nullopt) ? mid : std::max(mid, *r);
        }
        else {
          end = mid;
        }
      }
      
      // TODO
      if(r == std::nullopt) {
        continue;
      }
      
      // l = minArg {span[1] >= zoomX[0]}
      beg = 0, end = slen;
      while(beg < end) {
        mid = (beg + end) >> 1;
        if(_wd[w[i]].tasks[mid].end >= zoomx->beg) {
          end = mid;
          l = (l == std::nullopt) ? mid : std::min(mid, *l);
        }
        else {
          beg = mid + 1;
        }
      };
      
      // TODO
      if(l == std::nullopt || *l > *r) {
        continue;
      }
      
      // range ok
      //_wd[wd[i]].tbeg = *l;
      //_wd[wd[i]].tend = *r+1;
      //R += (*r + 1 - *l);
      
      // TODO: can we use priority queue to do clustering
      for(size_t s=*l; s<=*r; s++) {
        if(s != *r) {
          clusters[w[i]].emplace_back(
            w[i], 
            s, 
            s,
            _wd[w[i]].tasks[s+1].end - _wd[w[i]].tasks[s].beg
          );
        }
        else {  // boundary
          clusters[w[i]].emplace_back(
            w[i], s, s, std::numeric_limits<size_t>::max()
          );
        }

        auto itr = std::prev(clusters[w[i]].end());
        heap.push(itr);
    
        while(heap.size() && heap.size() >= _max_rendered_tasks) {

          auto top = heap.top(); 

          // if all clusters are in boundary - no need to cluster anymore
          if(top->g == std::numeric_limits<size_t>::max()) {
            break;
          }
          
          // remove the top element and cluster it with the next
          heap.pop();
          
          // merge top with top->next 
          std::next(top)->f = top->f;
          clusters[top->w].erase(top);
        }
      }
    }

    // Output the segments
    bool first_worker = true;
    os << "[";
    for(size_t i=0; i<w.size(); i++) {

      if(!first_worker) {
        os << ",";
      }
      else {
        first_worker = false;
      }

      os << "{\"executor\":\"" << _wd[w[i]].eid << "\","
         << "\"worker\":\"" << "E" << _wd[w[i]].eid 
                            << ".W" << _wd[w[i]].wid 
                            << ".L" << _wd[w[i]].lid << "\","
         << "\"tasks\":\"" << clusters[w[i]].size() << "\","
         << "\"segs\": [";
      
      size_t T=0, loads[TASK_TYPES.size()] = {0};
      bool first_cluster = true;
        
      for(const auto& cluster : clusters[w[i]]) {  

        if(!first_cluster) {
          os << ",";
        }
        else {
          first_cluster = false;
        }

        // single task
        os << "{";
        if(cluster.f == cluster.t) {
          const auto& task = _wd[w[i]].tasks[cluster.f];
          os << "\"name\":\"" << task.name << "\","
             << "\"type\":\""  << task_type_to_string(task.type) << "\","
             << "\"span\": ["  << task.beg << "," << task.end << "]";
        }
        else {
          const auto& ftask = _wd[w[i]].tasks[cluster.f];
          const auto& ttask = _wd[w[i]].tasks[cluster.t];
          os << "\"name\":\"-\","
             << "\"type\":\"clustered\","
             << "\"span\": ["  << ftask.beg << "," << ttask.end << "]";
        }
        os << "}";

        // calculate load
        // TODO optimization with DP
        for(size_t j=cluster.f; j<=cluster.t; j++) {
          size_t t = _wd[w[i]].tasks[j].end - _wd[w[i]].tasks[j].beg;
          T += t;
          loads[_wd[w[i]].tasks[j].type] += t;
        }
      }
      os << "],";  // end segs

      // load
      os << "\"load\":[";
      size_t x = 0;
      for(size_t k=0; k<TASK_TYPES.size(); k++) {
        auto type = TASK_TYPES[k];
        if(k) os << ",";
        os << "{\"type\":\"" << task_type_to_string(type) << "\","
           << "\"span\":[" << x << "," << x+loads[type] << "],"
           << "\"ratio\":" << (T>0 ? loads[type]*100.0f/T : 0) << "}";
        x+=loads[type];
      }
      os << "],";
      
      // totalTime
      os << "\"totalTime\":" << T;

      os << "}";
    }
    os << "]";
  }

  private:

    std::vector<WorkerData> _wd;

    size_t _minX {std::numeric_limits<size_t>::max()};
    size_t _maxX {std::numeric_limits<size_t>::min()};
    size_t _num_tasks {0};
    size_t _max_rendered_tasks {500};

    std::vector<std::vector<std::vector<size_t>>> _ewl2wd;
};

}  // namespace tf ------------------------------------------------------------

int main(int argc, char* argv[]) {
  
  // parse arguments
  CLI::App app{"tfprof"};

  int port {8080};  
  app.add_option("-p,--port", port, "port to listen (default=8080)");

  std::string input;
  app.add_option("-i,--input", input, "input profiling file")
     ->required();

  CLI11_PARSE(app, argc, argv);

  tf::Database db(input);

  tf::Database::ZoomX zoomx;
  tf::Database::ZoomY zoomy;

  zoomx.beg = 0, zoomx.end = 100;

  std::ostringstream oss;

  db.query(oss, std::nullopt, std::nullopt);

  std::cout << oss.str() << '\n';
  
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


