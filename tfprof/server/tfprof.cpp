#include <taskflow/taskflow.hpp>
#include <cmath>
#include <httplib/httplib.hpp>
#include <CLI11/CLI11.hpp>
#include <nlohmann/json.hpp>
  
// TODO
namespace tf {

class Database {

  public:

  enum ViewType {
    CLUSTER = 0,
    CRITICALITY
  };

  struct ZoomX {
    double beg;
    double end;

    ZoomX() = default;
    ZoomX(double b, double e) : beg{b}, end{e} {}
  };

  struct ZoomY {

    std::vector<std::string> workers;
    
    ZoomY() = default;
    ZoomY(const ZoomY&) = delete;
    ZoomY(ZoomY&&) = default;

    ZoomY& operator = (const ZoomY&) = delete;
    ZoomY& operator = (ZoomY&&) = delete;
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

    size_t span() const {
      return end - beg;
    }
  };

  struct WorkerData {
    size_t eid;
    size_t wid;
    size_t lid;
    std::vector<TaskData> tasks; 

    std::string name;

    WorkerData(size_t e, size_t w, size_t l, const std::string& n) : 
      eid{e}, wid{w}, lid{l}, name {n} {
    }

    WorkerData(const WorkerData&) = delete;
    WorkerData(WorkerData&&) = default;

    WorkerData& operator = (const WorkerData&) = delete;
    WorkerData& operator = (WorkerData&&) = default; 

    template <typename T>
    std::optional<size_t> lower_bound(T value) const {
      
      size_t slen = tasks.size();
      size_t beg, end, mid;
      std::optional<size_t> l;

      // l = minArg {span[1] >= zoomX[0]}
      beg = 0, end = slen;
      while(beg < end) {
        mid = (beg + end) >> 1;
        if(tasks[mid].end >= value) {
          end = mid;
          l = (l == std::nullopt) ? mid : std::min(mid, *l);
        }
        else {
          beg = mid + 1;
        }
      };

      return l;
    }
    
    template <typename T>
    std::optional<size_t> upper_bound(T value) const {
      
      size_t slen = tasks.size();
      size_t beg, end, mid;
      std::optional<size_t> r;

      // r = maxArg {span[0] <= zoomX[1]}
      beg = 0, end = slen;
      while(beg < end) {
        mid = (beg + end) >> 1;
        if(tasks[mid].beg <= value) {
          beg = mid + 1;
          r = (r == std::nullopt) ? mid : std::max(mid, *r);
        }
        else {
          end = mid;
        }
      }

      return r;
    }
  };

  struct Cluster {
    size_t w;
    size_t f;  // from task
    size_t t;  // to task   (inclusive)
    size_t k;  // key

    Cluster(size_t in_w, size_t in_f, size_t in_t, size_t in_k) :
      w{in_w}, f{in_f}, t{in_t}, k{in_k} {
    }

    using iterator_t = std::list<Cluster>::iterator;
  };

  struct ClusterComparator {
    bool operator () (Cluster::iterator_t a, Cluster::iterator_t b) {
      return a->k > b->k;
    }
  };

  struct ClusterHeap : public std::priority_queue<
    Cluster::iterator_t, std::vector<Cluster::iterator_t>, ClusterComparator
  > {
  };

  public:

  Database(const std::string& fpath) {

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
    _num_executors = pd.timelines.size();
    for(size_t e=0; e<pd.timelines.size(); e++) {
      _num_workers += pd.timelines[e].segments.size();
      for(size_t w=0; w<pd.timelines[e].segments.size(); w++) {
        for(size_t l=0; l<pd.timelines[e].segments[w].size(); l++) {
          _num_tasks += pd.timelines[e].segments[w][l].size();

          // a new worker data
          WorkerData wd(e, w, l, stringify("E", e, ".W", w, ".L", l));

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
          _wdmap[wd.name] = _wd.size();
          _wd.push_back(std::move(wd));
        }
      }
    }
  }

  void query(
    std::ostream& os, 
    std::optional<ZoomX> zoomx, 
    std::optional<ZoomY> zoomy,
    ViewType method,
    size_t limit
  ) const {

    // Acquire the range of worker id
    std::vector<size_t> w;
    
    if(zoomy) {
      w.resize(zoomy->workers.size());
      for(size_t i=0; i<zoomy->workers.size(); i++) {
        auto itr = _wdmap.find(zoomy->workers[i]);
        if(itr == _wdmap.end()) {
          TF_THROW("failed to find worker ", zoomy->workers[i]);
        }
        w[i] = itr->second;
      }
    }
    else {
      w.resize(_wd.size());
      for(size_t i=0; i<_wd.size(); i++) {
        w[i] = i;
      }
    }

    if(!zoomx) {
      zoomx.emplace(_minX, _maxX);
    }

    /*std::cout << "query zoomX=[" << zoomx->beg << ", " << zoomx->end 
              << "] zoomY=[";
    for(size_t i=0; i<w.size(); ++i) {
      if(i) std::cout << ", ";
      std::cout << _wd[w[i]].name;
    }
    std::cout << "] limit=" << limit << '\n';*/

    std::vector<std::list<Cluster>> clusters{w.size()};
    ClusterHeap heap;
    
    // bsearch the range of segments for each worker data
    // TODO: parallel_for?
    for(size_t i=0; i<w.size(); i++) {

      // r = maxArg {span[0] <= zoomX[1]}
      auto r = _wd[w[i]].upper_bound(zoomx->end);
      if(r == std::nullopt) {
        continue;
      }
      
      // l = minArg {span[1] >= zoomX[0]}
      auto l = _wd[w[i]].lower_bound(zoomx->beg);
      if(l == std::nullopt || *l > *r) {
        continue;
      }
      
      // range ok
      for(size_t s=*l; s<=*r; s++) {
        
        switch(method) {
          case CLUSTER:
          if(s != *r) {
            clusters[i].emplace_back(
              i, 
              s, 
              s,
              _wd[w[i]].tasks[s+1].end - _wd[w[i]].tasks[s].beg
            );
          }
          else {  // boundary
            clusters[i].emplace_back(
              i, s, s, std::numeric_limits<size_t>::max()
            );
          }
          break;

          case CRITICALITY:
            clusters[i].emplace_back(i, s, s, _wd[w[i]].tasks[s].span());
          break;
        }

        heap.push(std::prev(clusters[i].end()));

        if(method == CRITICALITY) {
          while(heap.size() > limit) {
            auto top = heap.top();
            clusters[top->w].erase(top);
            heap.pop();
          }
        }
      }
      
      if(method == CLUSTER) {
        // while loop must sit after clustering is done
        // because we have std::next(top)-> = top->f
        while(heap.size() > limit) {

          auto top = heap.top(); 

          // if all clusters are in boundary - no need to cluster anymore
          if(top->k == std::numeric_limits<size_t>::max()) {
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
         << "\"worker\":\"" << _wd[w[i]].name << "\","
         << "\"tasks\":\"" << clusters[i].size() << "\","
         << "\"segs\": [";
      
      size_t T=0, loads[TASK_TYPES.size()] = {0};
      bool first_cluster = true;
        
      for(const auto& cluster : clusters[i]) {  

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
          os << "\"name\":\"(" << (cluster.t-cluster.f+1) << " tasks)\","
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

  size_t minX() const {
    return _minX;
  } 

  size_t maxX() const {
    return _maxX;
  }

  size_t num_tasks() const {
    return _num_tasks;
  }

  size_t num_executors() const {
    return _num_executors;
  }

  size_t num_workers() const {
    return _num_workers;
  }

  private:

    std::vector<WorkerData> _wd;

    size_t _minX {std::numeric_limits<size_t>::max()};
    size_t _maxX {std::numeric_limits<size_t>::min()};
    size_t _num_tasks {0};
    size_t _num_executors {0};
    size_t _num_workers {0};

    std::unordered_map<std::string, size_t> _wdmap;
};

}  // namespace tf ------------------------------------------------------------

int main(int argc, char* argv[]) {
  
  // parse arguments
  CLI::App app{"tfprof"};

  int port{8080};  
  app.add_option("-p,--port", port, "port to listen (default=8080)");

  std::string input;
  app.add_option("-i,--input", input, "input profiling file")
     ->required();

  std::string mount;
  app.add_option("-m,--mount", mount, "mount path to index.html")
     ->required();

  CLI11_PARSE(app, argc, argv);
  
  // create a database
  tf::Database db(input);

  // launc the server
  httplib::Server svr;

  auto ret = svr.set_mount_point("/", mount.c_str());
  if (!ret) {
    TF_THROW("failed to mount path to ", mount);
  }
  
  svr.Put("/queryInfo", [&db, &input](const httplib::Request& req, httplib::Response& res){
    std::cout << req.method << " /queryInfo" << std::endl;
   
    std::ostringstream oss;

    oss << "{\"tfpFile\":\"" << input << "\""
        << ",\"numTasks\":" << db.num_tasks() 
        << ",\"numExecutors\":" << db.num_executors()
        << ",\"numWorkers\":" << db.num_workers() << '}'; 

    res.set_content(oss.str().c_str(), "application/json");
  });

  svr.Put("/queryData", [&db](const httplib::Request& req, httplib::Response& res){
    
    std::cout << req.method << " /queryData" << '\n';
    std::cout << req.body << '\n';
    
    auto body = nlohmann::json::parse(req.body);

    const auto& jx = body["zoomX"];
    const auto& jy = body["zoomY"];
    const auto& jv = body["view"];
    size_t jl = body["limit"];

    std::optional<tf::Database::ZoomX> zoomx;
    std::optional<tf::Database::ZoomY> zoomy;
    tf::Database::ViewType view_type = tf::Database::CRITICALITY;

    if(jx.is_array() && jx.size() == 2) {
      zoomx.emplace(jx[0], jx[1]);
    }

    if(jy.is_array()) {
      zoomy.emplace();
      for(auto& w : jy) {
        zoomy->workers.push_back(std::move(w));
      }
    }

    if(jv == "Cluster") {
      view_type = tf::Database::CLUSTER;
    }

    std::ostringstream oss;
    db.query(oss, std::move(zoomx), std::move(zoomy), view_type, jl);

    res.set_content(oss.str().c_str(), "application/json");
  });

  svr.listen("0.0.0.0", port);

  return 0;
}


