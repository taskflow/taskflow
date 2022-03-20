// 3rd-party include

#include <httplib/httplib.hpp>
#include <CLI11/CLI11.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <taskflow/taskflow.hpp>
#include <cmath>

namespace tf {

class Database {

  public:

  enum ViewType {
    CLUSTER = 0,
    CRITICALITY
  };

  struct WorkerData {
    size_t eid;
    size_t wid;
    size_t lid;
    std::string name;
    std::vector<Segment> tasks;

    WorkerData(
      size_t e, size_t w, size_t l, std::string n, std::vector<Segment> t
    ) :
      eid{e}, wid{w}, lid{l}, name {std::move(n)}, tasks{std::move(t)} {
    }

    WorkerData(const WorkerData&) = delete;
    WorkerData(WorkerData&&) = default;

    WorkerData& operator = (const WorkerData&) = delete;
    WorkerData& operator = (WorkerData&&) = default;

    std::optional<size_t> lower_bound(observer_stamp_t value) const {

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

    std::optional<size_t> upper_bound(observer_stamp_t value) const {

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

  struct Criticality {

    size_t i;
    std::vector<Segment>::const_iterator key;

    Criticality(size_t in_i, std::vector<Segment>::const_iterator in_key) :
      i{in_i}, key{in_key} {
    }
  };

  struct CriticalityComparator {
    bool operator () (const Criticality& a, const Criticality& b) const {
      return a.key->span() > b.key->span();
    }
  };

  struct CriticalityHeap : public std::priority_queue<
    Criticality, std::vector<Criticality>, CriticalityComparator
  > {

    void sort() {
      std::sort(c.begin(), c.end(), [] (const auto& a, const auto& b) {
        if(a.i == b.i) {
          return a.key->beg < b.key->beg;
        }
        return a.i < b.i;
      });
    }

    const std::vector<Criticality>& get() const {
      return c;
    }
  };

  struct Cluster {
    size_t i;
    size_t f;  // from task
    size_t t;  // to task   (inclusive)
    observer_stamp_t::duration k;  // key

    Cluster(size_t in_i, size_t in_f, size_t in_t, observer_stamp_t::duration in_k) :
      i{in_i}, f{in_f}, t{in_t}, k{in_k} {
    }

    using iterator_t = std::list<Cluster>::iterator;
  };

  struct ClusterComparator {
    bool operator () (Cluster::iterator_t a, Cluster::iterator_t b) const {
      return a->k > b->k;
    }
  };

  using ClusterHeap = std::priority_queue<
    Cluster::iterator_t, std::vector<Cluster::iterator_t>, ClusterComparator
  >;

  public:

  Database(const std::string& fpath) {

    std::ifstream ifs(fpath);

    if(!ifs) {
      TF_THROW("failed to open profile data ", fpath);
    }

    ProfileData pd;
    tf::Deserializer<std::ifstream> deserializer(ifs);
    deserializer(pd);

    // find the minimum starting point

    for(auto& timeline : pd.timelines) {
      if(timeline.origin < _minX) {
        _minX = timeline.origin;
      }
    }

    // conver to flat data
    _num_executors = pd.timelines.size();
    for(size_t e=0; e<pd.timelines.size(); e++) {
      _num_workers += pd.timelines[e].segments.size();
      for(size_t w=0; w<pd.timelines[e].segments.size(); w++) {
        for(size_t l=0; l<pd.timelines[e].segments[w].size(); l++) {
          // a new worker data
          WorkerData wd(
            e, w, l, stringify("E", e, ".W", w, ".L", l),
            std::move(pd.timelines[e].segments[w][l])
          );
          if(!wd.tasks.empty()) {
            if(wd.tasks.front().beg < _minX) _minX = wd.tasks.front().beg;
            if(wd.tasks.back().end > _maxX) _maxX = wd.tasks.back().end;
          }
          _num_tasks += wd.tasks.size();
          _wdmap[wd.name] = _wd.size();
          _wd.push_back(std::move(wd));
        }
      }
    }
  }

  template <typename D>
  void query_criticality(
    std::ostream& os,
    const std::optional<D>& xbeg, const std::optional<D>& xend,
    const std::optional<std::vector<std::string>>& workers,
    size_t limit
  ) const {

    auto x = decode_zoomx(xbeg, xend);
    auto w = decode_zoomy(workers);

    CriticalityHeap heap;

    // bsearch the range of segments for each worker data
    // TODO: parallel_for?
    for(size_t i=0; i<w.size(); i++) {

      // r = maxArg {span[0] <= zoomX[1]}
      auto r = _wd[w[i]].upper_bound(x.second);
      if(r == std::nullopt) {
        continue;
      }

      // l = minArg {span[1] >= zoomX[0]}
      auto l = _wd[w[i]].lower_bound(x.first);
      if(l == std::nullopt || *l > *r) {
        continue;
      }

      // range ok
      for(size_t s=*l; s<=*r; s++) {
        heap.emplace(i, _wd[w[i]].tasks.begin() + s);
        while(heap.size() > limit) {
          heap.pop();
        }
      }
    }

    heap.sort();

    auto& crits = heap.get();

    size_t cursor = 0;

    // Output the segments
    bool first_worker = true;
    os << "[";
    for(size_t i=0; i<w.size(); i++) {

      if(cursor < crits.size() && crits[cursor].i < i) {
        TF_THROW("impossible ...");
      }

      if(!first_worker) {
        os << ",";
      }
      else {
        first_worker = false;
      }

      os << "{\"executor\":\"" << _wd[w[i]].eid << "\","
         << "\"worker\":\"" << _wd[w[i]].name << "\","
         << "\"segs\": [";

      size_t T=0, loads[TASK_TYPES.size()] = {0}, n=0;
      bool first_crit = true;

      for(; cursor < crits.size() && crits[cursor].i == i; cursor++) {

        n++;

        if(!first_crit) {
          os << ",";
        }
        else {
          first_crit = false;
        }

        // single task
        os << "{";
        const auto& task = *crits[cursor].key;
        os << "\"name\":\"" << task.name << "\","
           << "\"type\":\"" << to_string(task.type) << "\","
           << "\"span\": [" << std::chrono::duration_cast<D>(task.beg-_minX).count()
                            << ","
                            << std::chrono::duration_cast<D>(task.end-_minX).count()
                            << "]";
        os << "}";

        // calculate load
        size_t t = std::chrono::duration_cast<D>(task.span()).count();
        T += t;
        loads[static_cast<int>(task.type)] += t;
      }
      os << "],\"tasks\":\"" << n << "\",";

      // load
      os << "\"load\":[";
      size_t x = 0;
      for(size_t k=0; k<TASK_TYPES.size(); k++) {
        auto type = static_cast<int>(TASK_TYPES[k]);
        if(k) os << ",";
        os << "{\"type\":\"" << to_string(TASK_TYPES[k]) << "\","
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

  template <typename D>
  void query_cluster(
    std::ostream& os,
    const std::optional<D>& xbeg, const std::optional<D>& xend,
    const std::optional<std::vector<std::string>>& workers,
    size_t limit
  ) const {

    auto x = decode_zoomx(xbeg, xend);
    auto w = decode_zoomy(workers);

    std::vector<std::list<Cluster>> clusters{w.size()};
    ClusterHeap heap;

    // bsearch the range of segments for each worker data
    // TODO: parallel_for?
    for(size_t i=0; i<w.size(); i++) {

      // r = maxArg {span[0] <= zoomX[1]}
      auto r = _wd[w[i]].upper_bound(x.second);
      if(r == std::nullopt) {
        continue;
      }

      // l = minArg {span[1] >= zoomX[0]}
      auto l = _wd[w[i]].lower_bound(x.first);
      if(l == std::nullopt || *l > *r) {
        continue;
      }

      // range ok
      for(size_t s=*l; s<=*r; s++) {
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
            i, s, s, observer_stamp_t::duration::max()
          );
        }
        heap.push(std::prev(clusters[i].end()));
      }

      // while loop must sit after clustering is done
      // because we have std::next(top)-> = top->f
      while(heap.size() > limit) {

        auto top = heap.top();

        // if all clusters are in boundary - no need to cluster anymore
        if(top->k == observer_stamp_t::duration::max()) {
          break;
        }

        // remove the top element and cluster it with the next
        heap.pop();

        // merge top with top->next
        std::next(top)->f = top->f;
        clusters[top->i].erase(top);
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
             << "\"type\":\"" << to_string(task.type) << "\","
             << "\"span\": [" << std::chrono::duration_cast<D>(task.beg-_minX).count()
                              << ","
                              << std::chrono::duration_cast<D>(task.end-_minX).count()
                              << "]";
        }
        else {
          const auto& ftask = _wd[w[i]].tasks[cluster.f];
          const auto& ttask = _wd[w[i]].tasks[cluster.t];
          os << "\"name\":\"(" << (cluster.t-cluster.f+1) << " tasks)\","
             << "\"type\":\"clustered\","
             << "\"span\": ["  << std::chrono::duration_cast<D>(ftask.beg-_minX).count()
                               << ","
                               << std::chrono::duration_cast<D>(ttask.end-_minX).count()
                               << "]";
        }
        os << "}";

        // calculate load
        // TODO optimization with DP
        for(size_t j=cluster.f; j<=cluster.t; j++) {
          size_t t = std::chrono::duration_cast<D>(_wd[w[i]].tasks[j].span()).count();
          T += t;
          loads[static_cast<int>(_wd[w[i]].tasks[j].type)] += t;
        }
      }
      os << "],";  // end segs

      // load
      os << "\"load\":[";
      size_t x = 0;
      for(size_t k=0; k<TASK_TYPES.size(); k++) {
        auto type = static_cast<int>(TASK_TYPES[k]);
        if(k) os << ",";
        os << "{\"type\":\"" << to_string(TASK_TYPES[k]) << "\","
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

  observer_stamp_t minX() const {
    return _minX;
  }

  observer_stamp_t maxX() const {
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

    // {std::numeric_limits<size_t>::max()};
    // {std::numeric_limits<size_t>::lowest()};

    observer_stamp_t _minX {observer_stamp_t::max()};
    observer_stamp_t _maxX {observer_stamp_t::min()};

    size_t _num_tasks {0};
    size_t _num_executors {0};
    size_t _num_workers {0};

    std::unordered_map<std::string, size_t> _wdmap;

    template <typename D>
    std::pair<observer_stamp_t, observer_stamp_t>
    decode_zoomx(std::optional<D> beg, std::optional<D> end) const {
      observer_stamp_t b = beg ? *beg + _minX : _minX;
      observer_stamp_t e = end ? *end + _minX : _maxX;
      return {b, e};
    }

    std::vector<size_t> decode_zoomy(std::optional<std::vector<std::string>> zoomy) const {
      if(zoomy) {
        std::vector<size_t> w(zoomy->size());
        for(size_t i=0; i<zoomy->size(); i++) {
          auto itr = _wdmap.find((*zoomy)[i]);
          if(itr == _wdmap.end()) {
            TF_THROW("failed to find worker ", (*zoomy)[i]);
          }
          w[i] = itr->second;
        }
        return w;
      }
      else {
        std::vector<size_t> w(_wd.size());
        for(size_t i=0; i<_wd.size(); i++) {
          w[i] = i;
        }
        return w;
      }
    }
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

  // change log pattern
  spdlog::set_pattern("[%^%L %D %H:%M:%S.%e%$] %v");
  spdlog::set_level(spdlog::level::debug); // Set global log level to debug

  spdlog::info("reading database {} ...", input);

  // create a database
  tf::Database db(input);
  spdlog::info(
    "read {} (#tasks={:d}, #executors={:d}, #workers={:d})",
    input, db.num_tasks(), db.num_executors(), db.num_workers()
  );

  // create a http server
  httplib::Server server;

  if(server.set_mount_point("/", mount.c_str())) {
    spdlog::info("mounted '/' to {}", mount);
  }
  else {
    spdlog::critical("failed to mount '/' to {}", mount);
  }

  // Put method: queryInfo
  server.Put("/queryInfo",
    [&db, &input](const httplib::Request& req, httplib::Response& res){
      spdlog::info(
        "/queryInfo: connected a new client {0}:{1:d}",
        req.remote_addr, req.remote_port
      );

      std::ostringstream oss;
      oss << "{\"tfpFile\":\"" << input << "\""
          << ",\"numTasks\":" << db.num_tasks()
          << ",\"numExecutors\":" << db.num_executors()
          << ",\"numWorkers\":" << db.num_workers() << '}';

      res.set_content(oss.str().c_str(), "application/json");
      spdlog::info("/queryInfo: sent {0:d} bytes", oss.str().size());
    }
  );

  // Put method: queryData
  server.Put("/queryData",
    [&db](const httplib::Request& req, httplib::Response& res){

      auto body = nlohmann::json::parse(req.body);

      const auto& jx = body["zoomX"];
      const auto& jy = body["zoomY"];
      const auto& jv = body["view"];
      size_t jl = body["limit"];

      spdlog::info(
        "/queryData: zoomX={}, zoomY=[...{} workers], view={}, limit={}",
        jx.dump(), jy.size(), jv.dump(), jl
      );

      std::optional<std::chrono::microseconds> xbeg, xend;
      std::optional<std::vector<std::string>> y;
      tf::Database::ViewType view_type = tf::Database::CLUSTER;

      if(jx.is_array() && jx.size() == 2) {
        xbeg = std::chrono::microseconds(std::llround((double)jx[0]));
        xend = std::chrono::microseconds(std::llround((double)jx[1]));
      }

      if(jy.is_array()) {
        y.emplace();
        for(auto& w : jy) {
          y->push_back(std::move(w));
        }
      }

      if(jv == "Criticality") {
        view_type = tf::Database::CRITICALITY;
      }

      std::ostringstream oss;

      switch(view_type) {
        case tf::Database::CRITICALITY:
          db.query_criticality<std::chrono::microseconds>(oss, xbeg, xend, y, jl);
        break;
        case tf::Database::CLUSTER:
          db.query_cluster<std::chrono::microseconds>(oss, xbeg, xend, y, jl);
        break;
      }

      res.set_content(oss.str().c_str(), "application/json");
      spdlog::info("/queryData: sent {0:d} bytes", oss.str().size());
    }
  );

  spdlog::info("listening to http://localhost:{:d} ...", port);
  server.listen("0.0.0.0", port);
  spdlog::info("shut down server");

  return 0;
}


