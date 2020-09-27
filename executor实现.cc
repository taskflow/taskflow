代码文件：executor.hpp


Executor::run(Taskflow& f)
Executor::run_until(Taskflow& f, P&& pred)
Executor::_schedule(PassiveVector<Node*>& nodes)
  for node : nodes
    // 把node放到任务队列
    // domain：STATIC_WORK DYNAMIC_WORK(动态子图) CONDITION_WORK
    _wsq[node->domain()].push(node)


// 执行节点
Executor::_invoke(Worker& worker, Node* node)
  调用work  // nstd::get<Node::StaticWork>(node->_handle).work()
  for suc_node : node->_successors
    if --(suc_node->_join_counter) == 0
      // 触发后继执行
      _schedule(suc_node)
