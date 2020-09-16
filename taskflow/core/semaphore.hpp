#pragma once

namespace tf {

class Constraint;

// ----------------------------------------------------------------------------
// Semaphore
// ----------------------------------------------------------------------------

/**
@class Semaphore

@brief handle to a concurrency constraint

A Semaphore is a handle object of a constraint which nodes in the
dependency graph must obey.  It provides a set of methods for users to
access and modify the attributes of the associated constraint.

*/
class Semaphore {

  friend class FlowBuilder;
  friend class Task;

  public:

  private:

    explicit Semaphore(Constraint* c) : _constraint(c) {
    }

    Constraint* _constraint {nullptr};

};

}  // end of namespace tf. ---------------------------------------------------
