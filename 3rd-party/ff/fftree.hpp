/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */


/* ***************************************************************************
 *
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */
/*
 * fftree.hpp
 *
 *  Created on: 1 Oct 2014
 *      Author: drocco
 *
 *  Modified:  
 *    - 20 Jul 2015   Massimo
 *      added mutex to protect the roots set
 *
 */
#ifndef FFTREE_HPP_
#define FFTREE_HPP_

#include <ostream>
#include <vector>
#include <set>
#include <ff/node.hpp>
#include <ff/platforms/platform.h>
#include <mutex>

namespace ff {

#undef DEBUG_FFTREE

//static pthread_mutex_t treeLock = PTHREAD_MUTEX_INITIALIZER;
static std::mutex treeLock;
static std::set<fftree *> roots;


union ffnode_ptr {
	ff_node *ffnode;
	ff_thread *ffthread;
};

struct threadcount_t {
	size_t n_emitter;
	size_t n_collector;
	size_t n_workers;
	size_t n_docomp;
};

struct fftree {
	bool isroot, do_comp, hasocl, hastpc;
	fftype nodetype;
	std::vector<std::pair<fftree *, bool> > children;
	ffnode_ptr ffnode;

	fftree(ff_node *ffnode_, fftype nodetype_) :
			nodetype(nodetype_) {
		do_comp = (nodetype == WORKER || nodetype == OCL_WORKER || nodetype == TPC_WORKER);
		ffnode.ffnode = ffnode_;
		isroot = ispattern();
        hasocl = (nodetype == OCL_WORKER);
        hastpc = (nodetype == TPC_WORKER);
		if (isroot){
			treeLock.lock();
			//pthread_mutex_lock(&treeLock);
#if defined(DEBUG_FFTREE)
            std::pair<std::set<fftree*>::iterator,bool> it;
			it = roots.insert(this);
            assert(it.second != false);
#else
            roots.insert(this);
#endif
			treeLock.unlock();
			//pthread_mutex_unlock(&treeLock);
        }
	}

	fftree(ff_thread *ffnode_, fftype nodetype_) :
			nodetype(nodetype_) {
		do_comp = (nodetype == WORKER || nodetype == OCL_WORKER);
        ffnode.ffthread = ffnode_;
		isroot = ispattern();
        hasocl = (nodetype == OCL_WORKER);
        hastpc = (nodetype == TPC_WORKER);
		if (isroot) {
			treeLock.lock();
            //pthread_mutex_lock(&treeLock);
#if defined(DEBUG_FFTREE)
            std::pair<std::set<fftree*>::iterator,bool> it;
			it = roots.insert(this);
            assert(it.second != false);
#else
            roots.insert(this);
#endif
			treeLock.unlock();
			//pthread_mutex_unlock(&treeLock);
        }
	}

	~fftree() {
		for (size_t i = 0; i < children.size(); ++i)
			if (children[i].first && !children[i].second)
				delete children[i].first;
		if (isroot) {
			treeLock.lock();
            //pthread_mutex_lock(&treeLock);
#if defined(DEBUG_FFTREE)
            std::set<fftree *>::iterator it = roots.find(this);
            assert(it != roots.end());
            roots.erase(it);
#else
            roots.erase(this);
#endif
			treeLock.unlock();
            //pthread_mutex_unlock(&treeLock);
        }
	}

	void add_child(fftree *t) {
		children.push_back(std::make_pair(t,t?t->ispattern():false));
		if (t && t->isroot) {
			treeLock.lock();
            //pthread_mutex_lock(&treeLock);
			roots.erase(t);
			treeLock.unlock();
            //pthread_mutex_unlock(&treeLock);
        }
	}

	void update_child(unsigned int idx, fftree *t) {
		if (children[idx].first && !children[idx].second)
			delete children[idx].first;
		children[idx] = std::make_pair(t,t?t->ispattern():false);
		if (t && t->isroot) {
			treeLock.lock();
            //pthread_mutex_lock(&treeLock);
			roots.erase(t);
			treeLock.unlock();
            //pthread_mutex_unlock(&treeLock);
        }
	}

	std::string fftype_tostr(fftype t) {
		switch (t) {
		case FARM:
			return "FARM";
		case PIPE:
			return "PIPE";
		case EMITTER:
			return "EMITTER";
		case WORKER:
			return "WORKER";
        case OCL_WORKER:
            return "OPENCL WORKER";
        case TPC_WORKER:
            return "TPC WORKER";
		case COLLECTOR:
			return "COLLECTOR";
		}
		return std::string("puppa"); //never reached
	}

	bool ispattern() const {
		return nodetype == FARM || nodetype == PIPE;
	}

    bool hasOpenCLNode() const { 
        if (hasocl) return true;
        for (size_t i = 0; i < children.size(); ++i)
            if (children[i].first && children[i].first->hasOpenCLNode()) return true;
        return false;
    }

    bool hasTPCNode() const { 
        if (hastpc) return true;
        for (size_t i = 0; i < children.size(); ++i)
            if (children[i].first && children[i].first->hasTPCNode()) return true;
        return false;
    }


	void print(std::ostream &os) {
		os << (ispattern() ? "[" : "(") << fftype_tostr(nodetype);
		if (do_comp)
			os << " docomp";
		for (size_t i = 0; i < children.size(); ++i) {
			if (children[i].first) {
				os << " ";
				children[i].first->print(os);
			}
		}
		os << (ispattern() ? "]" : ")");
	}

	size_t threadcount(threadcount_t *tc) const {
		size_t cnt = !ispattern();
		if (cnt) {
			tc->n_emitter   += nodetype == EMITTER;
			tc->n_collector += nodetype == COLLECTOR;
			tc->n_workers   += (nodetype == WORKER || nodetype == OCL_WORKER);
			tc->n_docomp    += do_comp;
		}
		for (size_t i = 0; i < children.size(); ++i)
			if (children[i].first)
				cnt += children[i].first->threadcount(tc);
		return cnt;
	}
};

static inline void print_fftrees(std::ostream &os) {
	size_t t = 0, total = 0;
	for (std::set<fftree *>::const_iterator it = roots.begin();
			it != roots.end(); ++it) {
		threadcount_t tc;
		os << "> ff tree @" << *it << "\n";
		(*it)->print(os);
		total += (t = (*it)->threadcount(&tc));
		os << "\nthread count = " << t << "\n";
	}
	os << "total thread count = " << total << "\n";
}

} // namespace

#endif /* FFTREE_HPP_ */
