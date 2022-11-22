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

#ifndef FF_STENCIL_HPP
#define FF_STENCIL_HPP

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <ff/utils.hpp>
#include <ff/node.hpp>
#include <ff/parallel_for.hpp>


namespace ff {


template<typename T>
class stencilTask {
public:
    typedef T base_type;
    
    stencilTask():InTask(NULL),OutTask(NULL),Xsize(0),Ysize(0),Zsize(0),
                  Xstart(0),Xstop(0),Xstep(0),
                  Ystart(0),Ystop(0),Ystep(0),
                  Zstart(0),Zstop(0),Zstep(0),
                  Youtsize(0) {}
    stencilTask(T *t, 
                size_t Xsize, size_t Xstart, size_t Xstop, size_t Xstep):
        InTask(t),Xsize(Xsize),Ysize(1),Zsize(1),
        Xstart(Xstart),Xstop(Xstop),Xstep(Xstep),
        Ystart(0),Ystop(0),Ystep(0), 
        Zstart(0),Zstop(0),Zstep(0),
        Youtsize(0) {}
    stencilTask(T *t, 
                size_t Xsize, size_t Xstart, size_t Xstop, size_t Xstep, 
                size_t Ysize, size_t Ystart, size_t Ystop, size_t Ystep):
        InTask(t),Xsize(Xsize),Ysize(Ysize),Zsize(1),
        Xstart(Xstart),Xstop(Xstop),Xstep(Xstep),
        Ystart(Ystart),Ystop(Ystop),Ystep(Ystep),
        Zstart(0),Zstop(1),Zstep(0),
        Youtsize(0) {}
    stencilTask(T *t, 
                size_t Xsize, size_t Xstart, size_t Xstop, size_t Xstep, 
                size_t Ysize, size_t Ystart, size_t Ystop, size_t Ystep, 
                size_t Zsize, size_t Zstart, size_t Zstop, size_t Zstep):
        InTask(t),Xsize(Xsize),Ysize(Ysize),Zsize(Zsize),
        Xstart(Xstart),Xstop(Xstop),Xstep(Xstep),
        Ystart(Ystart),Ystop(Ystop),Ystep(Ystep),
        Zstart(Zstart),Zstop(Zstop),Zstep(Zstep),
        Youtsize(0) {}

    stencilTask& operator=(const stencilTask &t) { 
        InTask = t.getInPtr(); OutTask=t.getOutPtr();
        Xsize=t.X_size(); Ysize=t.Y_size(); Zsize=t.Z_size();
        Xstart=t.X_start(); Xstop=t.X_stop(); Xstep=t.X_step();
        Ystart=t.Y_start(); Ystop=t.Y_stop(); Ystep=t.Y_step();
        Zstart=t.Z_start(); Zstop=t.Z_stop(); Zstep=t.Z_step();
        Youtsize=t.Y_osize();
        return *this; 
    }

    
    inline size_t X_start()  const { return Xstart; }
    inline size_t X_stop()   const { return Xstop; }
    inline size_t X_step()   const { return Xstep; }
    inline size_t Y_start()  const { return Ystart; }
    inline size_t Y_stop()   const { return Ystop; }
    inline size_t Y_step()   const { return Ystep; }
    inline size_t Z_start()  const { return Zstart; }
    inline size_t Z_stop()   const { return Zstop; }
    inline size_t Z_step()   const { return Zstep; }
    inline size_t X_size()   const { return Xsize; }
    inline size_t Y_size()   const { return Ysize; }
    inline size_t Z_size()   const { return Zsize; }
    inline size_t Y_osize()  const { return Youtsize; }
    inline size_t size()     const { return X_size()*Y_size()*Z_size(); }
    inline size_t bytesize() const { return size()*sizeof(T); } 
    
    void   setX(size_t x1, size_t x2, size_t x3) {
        Xstart = x1; Xstop = x2; Xstep = x3;
    }
    void   setY(size_t y1, size_t y2, size_t y3) {
        Ystart = y1; Ystop = y2; Ystep = y3;
    }
    void   setZ(size_t z1, size_t z2, size_t z3) {
        Zstart = z1; Zstop = z2; Zstep = z3;
    }

    void   setInTask(T *t, size_t X) { 
        if (t) InTask=t; 
        Xsize=X; Ysize=1; Zsize=1;
    }
    void   setInTask(T *t, size_t X, size_t Y) {
        if (t) InTask=t; 
        Xsize=X; Ysize=Y; Zsize=1;
    }
    void   setInTask(T *t, size_t X, size_t Y, size_t Z) {
        if (t) InTask=t; 
        Xsize=X; Ysize=Y; Zsize=Z;
    }
    void   setOutTask(T *t, size_t Yout) { 
        if (t) OutTask=t; 
        Youtsize=Yout;
    }

    T*     getInPtr() const  { return InTask;}
    T*     getOutPtr() const { return OutTask;}
    
    inline void  swap() { T *tmp=InTask; InTask=OutTask; OutTask=tmp;}

protected:
    T *InTask;
    T *OutTask;
    size_t Xsize,Ysize,Zsize;
    size_t Xstart,Xstop,Xstep;
    size_t Ystart,Ystop,Ystep;
    size_t Zstart,Zstop,Zstep;
    size_t Youtsize;
};


template<typename T>
class stencil2D: public ff_node {
public:
    typedef ParallelForReduce<T> parloop_t;
private:
    typedef std::function<T(size_t i, size_t j, void *)>                                    init_F_t;
    typedef std::function<void(parloop_t &loopInit, T *M, 
                               const size_t Xsize, const size_t Ysize)>                     init2_F_t;
    typedef std::function<T(long i, long j, T *in, const size_t X, const size_t Y)>         compute_F_t;
    typedef std::function<T(long i, long j, T *in, const size_t X, const size_t Y, 
                            T& reduceVar)>                                                  reduce_F_t;
    typedef std::function<void(parloop_t &loopCompute, T *in, T *out, 
                               const size_t Xsize, const size_t Xstart, const size_t Xstop, 
                               const size_t Ysize, const size_t Ystart, const size_t Ystop, 
                               T& reduceVar)>                                               reduce2_F_t;
    typedef std::function<void(T *inout, const size_t X, const size_t Y, T& reduceVar)>     prepost_F_t;
    typedef std::function<void(T& reduceVar, T val)>                                        reduceOp_F_t;
    typedef std::function<bool(T reduceVar, const size_t iter)>                             iterCond_F_t;


    enum { DEFAULT_STENCIL_CHUNKSIZE = 8 };

    static void reduceOpDefault(T&, T) {}
    
public:
    stencil2D(T *Min, T *Mout, const size_t Xsize, const size_t Ysize, const size_t Youtsize,
              int nw, int Yradius=1,int Xradius=1,bool ghostcells=false, 
              const size_t chunksize=DEFAULT_STENCIL_CHUNKSIZE):
        oneShot(true), ghosts(ghostcells),nw(nw), Yradius(Yradius), Xradius(Xradius),
        chunkSize(chunksize),
        extraInitInParam(NULL),extraInitOutParam(NULL),
        initInF1(NULL), initOutF1(NULL),initInF2(NULL),initOutF2(NULL), 
        beforeFor(NULL), computeF(NULL), computeFReduce1(NULL), computeFReduce2(NULL), afterFor(NULL),
        reduceOp(reduceOpDefault), iterCondition(NULL), identityValue((T)0), reduceVar((T)0),
        iter(0), maxIter(1) { 

        Task.setInTask(Min, Xsize, Ysize);
        // TODO
        assert(Ysize==Youtsize);
        Task.setOutTask(Mout, Youtsize);
        
        skipfirstpop();
    }
    
    stencil2D(int nw, int Yradius=1, int Xradius=1,bool ghostcells=false,
              const size_t chunksize=DEFAULT_STENCIL_CHUNKSIZE):
        oneShot(false), ghosts(ghostcells),nw(nw),Yradius(Yradius),Xradius(Xradius),
        chunkSize(chunksize),
        extraInitInParam(NULL),extraInitOutParam(NULL),
        initInF1(NULL), initOutF1(NULL),initInF2(NULL),initOutF2(NULL), 
        beforeFor(NULL), computeF(NULL), computeFReduce1(NULL), computeFReduce2(NULL),
        afterFor(NULL),
        reduceOp(reduceOpDefault), iterCondition(NULL), identityValue((T)0), reduceVar((T)0),
        iter(0), maxIter(1),ploop(nw,true) { }
    
    ~stencil2D() {}
    
    void initInFunc(init_F_t F, void *extra) { 
        if (!oneShot) {
            error("stencil2D: initInFunc: the provided init function will not be called in this configuration\n");
        }
        initInF1 = F; extraInitInParam = extra; 
    } 
    void initInFuncAll(init2_F_t F) { 
        if (!oneShot) {
            error("stencil2D: initInFunc: the provided init function will not be called in this configuration\n");
        }
        initInF2 = F;
    } 
    void initOutFunc(init_F_t F, void *extra) { 
        if (!oneShot) {
            error("stencil2D: initOutFunc: the provided init function will not be called in this configuration\n");
        }
        initOutF1 = F; extraInitOutParam = extra; 
    } 
    void initOutFuncAll(init2_F_t F) { 
        if (!oneShot) {
            error("stencil2D: initOutFunc: the provided init function will not be called in this configuration\n");
        }
        initOutF2 = F; 
    } 

    void preFunc(prepost_F_t F)          { beforeFor       = F; }

    void computeFunc(reduce_F_t F,
                     size_t xstart=0, size_t xstop=0, size_t xstep=1,
                     size_t ystart=0, size_t ystop=0, size_t ystep=1,
                     size_t zstart=0, size_t zstop=0, size_t zstep=1) { 
        Task.setX(xstart,xstop?xstop:Task.X_size(),xstep);
        Task.setY(ystart,ystop?ystop:Task.Y_size(),ystep);
        Task.setZ(zstart,zstop?zstop:Task.Z_size(),zstep);
        computeFReduce1  = F; 
    }
    void computeFuncAll(reduce2_F_t F,
                     size_t xstart=0, size_t xstop=0, size_t xstep=1,
                     size_t ystart=0, size_t ystop=0, size_t ystep=1,
                     size_t zstart=0, size_t zstop=0, size_t zstep=1) { 
        Task.setX(xstart,xstop?xstop:Task.X_size(),xstep);
        Task.setY(ystart,ystop?ystop:Task.Y_size(),ystep);
        Task.setZ(zstart,zstop?zstop:Task.Z_size(),zstep);
        computeFReduce2  = F; 
    }
    void postFunc(prepost_F_t F)         { afterFor        = F; }
    
    void reduceFunc(iterCond_F_t I, size_t maxI, 
                    reduceOp_F_t R, T iV) { 
        iterCondition = I;
        maxIter = maxI;
        reduceOp = R; 
        identityValue=iV;
    }
    
    // swaps input and output matrices 
    inline void swap() {  Task.swap(); }
    
    inline void *svc(void *task) { 
        if (task != NULL) Task = *((stencilTask<T>*)task);
        T *Min  = Task.getInPtr();
        T *Mout = Task.getOutPtr();
        
        if (oneShot && (initInF1 || initInF2)) {
            const size_t& Xsize  = Task.X_size();
            const size_t& Ysize  = Task.Y_size();

            if (initInF1) {
                ploop.parallel_for(0,Xsize,1,chunkSize,[&](const long i) {
                        for(long j=0; j< Ysize; ++j)
                            Min[i*Xsize+j] = initInF1(i,j, extraInitInParam);
                    }, nw);
            } else {
                initInF2(ploop, Min, Xsize, Ysize);
            }
        }
        if (oneShot && (initOutF1 || initOutF2)) {
            const size_t& Xsize  = Task.X_size();
            const size_t& Ysize  = Task.Y_size();

            if (initOutF1) {
                ploop.parallel_for(0,Xsize,1,chunkSize, [&](const long i) {
                    for(long j=0; j< Ysize; ++j)
                        Mout[i*Xsize+j] = initOutF1(i,j, extraInitOutParam);
                }, nw);
            } else {
                initOutF2(ploop, Mout, Xsize, Ysize);
            }
        }

        {
            const size_t& Xsize  = Task.X_size();
            const size_t& Ysize  = Task.Y_size();
            const size_t& Xstart = Task.X_start();
            const size_t& Xstop  = Task.X_stop();
            const size_t& Xstep  = Task.X_step();
            const size_t& Ystart = Task.Y_start();
            const size_t& Ystop  = Task.Y_stop();
            const size_t& Ystep  = Task.Y_step();

            T rVar = reduceVar;            
            rVar = identityValue;
            iter = 0;
            swap(); // because of the next swap op 
            if (computeFReduce1) {
                do {
                    swap();
                    Min  = Task.getInPtr(); Mout = Task.getOutPtr();
                    
                    if (beforeFor) beforeFor(Min,Xsize, Ysize, rVar);
                    ploop.parallel_reduce(rVar,identityValue, Xstart,Xstop, Xstep, chunkSize,
                                          [&](const long i,T &rVar) {
                                              for(long j=Ystart; j< Ystop; j+=Ystep) {
                                                  Mout[i*Xsize+j] = computeFReduce1(i,j,Min,Xsize,Ysize,rVar);		
                                              }                                             
                                          }, reduceOp, nw);
                    

                    if (afterFor) afterFor(Mout,Xsize, Ysize, rVar);
                    
                } while(++iter<maxIter && iterCondition(rVar, iter));
            } else {
                do {
                    swap();
                    Min  = Task.getInPtr(); Mout = Task.getOutPtr();
                    
                    if (beforeFor) beforeFor(Min,Xsize, Ysize, rVar);
                    computeFReduce2(ploop, Min,Mout,Xsize, Xstart,Xstop, Ysize,Ystart,Ystop,rVar);		
                    if (afterFor) afterFor(Mout,Xsize, Ysize, rVar);
                    
                } while(++iter<maxIter && iterCondition(rVar, iter));
            }
            reduceVar = rVar;
        }   
        
        return (oneShot?NULL:task);
    }
    
    size_t  getIter() const { return iter; }
    const T& getReduceVar() const { return reduceVar; }
    
    virtual inline int run_and_wait_end() {
        if (isfrozen()) {
            stop();
            thaw();
            if (wait()<0) return -1;
            return 0;
        }
        stop();
        if (run()<0) return -1;
        if (wait()<0) return -1;
        return 0;
    }

protected:
    const bool   oneShot;
    const bool   ghosts;
    const int    nw;
    const int    Yradius;
    const int    Xradius;
    const size_t chunkSize;
    void      * extraInitInParam;
    void      * extraInitOutParam;

    init_F_t     initInF1;
    init_F_t     initOutF1;
    init2_F_t    initInF2;
    init2_F_t    initOutF2;
    prepost_F_t  beforeFor;
    compute_F_t  computeF;
    reduce_F_t   computeFReduce1;
    reduce2_F_t  computeFReduce2;
    prepost_F_t  afterFor;
    reduceOp_F_t reduceOp;
    iterCond_F_t iterCondition;
    T            identityValue;
    T            reduceVar;
    size_t       iter;
    size_t       maxIter;
    stencilTask<T> Task;

    parloop_t    ploop;
};
    
} // namespace

#endif /* FF_STENCIL_HPP */
