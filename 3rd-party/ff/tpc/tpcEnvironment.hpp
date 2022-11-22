/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file tpcEnvironment.hpp
 *  \ingroup aux_classes
 *
 *  \brief This file includes the basic support for TPC platforms
 *
 *  Realises a singleton class that keep the status of the TPC platform
 *  creates contexts, command queues etc.
 */

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
 * Massimo Torquati:   torquati@di.unipi.it
 *
 */

#ifndef FF_TPCENVIRONMENT_HPP
#define FF_TPCENVIRONMENT_HPP

#if defined(FF_TPC)

#include <cstdlib>
#include <pthread.h>
#include <atomic>

#include <ff/utils.hpp>
#include <ff/tpc/tpc_api.h>
using namespace rpr;
using namespace tpc;

namespace ff {


static pthread_mutex_t tpcInstanceMutex = PTHREAD_MUTEX_INITIALIZER;


/*!
 *  \class tpcEnvironment
 *  \ingroup aux_classes
 *
 *  \brief TPC platform inspection and setup
 *
 * \note Multiple TPC devices are currently not managed. 
 *
 */

class tpcEnvironment {
private:
    //NOTE: currently only one single TPC device is supported
    tpc_ctx_t     *ctx;
    tpc_dev_ctx_t *dev_ctx;

protected:
    tpcEnvironment():ctx(NULL),dev_ctx(NULL) {
        tpcId = 0;
        
        bool ok = false;
        tpc_res_t r = tpc_init(&ctx);
        if (r == TPC_SUCCESS) {
            // FIX: need to support multiple TPC devices
            r = tpc_create_device(ctx, 0, &dev_ctx, TPC_DEVICE_CREATE_FLAGS_NONE);
            ok = r == TPC_SUCCESS;
        }
        if (! ok) { 
            tpc_deinit(ctx);
            ctx = NULL, dev_ctx=NULL;
            error("tpcEnvironment::tpcEnvironment FATAL ERROR, unable to create TPC device\n");
            abort();
        }               
    }
 
public:
    ~tpcEnvironment() {
        if (ctx != NULL) {
            tpc_destroy_device(ctx,dev_ctx);
            tpc_deinit(ctx);
        }
    }
   
    static inline tpcEnvironment * instance() {
        while (!m_tpcEnvironment) {
            pthread_mutex_lock(&tpcInstanceMutex);
            if (!m_tpcEnvironment) {
                m_tpcEnvironment = new tpcEnvironment();
            }
            assert(m_tpcEnvironment);
            pthread_mutex_unlock(&tpcInstanceMutex);
        }
        return m_tpcEnvironment; 
    }
    unsigned long getTPCID() {  return ++tpcId; }

    tpc_dev_ctx_t *const getTPCDevice(bool exclusive=false) {
        return dev_ctx;
    }

private:
    tpcEnvironment(tpcEnvironment const&){};
    tpcEnvironment& operator=(tpcEnvironment const&){ return *this;};
private:    
    static tpcEnvironment * m_tpcEnvironment;
    std::atomic_long tpcId;
};
tpcEnvironment* tpcEnvironment::m_tpcEnvironment = NULL;

} // namespace

#else  // FF_TPC not defined

namespace ff {
class tpcEnvironment{
private:
    tpcEnvironment() {}
public:
    static inline tpcEnvironment * instance() { return NULL; }
};
} // namespace
#endif /* FF_TPC */
#endif /* FF_TPCENVIRONMENT_HPP */
