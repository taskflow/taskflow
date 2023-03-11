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
#ifndef DISABLE_FF_DISTRIBUTED

#ifndef FF_DFF_HPP
#define FF_DFF_HPP

#define DFF_ENABLED

#if !defined(DFF_EXCLUDE_MPI)
#define DFF_MPI
#endif

#if !defined(DFF_EXCLUDE_BLOCKING)
#define BLOCKING_MODE
#else
#undef BLOCKING_MODE
#endif


// default size of the batching buffer
#if !defined(DEFAULT_BATCH_SIZE)
#define DEFAULT_BATCH_SIZE        1
#endif

// default number of On-The-Fly messages
#if !defined(DEFAULT_INTERNALMSG_OTF)
#define DEFAULT_INTERNALMSG_OTF  10
#endif
#if !defined(DEFAULT_MESSAGE_OTF)
#define DEFAULT_MESSAGE_OTF     100
#endif


#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_dgroups.hpp>

#include<ff/distributed/ff_dinterface.hpp>

#endif /* FF_DFF_HPP */

#else /* DISABLE_FF_DISTRIBUTED */

#if !defined(DFF_EXCLUDE_BLOCKING)
#define BLOCKING_MODE
#else
#undef BLOCKING_MODE
#endif

#include <ff/ff.hpp>
#include <ff/distributed/ff_dinterface.hpp>
#include <iostream>
namespace ff {
    std::ostream& cout = std::cout;

    template<class CharT, class Traits>
    auto& endl(std::basic_ostream<CharT, Traits>& os){return std::endl(os);}
}
static inline int DFF_Init(int& argc, char**& argv){ return 0; }
#endif /* DISABLE_FF_DISTRIBUTED */
