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
 * Author: Massimo Torquati
 *
 * Date:    May   2016
 *
 */
#ifndef REPARA_BASE_KERNEL_TASK_HPP
#define REPARA_BASE_KERNEL_TASK_HPP

// enables REPARA support in the FastFlow run-time
#if !defined(FF_REPARA)
#error "FF_REPARA not defined"
#endif
// defines FastFlow blocking mode
#if !defined(BLOCKING_MODE)
#error "BLOCKING_MODE not defined"
#endif

#include <string>


namespace repara {
namespace rprkernels {

/** 
 * REPARA base kernel task
 *
 */
struct baseKernelTask {
    baseKernelTask():task_index{0} {}
    baseKernelTask(size_t idx, const std::string &cmd):
        task_index{idx}, cmd{cmd} {}

    size_t task_index;
    size_t kernel_id;
    std::string cmd;
};

};
};
#endif /* REPARA_BASE_KERNEL_TASK_HPP */
