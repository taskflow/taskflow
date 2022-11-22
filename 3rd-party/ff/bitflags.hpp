/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file bitflags.hpp
 *  \ingroup building_blocks
 *
 *
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
#ifndef FF_BITFLAGS_HPP
#define FF_BITFLAGS_HPP

#include <vector>
#include <string>
#include <cassert>
namespace ff {

/**
 * Flags used in the  \ref setInPtr and \ref setOutPtr methods for 
 * providing commands to the run-time concerning H2D and D2H data transfers 
 * and device memory allocation.
 *
 */
enum class CopyFlags    { DONTCOPY, COPY };
enum class ReuseFlags   { DONTREUSE,  REUSE };
enum class ReleaseFlags { DONTRELEASE, RELEASE };

struct MemoryFlags {  
    MemoryFlags():copy(CopyFlags::COPY),
                  reuse(ReuseFlags::DONTREUSE),
                  release(ReleaseFlags::DONTRELEASE) {}
    MemoryFlags(CopyFlags c, ReuseFlags r, ReleaseFlags f):copy(c),reuse(r),release(f) {}
    CopyFlags    copy;
    ReuseFlags   reuse;
    ReleaseFlags release;
};

using memoryflagsVector = std::vector<MemoryFlags>;

/**
 * cmd format:
 *  ...;$kernel_1;GPU:0; UF;SF;...;$kernel_2;CPU:0;.... ;$ .....
 *
 * S: send (COPYTO)
 * U: reUse (REUSE)
 * R: receive (COPYFROM)
 * F: free/remove (RELEASE)
 */
static inline const memoryflagsVector extractFlags(const std::string &cmd, const int kernel_id) {
    memoryflagsVector V;
    // no command in input, we just return a vector with one (default) entry
    if (cmd == "") { 
        V.resize(1);
        return V;
    }
    const std::string kid = "kernel_"+std::to_string(kernel_id);
    const char semicolon = ';';
    size_t n = cmd.rfind(kid);
    assert(n != std::string::npos);
    n = cmd.find_first_of(semicolon, n);   // first ';' after kernel_id
    assert(n != std::string::npos);
    n = cmd.find_first_of(semicolon, n+1); // first ';' after device_id
    assert(n != std::string::npos);

    size_t m = cmd.find_first_of('$', n+1);
    assert(m != std::string::npos);
    // gets just the sub-string of the command related to the memory flags 
    // starting and ending with ';' (i.e. ; UF;SF;...;)
    const std::string &flag_string = cmd.substr(n,m-n);
    assert(flag_string != "");

    n = 0;
    m = flag_string.find_first_of(semicolon,n+1);  
    assert(m != std::string::npos);
    V.reserve(10);    
    do {
        const std::string &flags = flag_string.substr(n+1, m-n-1);
        
        struct MemoryFlags mf;
        if (flags.find('S') != std::string::npos)      mf.copy = CopyFlags::COPY;
        else if (flags.find('R') != std::string::npos) mf.copy = CopyFlags::COPY;
        else                                           mf.copy = CopyFlags::DONTCOPY;
        mf.reuse   = (flags.find('U')!=std::string::npos ? ReuseFlags::REUSE     : ReuseFlags::DONTREUSE);
        mf.release = (flags.find('F')!=std::string::npos ? ReleaseFlags::RELEASE : ReleaseFlags::DONTRELEASE);
        V.push_back(mf);
        
        n = m;
        m = flag_string.find_first_of(semicolon, n+1);  
    } while(m != std::string::npos);
    
    return V;
}
    
static inline CopyFlags getCopy(int pos, const memoryflagsVector &V) {
    return V[pos].copy;
}
static inline ReuseFlags getReuse(int pos, const memoryflagsVector &V) {
    return V[pos].reuse;
}
static inline ReleaseFlags getRelease(int pos, const memoryflagsVector &V) {
    return V[pos].release;
}

// *******************************************************************


} // namespace


#endif /* FF_BITFLAGS_HPP */
