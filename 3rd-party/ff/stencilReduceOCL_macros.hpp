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
 * stencilReduceOCL_macros.hpp
 *
 *  Created on: Feb 13, 2015
 *      Author: drocco
 */

#ifndef STENCILREDUCEOCL_MACROS_HPP_
#define STENCILREDUCEOCL_MACROS_HPP_

#include <string>
#include <sstream>

namespace ff {

#if 1 //explicit input, single-device

/*
 * both indexed and direct elemental function for 1D map.
 * f: N -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'basictype' is the element type
 * 'param' is the value of the input element
 * 'idx' is the index of the input element
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_MAP_ELEMFUNC(name, basictype, param, idx, ...)	\
static char name[] =\
"kern_" #name "|"\
#basictype "|"\
#basictype " f" #name "(" #basictype " " #param ",const int " #idx ") {\n" #__VA_ARGS__";\n}\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #basictype  "* input,\n"\
"\t__global " #basictype "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo) {\n"\
"\t    int i = get_global_id(0);\n"\
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i] = f" #name "(input[i],i);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"



/*
 * both indexed and direct elemental function for 1D map
 * with different input/output types.
 * f: N -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'outT' is the output element type
 * 'T' is the input element type
 * 'param' is the value of the input element
 * 'idx' is the index of the input element
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_MAP_ELEMFUNC_IO(name, outT, T, param, idx, ...)\
static char name[] =\
"kern_" #name "|"\
#outT "|"\
#outT " f" #name "(" #T " " #param ", const int " #idx ") {\n" #__VA_ARGS__";\n}\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #outT "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo) {\n"\
"\t    int i = get_global_id(0);\n"\
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i] = f" #name "(input[i],i);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"


/*
 * indexed elemental function for 1D stencil.
 * f: N -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type of the input
 * 'size' is the size of the input array
 * 'idx' is the index of the element
 * 'in' is the input array
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC(name,T,size,idx,in, ...)\
static char name[] =\
"kern_" #name "|"\
#T "|"\
"\n\n" #T " f" #name "(\n"\
"\t__global " #T "* " #in ",\n"\
"\tconst uint " #size ",\n"\
"\tconst int " #idx ") {\n"\
"\t   " #__VA_ARGS__";\n"\
"}\n\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo) {\n"\
"\t    int i = get_global_id(0);\n"\
"\t    int ig = i + offset;\n"\
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i+halo] = f" #name "(input+halo,inSize,ig);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"



/*
 * indexed elemental function for 1D stencil with read-only environment.
 * f: N -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type of the input
 * 'size' is the size of the input array
 * 'idx' is the index of the element
 * 'in' is the input array
 * 'env1T' is the element type of the constant environment array
 * 'env1' is the constant environment array
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC_ENV(name,T,size,idx,in,env1T,env1, ...)\
static char name[] =\
"kern_" #name "|"\
#T "|"\
"\n\n" #T " f" #name "(\n"\
"\t__global " #T "* " #in ",\n"\
"\tconst uint " #size ",\n"\
"\tconst int " #idx ",\n"\
"\t__global const " #env1T "* " #env1 ") {\n"\
"\t   " #__VA_ARGS__";\n"\
"}\n\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo,\n"\
"\t__global const " #env1T "* env1) {\n"\
"\t    int i = get_global_id(0);\n"\
"\t    int ig = i + offset;\n"\
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i+halo] = f" #name "(input+halo,inSize,ig,env1);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"



/*
 * indexed elemental function for 1D stencil with two read-only environments.
 * f: N -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type of the input
 * 'size' is the size of the input array
 * 'idx' is the index of the input element
 * 'in' is the input array
 * 'env1T' is the element type of the constant environment array
 * 'env1' is the constant environment array
 * 'env2T' is the element type of the constant environment value
 * 'env2' is (a pointer to) the constant environment value
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC_2ENV(name,T,size,idx,in,env1T,env1,env2T,env2, ...)\
static char name[] =\
"kern_" #name "|"\
#T "|"\
"\n\n" #T " f" #name "(\n"\
"\t__global " #T "* " #in ",\n"\
"\tconst uint " #size ",\n"\
"\tconst int " #idx ",\n"\
"\t__global const " #env1T "* " #env1 ",\n"\
"\t__global const " #env2T "* " #env2 ") {\n"\
"\t   " #__VA_ARGS__";\n"\
"}\n\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo,\n"\
"\t__global const " #env1T "* env1,\n"\
"\t__global const " #env2T "* env2) {\n"\
"\t    int i = get_global_id(0);\n"\
"\t    int ig = i + offset;\n"\
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i+halo] = f" #name "(input+halo,inSize,ig,env1,env2);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"

/* Same as the previous one, but with different input output types
 * indexed elemental function for 1D stencil with two read-only environments.
 * f: N -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type of the input
 * 'size' is the size of the input array
 * 'idx' is the index of the input element
 * 'in' is the input array
 * 'env1T' is the element type of the constant environment array
 * 'env1' is the constant environment array
 * 'env2T' is the element type of the constant environment value
 * 'env2' is (a pointer to) the constant environment value
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC_2ENV_IO(name,T,outT,size,idx,in,env1T,env1,env2T,env2, ...)\
static char name[] =\
"kern_" #name "|"\
#outT "|"\
"\n\n" #outT " f" #name "(\n"\
"\t__global " #T "* " #in ",\n"\
"\tconst uint " #size ",\n"\
"\tconst int " #idx ",\n"\
"\t__global const " #env1T "* " #env1 ",\n"\
"\t__global const " #env2T "* " #env2 ") {\n"\
"\t   " #__VA_ARGS__";\n"\
"}\n\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #outT "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo,\n"\
"\t__global const " #env1T "* env1,\n"\
"\t__global const " #env2T "* env2) {\n"\
"\t    int i = get_global_id(0);\n"\
"\t    int ig = i + offset;\n"\
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i+halo] = f" #name "(input+halo,inSize,ig,env1,env2);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"

#endif

#if 1 //implicit input, multi-device support
/*
 * This file contains macros for defining 1D and 2D elemental functions.
 * An instance of the elemental function is executed for each element
 * in the logical input array/matrix of a stencilReduceLoop task.
 *
 * An elemental function f can be either:
 * - direct: input is the value of the element (f: T -> T)
 * - indexed: input is the index of the element in the input array (f: N -> T)
 * Direct and indexed functions characterize, respectively, map and stencil tasks.
 *
 * Indexed elemental functions access elements via pre-defined macros:
 * - GET_IN(i) returns the i-th element in the input array
 * - GET_IN(i,j) returns the (i,j)-th element in the input matrix
 *
 * Some elemental functions are defined for working with read-only environments.
 * Macros are provided for accessing the environment from indexed
 * elemental functions:
 * - GET_ENV(i) for single-environment functions
 * - GET_ENV1(i), GET_ENV2(i) ... for multi-environment functions
 *
 * A simple macro for defining reduce combinator function is provided.
 */

/*
 * direct elemental function for 1D map.
 * f: T -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type
 * 'val' is the value of the input element
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_MAP_ELEMFUNC_1D(name, T, val, ...)\
static char name[] =\
"kern_" #name "|"\
#T "|"\
#T " f" #name "(" #T " " #val ") {\n" #__VA_ARGS__";\n}\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo) {\n"\
"\t    int i = get_global_id(0);\n"\
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i] = f" #name "(input[i]);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"



/*
 * direct elemental function for 1D map.
 * f: T -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type
 * 'val' is the value of the input element
 * 'envT' is the element type of the constant environment
 * 'envval' is the value of the environment element
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_MAP_ELEMFUNC_1D_ENV(name, T, val, envT, envval, ...)\
static char name[] =\
"kern_" #name "|"\
#T "|\n"\
#T " f" #name "(" #T " " #val ", " #envT " " #envval ") {\n" #__VA_ARGS__";\n}\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo,\n"\
"\t__global const " #envT "* env) {\n"\
"\t    int i = get_global_id(0);\n"\
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i] = f" #name "(input[i], env[i]);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"



/*
 * direct elemental function for 1D map with different input/output types.
 * f: T1 -> T2
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the input element type
 * 'outT' is the output element type
 * 'val' is the value of the input element
 * 'idx' is the index of the input element
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_MAP_ELEMFUNC_1D_IO(name, T, outT, val, idx, ...)	\
static char name[] =\
"kern_" #name "|"\
#outT "|\n\n"\
#outT " f" #name "("#T" "#val ",\n"\
"\tconst int " #idx "\n"\
") {\n" #__VA_ARGS__";\n}\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #outT "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo) {\n"\
"\t    int i = get_global_id(0);\n"\
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i] = f" #name "(input[i], i);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"


/*
 * direct elemental function for 1D map.
 * f: T -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type
 * 'val' is the value of the input element
 * 'envT' is the element type of the constant environment
 * 'envval' is the value of the environment element
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_MAP_ELEMFUNC_1D_ENV_IO(name, T, val, envT, envval, idx, ...) \
static char name[] =\
"kern_" #name "|"\
#T "|\n"\
#T " f" #name "(" #T " " #val ", " #envT " " #envval ") {\n" #__VA_ARGS__";\n}\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo,\n"\
"\t__global const " #envT "* env) {\n"\
"\t    int i = get_global_id(0);\n"\
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i] = f" #name "(input[i], env[i], i);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"



/*
 * indexed elemental function for 1D stencil.
 * f: N -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type of the input
 * 'size' is the size of the input array (for bound checking)
 * 'idx' is the index of the element
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC_1D(name,T,size,idx, ...)\
static char name[] =\
"kern_" #name "|"\
#T "|"\
"\n\n"\
"#define GET_IN(i) (in[i-offset])\n"\
"#define GET_ENV1(i) (env1[i])\n"\
#T " f" #name "(\n"\
"\t__global " #T "* in,\n"\
"\tconst uint " #size ",\n"\
"\tconst int " #idx ",\n"\
"\tconst int offset\n) {\n"\
"\t   " #__VA_ARGS__";\n"\
"}\n\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo) {\n"\
"\t    size_t i = get_global_id(0);\n"\
"\t    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i+halo] = f" #name "(input+halo,inSize,i+offset,offset);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"



/*
 * indexed elemental function for 1D stencil with read-only environment.
 * f: N -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type of the input
 * 'size' is the size of the input array (for bound checking)
 * 'idx' is the index of the element
 * 'env1T' is the element type of the constant environment array
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC_1D_ENV(name,T,size,idx,env1T, ...)\
static char name[] =\
"kern_" #name "|"\
#T "|"\
"\n\n"\
"#define GET_IN(i) (in[(i)-offset])\n"\
"#define GET_ENV(i) (env[(i)])\n"\
#T " f" #name "(\n"\
"\t__global " #T "* in,\n"\
"\tconst uint " #size ",\n"\
"\tconst int " #idx ",\n"\
"\tconst int offset,\n"\
"\t__global const " #env1T "* env) {\n"\
"\t   " #__VA_ARGS__";\n"\
"}\n\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo,\n"\
"\t__global const " #env1T "* env) {\n"\
"\t    size_t i = get_global_id(0);\n"\
"\t    size_t ig = i + offset;\n"\
"\t    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i+halo] = f" #name "(input+halo,inSize,ig,offset,env);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"



/*
 * indexed elemental function for 1D stencil with two read-only environments.
 * f: N -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type of the input
 * 'size' is the size of the input array (for bound checking)
 * 'idx' is the index of the element
 * 'env1T' is the element type of the first constant environment array
 * 'env2T' is the element type of the second constant environment array
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC_1D_2ENV(name,T,size,idx,env1T,env2T, ...)\
static char name[] =\
"kern_" #name "|"\
#T "|"\
"\n\n"\
"#define GET_IN(i) (in[(i)-offset])\n"\
"#define GET_ENV1(i) (env1[i])\n"\
"#define GET_ENV2(i) (env2[i])\n"\
#T " f" #name "(\n"\
"\t__global " #T "* in,\n"\
"\tconst uint " #size ",\n"\
"\tconst int " #idx ",\n"\
"\tconst int offset,\n"\
"\t__global const " #env1T "* env1,\n"\
"\t__global const " #env2T "* env2) {\n"\
"\t   " #__VA_ARGS__";\n"\
"}\n\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inSize,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo,\n"\
"\t__global const " #env1T "* env1,\n"\
"\t__global const " #env1T "* env2) {\n"\
"\t    size_t i = get_global_id(0);\n"\
"\t    size_t ig = i + offset;\n"\
"\t    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i+halo] = f" #name "(input+halo,inSize,ig,offset,env1,env2);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"





/*
 * indexed elemental function for 2D stencil.
 * f: (N,N) -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type of the input
 * 'height' is the number of rows in the input array (for bound checking)
 * 'width' is the number of columns in the input array  (for bound checking)
 * 'row' is the row-index of the element
 * 'col' is the column-index of the element
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC_2D(name,T,height,width,row,col, ...)\
static char name[] =\
"kern_" #name "|"\
#T "|"\
"\n\n"\
"#define GET_IN(i,j) (in[((i)*"#width"+(j))-offset])\n"\
"#define GET_ENV1(i,j) (env1[((i)*"#width"+(j))])\n"\
#T " f" #name "(\n"\
"\t__global " #T "* in,\n"\
"\tconst uint " #height ",\n"\
"\tconst uint " #width ",\n"\
"\tconst int " #row ",\n"\
"\tconst int " #col ",\n"\
"\tconst int offset) {\n"\
"\t   " #__VA_ARGS__";\n"\
"}\n\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inHeight,\n"\
"\tconst uint inWidth,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo) {\n"\
"\t    size_t i = get_global_id(0);\n"\
"\t    size_t ig = i + offset;\n"\
"\t    size_t r = ig / inWidth;\n"\
"\t    size_t c = ig % inWidth;\n"\
"\t    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i+halo] = f" #name "(input+halo,inHeight,inWidth,r,c,offset);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"

/*
 * indexed elemental function for 2D stencil.
 * f: (N,N) -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type of the input
 * 'height' is the number of rows in the input array (for bound checking)
 * 'width' is the number of columns in the input array  (for bound checking)
 * 'row' is the row-index of the element
 * 'col' is the column-index of the element
 * 'env1T' is the element type of the constant environment array
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC_2D_ENV(name,T,height,width,row,col,env1T,...)\
static char name[] =\
"kern_" #name "|"\
#T "|"\
"\n\n"\
"#define GET_IN(i,j) (in[((i)*"#width"+(j))-offset])\n"\
"#define GET_ENV(i,j) (env1[((i)*"#width"+(j))])\n"\
#T " f" #name "(\n"\
"\t__global " #T "* in,\n"\
"\tconst uint " #height ",\n"\
"\tconst uint " #width ",\n"\
"\tconst int " #row ",\n"\
"\tconst int " #col ",\n"\
"\tconst int offset,\n"\
"\t__global const " #env1T "* env) {\n"\
"\t   " #__VA_ARGS__";\n"\
"}\n\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inHeight,\n"\
"\tconst uint inWidth,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo,\n"\
"\t__global const " #env1T "* env) {\n"\
"\t    size_t i = get_global_id(0);\n"\
"\t    size_t ig = i + offset;\n"\
"\t    size_t r = ig / inWidth;\n"\
"\t    size_t c = ig % inWidth;\n"\
"\t    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i+halo] = f" #name "(input+halo,inHeight,inWidth,r,c,offset,env);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"

/*
 * indexed elemental function for 2D stencil.
 * f: (N,N) -> T
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'T' is the element type of the input
 * 'height' is the number of rows in the input array (for bound checking)
 * 'width' is the number of columns in the input array  (for bound checking)
 * 'row' is the row-index of the element
 * 'col' is the column-index of the element
 * 'env1T' is the element type of the first constant environment array
 * 'env2T' is the element type of the second constant environment array
 * '...' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC_2D_2ENV(name,T,height,width,row,col,env1T,env2T,...)\
static char name[] =\
"kern_" #name "|"\
#T "|"\
"\n\n"\
"#define GET_IN(i,j) (in[((i)*"#width"+(j))-offset])\n"\
"#define GET_ENV1(i,j) (env1[((i)*"#width"+(j))])\n"\
"#define GET_ENV2(i,j) (env2[((i)*"#width"+(j))])\n"\
#T " f" #name "(\n"\
"\t__global " #T "* in,\n"\
"\tconst uint " #height ",\n"\
"\tconst uint " #width ",\n"\
"\tconst int " #row ",\n"\
"\tconst int " #col ",\n"\
"\tconst int offset,\n"\
"\t__global const " #env1T "* env1,\n"\
"\t__global const " #env2T "* env2) {\n"\
"\t   " #__VA_ARGS__";\n"\
"}\n\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #T "* output,\n"\
"\tconst uint inHeight,\n"\
"\tconst uint inWidth,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo,\n"\
"\t__global const " #env1T "* env1,\n"\
"\t__global const " #env2T "* env2) {\n"\
"\t    size_t i = get_global_id(0);\n"\
"\t    size_t ig = i + offset;\n"\
"\t    size_t r = ig / inWidth;\n"\
"\t    size_t c = ig % inWidth;\n"\
"\t    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i+halo] = f" #name "(input+halo,inHeight,inWidth,r,c,offset,env1,env2);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"



#define FF_OCL_STENCIL_ELEMFUNC_2D_IO(name,T, outT, height,width,row,col, ...) \
static char name[] =\
"kern_" #name "|"\
#T "|"\
"\n\n"\
"#define GET_IN(i,j) (in[((i)*"#width"+(j))-offset])\n"\
#outT " f" #name "(\n"\
"\t__global " #T "* in,\n"\
"\tconst uint " #height ",\n"\
"\tconst uint " #width ",\n"\
"\tconst int " #row ",\n"\
"\tconst int " #col ",\n"\
"\tconst int offset) {\n"\
"\t   " #__VA_ARGS__";\n"\
"}\n\n"\
"__kernel void kern_" #name "(\n"\
"\t__global " #T  "* input,\n"\
"\t__global " #outT "* output,\n"\
"\tconst uint inHeight,\n"\
"\tconst uint inWidth,\n"\
"\tconst uint maxItems,\n"\
"\tconst uint offset,\n"\
"\tconst uint halo) {\n"\
"\t    size_t i = get_global_id(0);\n"\
"\t    size_t ig = i + offset;\n"\
"\t    size_t r = ig / inWidth;\n"\
"\t    size_t c = ig % inWidth;\n"\
"\t    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"\
"\t    while(i < maxItems)  {\n"\
"\t        output[i+halo] = f" #name "(input+halo,inHeight,inWidth,r,c,offset);\n"\
"\t        i += gridSize;\n"\
"\t    }\n"\
"}\n"



#endif



//  x=f(param1,param2)   'x', 'param1', 'param2' have the same type
#define FF_OCL_REDUCE_COMBINATOR(name, basictype, param1, param2, ...)\
static char name[] =\
"kern_" #name "|"\
#basictype "|"\
#basictype " f" #name "(" #basictype " " #param1 ",\n"\
                          #basictype " " #param2 ") {\n" #__VA_ARGS__";\n}\n"\
"__kernel void kern_" #name "(__global " #basictype "* input, const uint halo, __global " #basictype "* output, const uint n, __local " #basictype "* sdata, "#basictype" idElem) {\n"\
"        uint blockSize = get_local_size(0);\n"\
"        uint tid = get_local_id(0);\n"\
"        uint i = get_group_id(0)*blockSize + get_local_id(0);\n"\
"        uint gridSize = blockSize*get_num_groups(0);\n"\
"        " #basictype " result = idElem; input += halo;\n"\
"        if(i < n) { result = input[i]; i += gridSize; }\n"\
"        while(i < n) {\n"\
"          result = f" #name "(result, input[i]);\n"\
"          i += gridSize;\n"\
"        }\n"\
"        sdata[tid] = result;\n"\
"        barrier(CLK_LOCAL_MEM_FENCE);\n"\
"        if(blockSize >= 512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"\
"        if(blockSize >= 256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"\
"        if(blockSize >= 128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"\
"        if(blockSize >=  64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"\
"        if(blockSize >=  32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"\
"        if(blockSize >=  16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"\
"        if(blockSize >=   8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"\
"        if(blockSize >=   4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"\
"        if(blockSize >=   2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }\n"\
"        if(tid == 0) output[get_group_id(0)] = sdata[tid];\n"\
"}\n";



#define FFGENERICFUNC(name, basictype, ...)\
    static char name[] =\
        "kern_" #name "|"\
        #basictype "|"\
        #__VA_ARGS__";\n\n"


/* ------------------------------------------------------------------------------------- */



// NOTE: A better check would be needed !
// both GNU g++ and Intel icpc define __GXX_EXPERIMENTAL_CXX0X__ if -std=c++0x or -std=c++11 is used 
// (icpc -E -dM -std=c++11 -x c++ /dev/null | grep GXX_EX)
#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || defined(HAS_CXX11_VARIADIC_TEMPLATES)

template< typename ... Args >
std::string stringer(Args const& ... args ) {
    std::ostringstream stream;
    using List= int[];
    
    // expanding a parameter pack is only valid in contexts 
    // where the parser expects a comma-separated list of entries
    (void)List{0, ( (void)(stream << args), 0 ) ... };
    
    return stream.str();
}
    
#define STRINGER(...) stringer(__VA_ARGS__)
    
#endif // c++11 check




}  // namespace

#endif /* STENCILREDUCEOCL_MACROS_HPP_ */
