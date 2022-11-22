#!/bin/bash
#
#
#
#  FastFlow is free software; you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License version 3 as
#  published by the Free Software Foundation.
#  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
#  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
#  License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#
# Author: Massimo Torquati
# 
# Requires: bash >=3.2, hwloc
#
# The script builds a string containing the list of logical core of the
# machine that are topologically contiguous, i.e the first context of the
# first core, the first context of the second core, and so on up to the
# last context of the last core.
# It also can set two variables:
# FF_NUM_CORES
# FF_NUM_REAL_CORES
#
# Example: Given the following OS numbering for the logical cores
#
#   CPU0:                 CPU1:
#    node0   || node1       node2 ||  node3
#   ------------------    ------------------ 
#   | 0 | 4 || 2 | 6 |    | 1 | 5 || 3 | 7 |
#   | 8 |12 ||10 |14 |    | 9 |13 ||11 |15 |
#   ------------------    ------------------
#
# the string produced is:
# "0,4,2,6,1,5,3,7,8,12,10,14,9,13,11,15".
# FF_NUM_CORES=16
# FF_NUM_REAL_CORES=8
#

topocmd=lstopo # or lstopo-no-graphics or hwloc-ls

# It checks if the topocmd is available
command -v $topocmd >/dev/null ||
    { echo >&2 "This script requires hwloc to be installed."; exit 1; }

physical=$($topocmd --only core | wc -l)  # gets the number of physical cores
logical=$($topocmd --only pu | wc -l)     # gets the number of logical cores
# this is the number of contexts per physical core
nway=$(($logical/$physical))    

# It reads lines from standard input into an indexed array variable.
# The right-hand side script returns the ids of the Processing Unit of
# the machine in the linear order.
# Considering the example above the topocmd command returns something like:
#   0 8 4 12 2 10 6 14 1 9 5 13 3 11 7 15.
# (We do not use mapfile command for portability on MacOS and bash<4)
while IFS= read -r line; do 
    array+=("$line")
done < <( $topocmd --only pu | awk -F'[#)]' '{print $3}' )

for((i=0;i<${#array[@]};i+=nway)); do
    for((j=0;j<$nway;++j)); do    
	V[j]+="${array[i+j]},"    
    done
done
for((j=0;j<$nway;++j)); do
     str+=${V[j]}
done
# remove the last comma
string=${str::${#str}-1}  # on bash>4 just ${str::-1}
echo "FF_MAPPING_STRING=\"$string\""
echo "FF_NUM_CORES=$logical"
echo "FF_NUM_REAL_CORES=$physical"

read -p "Do you want that I change the ./config.hpp file for you? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    sed -i -e "s/^#define FF_MAPPING_STRING \".*\"/#define FF_MAPPING_STRING \"$string\"/1" config.hpp
    if [ $? -eq 0 ]; then
	echo "This is the new FF_MAPPING_STRING variable in the ./config.hpp file:"
	echo -e "\033[1m $(grep -m1 "^#define FF_MAPPING_STRING \"" config.hpp) \033[0m"
    else
	echo "something went wrong when replacing the variable FF_MAPPING_STRING...."
	exit 1
    fi
    sed -i -e "s/^#define FF_NUM_CORES [-]\{0,1\}[[:digit:]].*/#define FF_NUM_CORES $logical/1" ./config.hpp
    if [ $? -eq 0 ]; then	      
    	echo "This is the new FF_NUM_CORES variable in the ./config.hpp file:"
	echo -e "\033[1m $(grep -m1 "^#define FF_NUM_CORES " config.hpp) \033[0m"
    else
	echo "something went wrong when replacing the variable FF_NUM_CORES...."
	exit 1
    fi
    sed -i -e "s/^#define FF_NUM_REAL_CORES [-]\{0,1\}[[:digit:]].*/#define FF_NUM_REAL_CORES $physical/1" ./config.hpp
    if [ $? -eq 0 ]; then
	echo "This is the new FF_NUM_REAL_CORES variable in the ./config.hpp file:"
	echo -e "\033[1m $(grep -m1 "^#define FF_NUM_REAL_CORES " config.hpp) \033[0m"	
    else
	echo "something went wrong when replacing the variable FF_NUM_REAL_CORES...."
	exit 1
    fi    
else
    echo "Ok, nothing has been changed"
fi

exit 0
