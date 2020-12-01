
# ==================================================================================================
# This file is part of the cuBLASt project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
# width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# ==================================================================================================
#
# Defines the following variables:
#   CUBLAS_FOUND          Boolean holding whether or not the cuBLAS library was found
#   CUBLAS_INCLUDE_DIRS   The CUDA and cuBLAS include directory
#   CUDA_LIBRARIES        The CUDA library
#   CUBLAS_LIBRARIES      The cuBLAS library
#
# In case CUDA is not installed in the default directory, set the CUDA_ROOT variable to point to
# the root of cuBLAS, such that 'cublas_v2.h' can be found in $CUDA_ROOT/include. This can either be
# done using an environmental variable (e.g. export CUDA_ROOT=/path/to/cuBLAS) or using a CMake
# variable (e.g. cmake -DCUDA_ROOT=/path/to/cuBLAS ..).
#
# ==================================================================================================

# Sets the possible install locations
set(CUBLAS_HINTS
  ${CUDA_ROOT}
  $ENV{CUDA_ROOT}
  $ENV{CUDA_TOOLKIT_ROOT_DIR}
)
set(CUBLAS_PATHS
  /usr
  /usr/local
  /usr/local/cuda
)

# Finds the include directories
find_path(CUBLAS_INCLUDE_DIRS
  NAMES cublas_v2.h cuda.h
  HINTS ${CUBLAS_HINTS}
  PATH_SUFFIXES include inc include/x86_64 include/x64
  PATHS ${CUBLAS_PATHS}
  DOC "cuBLAS include header cublas_v2.h"
)
mark_as_advanced(CUBLAS_INCLUDE_DIRS)

# Finds the libraries
find_library(CUDA_LIBRARIES
  NAMES cudart
  HINTS ${CUBLAS_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  PATHS ${CUBLAS_PATHS}
  DOC "CUDA library"
)
mark_as_advanced(CUDA_LIBRARIES)
find_library(CUBLAS_LIBRARIES
  NAMES cublas
  HINTS ${CUBLAS_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  PATHS ${CUBLAS_PATHS}
  DOC "cuBLAS library"
)
mark_as_advanced(CUBLAS_LIBRARIES)

# =============================================================================

# Notification messages
if(NOT CUBLAS_INCLUDE_DIRS)
  message(STATUS "Could NOT find 'cuBLAS.h', install CUDA/cuBLAS or set CUDA_ROOT")
endif()
if(NOT CUDA_LIBRARIES)
  message(STATUS "Could NOT find CUDA library, install it or set CUDA_ROOT")
endif()
if(NOT CUBLAS_LIBRARIES)
  message(STATUS "Could NOT find cuBLAS library, install it or set CUDA_ROOT")
endif()

# Determines whether or not cuBLAS was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuBLAS DEFAULT_MSG CUBLAS_INCLUDE_DIRS CUDA_LIBRARIES CUBLAS_LIBRARIES)

# =============================================================================
