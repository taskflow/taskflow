cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_library(tf_module)
set_property(TARGET tf_module PROPERTY INCLUDE_DIRECTORIES
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
)

target_sources(tf_module
    PUBLIC
        FILE_SET CXX_MODULES FILES
            tf.algorithm.cppm
            tf.core.cppm
            tf.utility.cppm
)

if(TF_BUILD_CUDA)
  target_sources(tf_module PUBLIC FILE_SET CXX_MODULES FILES tf_with_cuda.cppm tf.cuda.cppm)
else()
  target_sources(tf_module PUBLIC FILE_SET CXX_MODULES FILES tf.cppm)
endif()

target_compile_features(tf_module PUBLIC cxx_std_20)

target_link_libraries(tf_module PUBLIC Taskflow::Taskflow)

add_library(tf::module ALIAS tf_module)

# Installation
install(TARGETS tf_module
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/taskflow/modules
)
