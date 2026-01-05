add_library(tf_module)

target_sources(tf_module
    PUBLIC
        FILE_SET CXX_MODULES FILES
            tf.cppm
            tf.algorithm.cppm
            tf.core.cppm
            tf.cuda.cppm
            tf.utility.cppm
)

target_compile_features(tf_module PUBLIC cxx_std_20)

target_include_directories(tf_module PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/taskflow>
    $<INSTALL_INTERFACE:include>
)

add_library(tf::module ALIAS tf_module)

# Installation
install(TARGETS tf_module
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CLI11/modules
)