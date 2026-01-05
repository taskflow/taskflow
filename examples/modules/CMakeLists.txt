# -----------------------------------------------------------------------------
# modules examples
# -----------------------------------------------------------------------------

list(APPEND TF_MODULE_EXAMPLES 
    modules_basic
    modules_task_visitor
)

foreach(module_example IN LISTS TF_MODULE_EXAMPLES)
    add_executable(${module_example} ${module_example}.cpp)
    target_link_libraries(${module_example}
        tf::module ${ATOMIC_LIBRARY} Threads::Threads tf::default_settings
    )
    target_compile_features(${module_example} PRIVATE cxx_std_23)
endforeach()
