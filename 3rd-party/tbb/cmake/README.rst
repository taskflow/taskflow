.. contents::

Introduction
------------
Many developers use CMake to manage their development projects, so the Threading Building Blocks (TBB)
team created the set of CMake modules to simplify integration of the TBB library into a CMake project.
The modules are available starting from TBB 2017 U7 in `<tbb_root>/cmake <https://github.com/01org/tbb/tree/tbb_2017/cmake>`_.

About TBB
^^^^^^^^^^^^^^^
TBB is a library that supports scalable parallel programming using standard ISO C++ code. It does not require special languages or compilers. It is designed to promote scalable data parallel programming. Additionally, it fully supports nested parallelism, so you can build larger parallel components from smaller parallel components. To use the library, you specify tasks, not threads, and let the library map tasks onto threads in an efficient manner.

Many of the library interfaces employ generic programming, in which interfaces are defined by requirements on types and not specific types. The C++ Standard Template Library (STL) is an example of generic programming. Generic programming enables TBB to be flexible yet efficient. The generic interfaces enable you to customize components to your specific needs.

The net result is that TBB enables you to specify parallelism far more conveniently than using raw threads, and at the same time can improve performance.

References
^^^^^^^^^^
* `Official TBB open source site <https://www.threadingbuildingblocks.org/>`_
* `Official GitHub repository <https://github.com/01org/tbb>`_

Engineering team contacts
^^^^^^^^^^^^^^^^^^^^^^^^^
The TBB team is very interested in convenient integration of the TBB library into customer projects. These CMake modules were created to provide such a possibility for CMake projects using a simple but powerful interface. We hope you will try these modules and we are looking forward to receiving your feedback!

E-mail us: `inteltbbdevelopers@intel.com <mailto:inteltbbdevelopers@intel.com>`_.

Visit our `forum <https://software.intel.com/en-us/forums/intel-threading-building-blocks/>`_.

Release Notes
-------------
* Minimum supported CMake version: ``3.0.0``.
* TBB versioning via `find_package <https://cmake.org/cmake/help/latest/command/find_package.html>`_ has the following format: ``find_package(TBB <major>.<minor> ...)``.

Use cases of TBB integration into CMake-aware projects
------------------------------------------------------------
There are two types of TBB packages:
 * Binary packages with pre-built binaries for Windows* OS, Linux* OS and macOS*. They are available on the releases page of the Github repository: https://github.com/01org/tbb/releases. The main purpose of the binary package integration is the ability to build TBB header files and binaries into your CMake-aware project.
 * A source package is also available to download from the release page via the "Source code" link. In addition, it can be cloned from the repository by ``git clone https://github.com/01org/tbb.git``. The main purpose of the source package integration is to allow you to do a custom build of the TBB library from the source files and then build that into your CMake-aware project.

There are four types of CMake modules that can be used to integrate TBB: `TBBConfig`, `TBBGet`, `TBBMakeConfig` and `TBBBuild`. See `Technical documentation for CMake modules`_ section for additional details.

Binary package integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following use case is valid for packages starting from TBB 2017 U7:

* Download package manually and make integration.

 Pre-condition: Location of TBBConfig.cmake is available via ``TBB_DIR`` or ``CMAKE_PREFIX_PATH`` contains path to TBB root.

 CMake code for integration:
  .. code:: cmake

   find_package(TBB <options>)

The following use case is valid for all TBB 2017 packages.

* Download package using TBBGet_ and make integration.

 Pre-condition: TBB CMake modules are available via <path-to-tbb-cmake-modules>.

 CMake code for integration:
  .. code:: cmake

   include(<path-to-tbb-cmake-modules>/TBBGet.cmake)
   tbb_get(TBB_ROOT tbb_root CONFIG_DIR TBB_DIR)
   find_package(TBB <options>)

Source package integration
^^^^^^^^^^^^^^^^^^^^^^^^^^
* Build TBB from existing source files using TBBBuild_ and make integration.

 Pre-condition: TBB source code is available via <tbb_root> and TBB CMake modules are available via <path-to-tbb-cmake-modules>.

 CMake code for integration:
  .. code:: cmake

   include(<path-to-tbb-cmake-modules>/TBBBuild.cmake)
   tbb_build(TBB_ROOT <tbb_root> CONFIG_DIR TBB_DIR)
   find_package(TBB <options>)

* Download TBB source files using TBBGet_, build it using TBBBuild_ and make integration.

 Pre-condition: TBB CMake modules are available via <path-to-tbb-cmake-modules>.

 CMake code for integration:
  .. code:: cmake

   include(<path-to-tbb-cmake-modules>/TBBGet.cmake)
   include(<path-to-tbb-cmake-modules>/TBBBuild.cmake)
   tbb_get(TBB_ROOT tbb_root SOURCE_CODE)
   tbb_build(TBB_ROOT ${tbb_root} CONFIG_DIR TBB_DIR)
   find_package(TBB <options>)

Tutorials: TBB integration using CMake
--------------------------------------------
Binary TBB integration to the sub_string_finder sample (Windows* OS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we will integrate binary TBB package into the sub_string_finder sample on Windows* OS (Microsoft* Visual Studio).
This example is also applicable for other platforms with slight changes.
Place holders <version> and <date> should be replaced with the actual values for the TBB package being used. The example is written for `CMake 3.7.1`.

Precondition:
  * `Microsoft* Visual Studio 11` or higher.
  * `CMake 3.0.0` or higher.

#. Download the latest binary package for Windows from `this page <https://github.com/01org/tbb/releases/latest>`_ and unpack it to the directory ``C:\demo_tbb_cmake``.
#. In the directory ``C:\demo_tbb_cmake\tbb<version>_<date>oss\examples\GettingStarted\sub_string_finder`` create ``CMakeLists.txt`` file with the following content:
    .. code:: cmake

        cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

        project(sub_string_finder CXX)
        add_executable(sub_string_finder sub_string_finder.cpp)

        # find_package will search for available TBBConfig using variables CMAKE_PREFIX_PATH and TBB_DIR.
        find_package(TBB REQUIRED tbb)

        # Link TBB imported targets to the executable;
        # "TBB::tbb" can be used instead of "${TBB_IMPORTED_TARGETS}".
        target_link_libraries(sub_string_finder ${TBB_IMPORTED_TARGETS})
#. Run CMake GUI and:
    * Fill the following fields (you can use the buttons ``Browse Source...`` and ``Browse Build...`` accordingly)

     * Where is the source code: ``C:/demo_tbb_cmake/tbb<version>_<date>oss/examples/GettingStarted/sub_string_finder``
     * Where to build the binaries: ``C:/demo_tbb_cmake/tbb<version>_<date>oss/examples/GettingStarted/sub_string_finder/build``

    * Add new cache entry using button ``Add Entry`` to let CMake know where to search for TBBConfig:

     * Name: ``CMAKE_PREFIX_PATH``
     * Type: ``PATH``
     * Value: ``C:/demo_tbb_cmake/tbb<version>_<date>oss``

    * Push the button ``Generate`` and choose a proper generator for your Microsoft* Visual Studio version.
#. Now you can open the generated solution ``C:/demo_tbb_cmake/tbb<version>_<date>oss/examples/GettingStarted/sub_string_finder/build/sub_string_finder.sln`` in your Microsoft* Visual Studio and build it.

Source code integration of TBB to the sub_string_finder sample (Linux* OS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we will build TBB from source code with enabled Community Preview Features and link the sub_string_finder sample with the built library.
This example is also applicable for other platforms with slight changes.

Precondition:
  * `CMake 3.0.0` or higher.
  * `Git` (to clone the TBB repository from GitHub)

#. Create the directory ``~/demo_tbb_cmake``, go to the created directory and clone the TBB repository there:
    ``mkdir ~/demo_tbb_cmake ; cd ~/demo_tbb_cmake ; git clone https://github.com/01org/tbb.git``
#. In the directory ``~/demo_tbb_cmake/tbb/examples/GettingStarted/sub_string_finder`` create ``CMakeLists.txt`` file with following content:
    .. code:: cmake

     cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

     project(sub_string_finder CXX)
     add_executable(sub_string_finder sub_string_finder.cpp)

     include(${TBB_ROOT}/cmake/TBBBuild.cmake)

     # Build TBB with enabled Community Preview Features (CPF).
     tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR TBB_DIR MAKE_ARGS tbb_cpf=1)

     find_package(TBB REQUIRED tbb_preview)

     # Link TBB imported targets to the executable;
     # "TBB::tbb_preview" can be used instead of "${TBB_IMPORTED_TARGETS}".
     target_link_libraries(sub_string_finder ${TBB_IMPORTED_TARGETS})
#. Create a build directory for the sub_string_finder sample to perform build out of source, go to the created directory
    ``mkdir ~/demo_tbb_cmake/tbb/examples/GettingStarted/sub_string_finder/build ; cd ~/demo_tbb_cmake/tbb/examples/GettingStarted/sub_string_finder/build``
#. Run CMake to prepare Makefile for the sub_string_finder sample and provide TBB location (root) where to perform build:
    ``cmake -DTBB_ROOT=${HOME}/demo_tbb_cmake/tbb ..``
#. Make an executable and run it:
    ``make ; ./sub_string_finder``

Technical documentation for CMake modules
-----------------------------------------
TBBConfig
^^^^^^^^^

Configuration module for TBB library.

How to use this module in your CMake project:
 #. Add location of TBB (root) to `CMAKE_PREFIX_PATH <https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html>`_
    or specify location of TBBConfig.cmake in ``TBB_DIR``.
 #. Use `find_package <https://cmake.org/cmake/help/latest/command/find_package.html>`_ to configure TBB.
 #. Use provided variables and/or imported targets (described below) to work with TBB.

TBB components can be passed to `find_package <https://cmake.org/cmake/help/latest/command/find_package.html>`_
after keyword ``COMPONENTS`` or ``REQUIRED``.
Use basic names of components (``tbb``, ``tbbmalloc``, ``tbb_preview``, etc.).

If components are not specified then default are used: ``tbb``, ``tbbmalloc`` and ``tbbmalloc_proxy``.

If ``tbbmalloc_proxy`` is requested, ``tbbmalloc`` component will also be added and set as dependency for ``tbbmalloc_proxy``.

TBBConfig creates `imported targets <https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#imported-targets>`_ as
shared libraries using the following format: ``TBB::<component>`` (for example, ``TBB::tbb``, ``TBB::tbbmalloc``).

Set ``TBB_FIND_RELEASE_ONLY`` to ``TRUE`` before ``find_package`` call in order to search only for release TBB version. This variable helps to avoid simultaneous linkage of release and debug TBB versions when CMake configuration is `Debug` but a third-party component depends on release TBB version.
Variables set during TBB configuration:

=========================  ================================================
         Variable                            Description
=========================  ================================================
``TBB_FOUND``              TBB library is found
``TBB_<component>_FOUND``  specific TBB component is found
``TBB_IMPORTED_TARGETS``   all created TBB imported targets
``TBB_VERSION``            TBB version (format: ``<major>.<minor>``)
``TBB_INTERFACE_VERSION``  TBB interface version (can be empty, see below for details)
=========================  ================================================

TBBInstallConfig
^^^^^^^^^^^^^^^^

Module for generation and installation of TBB CMake configuration files (TBBConfig.cmake and TBBConfigVersion.cmake files) on Linux, macOS and Windows.

Provides the following functions:

 .. code:: cmake

  tbb_install_config(INSTALL_DIR <install_dir> SYSTEM_NAME Linux|Darwin|Windows
                     [TBB_VERSION <major>.<minor>|TBB_VERSION_FILE <version_file>]
                     [LIB_REL_PATH <lib_rel_path> INC_REL_PATH <inc_rel_path>]
                     [LIB_PATH <lib_path> INC_PATH <inc_path>])``

**Note: the module overwrites existing TBBConfig.cmake and TBBConfigVersion.cmake files in <install_dir>.**

``tbb_config_installer.cmake`` allows to run ``TBBInstallConfig.cmake`` from command line.
It accepts the same parameters as ``tbb_install_config`` function, run ``cmake -P tbb_config_installer.cmake`` to get help.

Use cases
"""""""""
**Prepare TBB CMake configuration files for custom TBB package.**

The use case is applicable for package maintainers who create own TBB packages and want to create TBBConfig.cmake and TBBConfigVersion.cmake for these packages.

===========================================  ===========================================================
              Parameter                                      Description
===========================================  ===========================================================
``INSTALL_DIR <directory>``                  Directory to install CMake configuration files
``SYSTEM_NAME Linux|Darwin|Windows``         OS name to generate config files for
``TBB_VERSION_FILE <version_file>``          Path to ``tbb_stddef.h`` to parse version from and
                                             write it to TBBConfigVersion.cmake
``TBB_VERSION <major>.<minor>``              Directly specified TBB version; alternative to ``TBB_VERSION_FILE`` parameter;
                                             ``TBB_INTERFACE_VERSION`` is set to empty value in this case
``LIB_REL_PATH <lib_rel_path>``              Relative path to TBB binaries (.lib files on Windows), default: ``../../../lib``
``BIN_REL_PATH <bin_rel_path>``              Relative path to TBB DLLs, default: ``../../../bin`` (applicable for Windows only)
``INC_REL_PATH <inc_rel_path>``              Relative path to TBB headers, default: ``../../../include``
===========================================  ===========================================================

*Example*

 Assume your package is installed to the following structure:

 * Binaries go to ``<prefix>/lib``
 * Headers go to ``<prefix>/include``
 * CMake configuration files go to ``<prefix>/lib/cmake/<package>``

 The package is packed from ``/my/package/content`` directory.

 ``cmake -DINSTALL_DIR=/my/package/content/lib/cmake/TBB -DSYSTEM_NAME=Linux -DTBB_VERSION_FILE=/my/package/content/include/tbb/tbb_stddef.h -P tbb_config_installer.cmake`` (default relative paths will be used)

**Install TBB CMake configuration files for installed TBB.**

The use case is applicable for users who have installed TBB, but do not have (or have incorrect) CMake configuration files for this TBB.

====================================  ==============================================
      Parameter                            Description
====================================  ==============================================
``INSTALL_DIR <directory>``           Directory to install CMake configuration files
``SYSTEM_NAME Linux|Darwin|Windows``  OS name to generate config files for
``LIB_PATH <lib_path>``               Path to installed TBB binaries (.lib files on Windows)
``BIN_PATH <bin_path>``               Path to installed TBB DLLs (applicable for Windows only)
``INC_PATH <inc_path>``               Path to installed TBB headers
====================================  ==============================================

``LIB_PATH`` and ``INC_PATH`` will be converted to relative paths based on ``INSTALL_DIR``.
By default TBB version will be parsed from ``<inc_path>/tbb/tbb_stddef.h``,
but it can be overridden by optional parameters ``TBB_VERSION_FILE`` or ``TBB_VERSION``.

*Example*

 TBB is installed to ``/usr`` directory.
 In order to create TBBConfig.cmake and TBBConfigVersion.cmake in ``/usr/lib/cmake/TBB`` run

 ``cmake -DINSTALL_DIR=/usr/lib/cmake/TBB -DSYSTEM_NAME=Linux -DLIB_PATH=/usr/lib -DINC_PATH=/usr/include -P tbb_config_installer.cmake``.

TBBGet
^^^^^^

Module for getting TBB library from `GitHub <https://github.com/01org/tbb>`_.

Provides the following functions:
 ``tbb_get(TBB_ROOT <variable> [RELEASE_TAG <release_tag>|LATEST] [SAVE_TO <path>] [SYSTEM_NAME Linux|Windows|Darwin] [CONFIG_DIR <variable> | SOURCE_CODE])``
  downloads TBB from GitHub and creates TBBConfig for the downloaded binary package if there is no TBBConfig.

  ====================================  ====================================
                     Parameter                       Description
  ====================================  ====================================
  ``TBB_ROOT <variable>``               a variable to save TBB root in, ``<variable>-NOTFOUND`` will be provided in case ``tbb_get`` is unsuccessful
  ``RELEASE_TAG <release_tag>|LATEST``  TBB release tag to be downloaded (for example, ``2017_U6``), ``LATEST`` is used by default
  ``SAVE_TO <path>``                    path to location at which to unpack downloaded TBB, ``${CMAKE_CURRENT_BINARY_DIR}/tbb_downloaded`` is used by default
  ``SYSTEM_NAME Linux|Windows|Darwin``  operating system name to download a binary package for,
                                        value of `CMAKE_SYSTEM_NAME <https://cmake.org/cmake/help/latest/variable/CMAKE_SYSTEM_NAME.html>`_ is used by default
  ``CONFIG_DIR <variable>``             a variable to save location of TBBConfig.cmake and TBBConfigVersion.cmake. Ignored if ``SOURCE_CODE`` specified
  ``SOURCE_CODE``                       flag to get TBB source code (instead of binary package)
  ====================================  ====================================

TBBMakeConfig
^^^^^^^^^^^^^

Module for making TBBConfig in `official TBB binary packages published on GitHub <https://github.com/01org/tbb/releases>`_.

This module is to be used for packages that do not have TBBConfig.

Provides the following functions:
 ``tbb_make_config(TBB_ROOT <path> CONFIG_DIR <variable> [SYSTEM_NAME Linux|Windows|Darwin])``
  creates CMake configuration files (TBBConfig.cmake and TBBConfigVersion.cmake) for TBB binary package.

  ====================================  ====================================
                     Parameter                       Description
  ====================================  ====================================
  ``TBB_ROOT <variable>``               path to TBB root
  ``CONFIG_DIR <variable>``             a variable to store location of the created configuration files
  ``SYSTEM_NAME Linux|Windows|Darwin``  operating system name of the binary TBB package,
                                        value of `CMAKE_SYSTEM_NAME <https://cmake.org/cmake/help/latest/variable/CMAKE_SYSTEM_NAME.html>`_ is used by default
  ====================================  ====================================

TBBBuild
^^^^^^^^

Module for building TBB library from the source code.

Provides the following functions:
 ``tbb_build(TBB_ROOT <tbb_root> CONFIG_DIR <variable> [MAKE_ARGS <custom_make_arguments>])``
  builds TBB from source code using the ``Makefile``, creates and provides the location of the CMake configuration files (TBBConfig.cmake and TBBConfigVersion.cmake) .

  =====================================  ====================================
                Parameter                             Description
  =====================================  ====================================
  ``TBB_ROOT <variable>``                path to TBB root
  ``CONFIG_DIR <variable>``              a variable to store location of the created configuration files,
                                         ``<variable>-NOTFOUND`` will be provided in case ``tbb_build`` is unsuccessful
  ``MAKE_ARGS <custom_make_arguments>``  custom arguments to be passed to ``make`` tool.

                                         The following arguments are always passed with automatically detected values to
                                         ``make`` tool if they are not redefined in ``<custom_make_arguments>``:

                                           - ``compiler=<compiler>``
                                           - ``tbb_build_dir=<tbb_build_dir>``
                                           - ``tbb_build_prefix=<tbb_build_prefix>``
                                           - ``-j<n>``
  =====================================  ====================================


------------

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

``*`` Other names and brands may be claimed as the property of others.
