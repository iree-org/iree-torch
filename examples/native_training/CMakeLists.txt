# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# IREE requires a minimum CMake version of 3.21 but may be possible to get
# working with older versions.
cmake_minimum_required(VERSION 3.21...3.24)

#-------------------------------------------------------------------------------
# Project configuration
#-------------------------------------------------------------------------------

project(iree-template-runtime-cmake VERSION 1.0 LANGUAGES CXX C)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#-------------------------------------------------------------------------------
# IREE subproject configuration
#-------------------------------------------------------------------------------
# IREE is a toolkit and has many optional features and build tuning parameters
# that will differ for everyone integrating it. This extends from big knobs
# like which HAL backends to support (CUDA, Vulkan, CPU single-threaded,
# CPU multi-threaded, etc) and to small knobs like whether to preserve
# iree_status_t failure messages (and the associated string manipulation code)
# or treat them like integers for code size reduction.

# Disable building the IREE compiler and test files.
# Compiler builds pull in the LLVM build and make cross-compiling difficult.
# It's recommended that the IREE packages are installed from pip or the compiler
# is built from source from the main IREE repository and installed for use by
# the subprojects involving the runtime. The main IREE repository compiles
# models as part of its build process but for most user workflows compilation
# can be handled much better by scripts and frontend integrations.
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)

# By default the IREE runtime will contain all of the HAL drivers that are able
# to be built based on available dependencies (CUDA SDK, etc).
# Users should only include the drivers they actually plan on using to save on
# compile time and binary size. This sample uses the local synchronous CPU
# executor and the embedded ELF loader.
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON)

# The CMake variant specified will change some runtime behavior such as
# including asserts in Debug builds. There's a group of features commonly used
# in size-sensitive builds (bare-metal embedded, web, etc) that can be easily
# toggled with the IREE_SIZE_OPTIMIZED flag. This disables debug features like
# VM disassembly traces, iree_status_t string messages, and statistics. Each of
# these is available for independent control and decoupled from the CMake
# variant to allow users to decide; for example helpful string error messages
# can be useful in MinSizeRel variants and may be worth the ~4KB binary size
# tradeoff to some users while others may be trying to get the runtime to fit
# in a 128KB ROM and will choose to disable them.
#
# Here as an example we tie IREE_SIZE_OPTIMIZED to the MinSizeRel variant. This
# means that Debug and RelWithDebInfo builds will get nice error messages while
# MinSizeRel will just get status code names.
if(CMAKE_BUILD_TYPE MATCHES MinSizeRel)
  set(IREE_SIZE_OPTIMIZED ON)
endif()

# Include the IREE submodule CMake project as a dependency.
# This will make the IREE targets available for use by subsequent commands.
#
# Note that EXCLUDE_FROM_ALL will prevent the IREE targets from showing in the
# CMake target listing by default - for most users this is fine but if deeply
# integrating it can be useful to drop it.
add_subdirectory(third_party/iree EXCLUDE_FROM_ALL)

#-------------------------------------------------------------------------------
# Application
#-------------------------------------------------------------------------------
# This is the runtime/src/iree/runtime/demo/ source copied here to show a basic
# command line application that can load a compiled IREE module and pass in
# buffers. Usage of the runtime APIs is documented in the main IREE repository.

set(_NAME "native_training")
add_executable(${_NAME} "main.c")

# TODO(benvanik): make iree_status_annotate_f always available as a function
# instead of defining it empty? otherwise optimized builds of the runtime won't
# export it but external libraries may pull it in.
target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

# Link in the IREE runtime library.
# This will pull in dependencies based on the configuration options above.
target_link_libraries(${_NAME} iree_runtime_runtime)
