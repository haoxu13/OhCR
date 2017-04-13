#
# Copyright (C) 2016 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Build a static lib and copy it into distribution place for independent app to use
# we could not delete the bin directory is because android studio would error out
# when it checks target's existance. Only distribution place is used as import to app

cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_VERBOSE_MAKEFILE on)

add_library(ccv_swt STATIC src/ccv_swt.c src/ccv_io.c)

# copy out the lib binary... need to leave the static lib around to pass gradle check
set(distribution_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../distribution)
set_target_properties(ccv_swt
                      PROPERTIES
                      ARCHIVE_OUTPUT_DIRECTORY
                      "${distribution_DIR}/ccv_swt/lib/${ANDROID_ABI}")

# copy out lib header file...
add_custom_command(TARGET ccv_swt POST_BUILD
                   COMMAND "${CMAKE_COMMAND}" -E
                   copy "${CMAKE_CURRENT_SOURCE_DIR}/src/ccv.h"
                   "${distribution_DIR}/ccv_swt/include/ccv.h"
                   COMMAND "${CMAKE_COMMAND}" -E
                   copy "${CMAKE_CURRENT_SOURCE_DIR}/src/ccv_internal.h"
                   "${distribution_DIR}/ccv_swt/include/ccv_internal.h"
#                   **** the following 2 lines are for potential future debug purpose ****
#                   COMMAND "${CMAKE_COMMAND}" -E
#                   remove_directory "${CMAKE_CURRENT_BINARY_DIR}"
                   COMMENT "Copying ccv_swt to output directory")


