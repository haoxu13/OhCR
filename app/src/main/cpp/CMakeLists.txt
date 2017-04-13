
#
# Copyright (C) The Android Open Source Project
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

cmake_minimum_required(VERSION 3.4.1)

# configure import libs
set(distribution_DIR ${CMAKE_SOURCE_DIR}/../../../../distribution)

add_library(lib_ccv_swt STATIC IMPORTED)
set_target_properties(lib_ccv_swt PROPERTIES IMPORTED_LOCATION
    ${distribution_DIR}/ccv_swt/lib/${ANDROID_ABI}/libccv_swt.a)


# build application's shared lib
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_library(ccv-libs SHARED
            ccv-libs.cpp)

target_include_directories(ccv-libs PRIVATE
                           ${distribution_DIR}/ccv_swt/include)

target_link_libraries(ccv-libs
                      android
                      lib_ccv_swt)