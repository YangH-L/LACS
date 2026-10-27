cmake_minimum_required(VERSION 3.12)
project(lacs CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

find_package(Threads REQUIRED)

add_executable(lacs_search src/lacs_search.cpp)
target_link_libraries(lacs_search Threads::Threads)
add_executable(lacs_convert src/lacs_convert.cpp)
add_executable(lacs_project src/lacs_project.cpp)
add_executable(lacs_lpa src/lacs_lpa.cpp)
