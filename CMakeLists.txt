cmake_minimum_required(VERSION 3.14)
project(cmaes VERSION 0.1.0 LANGUAGES CXX)

if(NOT TARGET Eigen3::Eigen)
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# Header-only: link cmaes::cmaes to get the include path, Eigen and C++17.
add_library(cmaes INTERFACE)
add_library(cmaes::cmaes ALIAS cmaes)
target_include_directories(cmaes INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(cmaes INTERFACE Eigen3::Eigen)
target_compile_features(cmaes INTERFACE cxx_std_17)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  find_package(Threads REQUIRED)
  enable_testing()
  add_executable(test_cmaes tests/test_cmaes.cpp)
  target_link_libraries(test_cmaes PRIVATE cmaes::cmaes)
  add_test(NAME test_cmaes COMMAND test_cmaes)

  foreach(ex rosenbrock ask_tell)
    add_executable(${ex} examples/${ex}.cpp)
    target_link_libraries(${ex} PRIVATE cmaes::cmaes Threads::Threads)
  endforeach()
endif()
