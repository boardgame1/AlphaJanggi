cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(AlphaJanggi)

set(ON 1)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

add_executable(AlphaJanggi pch.cpp MCTS.cpp Model.cpp Project1.cpp)
target_link_libraries(AlphaJanggi "${TORCH_LIBRARIES}")
set_property(TARGET AlphaJanggi PROPERTY CXX_STANDARD 14)
