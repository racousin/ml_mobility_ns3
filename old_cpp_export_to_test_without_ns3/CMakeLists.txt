cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(netmob25_trajectory_generator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(.)

# Add executable
add_executable(run_trajectory_gen
    main.cc
    trajectory_generator.cc
)

# Link libraries
target_link_libraries(run_trajectory_gen "${TORCH_LIBRARIES}")

# Set C++ standard for target
set_property(TARGET run_trajectory_gen PROPERTY CXX_STANDARD 17)

# Enable debugging symbols
set(CMAKE_BUILD_TYPE Debug)

# Print some useful information
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Project: vae_lstm_2025-07-16_12-30-52")