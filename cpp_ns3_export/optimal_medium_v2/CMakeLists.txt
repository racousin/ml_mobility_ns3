# Find LibTorch package (using CMAKE_PREFIX_PATH set by user)
find_package(Torch QUIET)

# Prepare source and header lists
set(mobility_sources_list
    helper/group-mobility-helper.cc
    helper/mobility-helper.cc
    helper/ns2-mobility-helper.cc
    model/box.cc
    model/constant-acceleration-mobility-model.cc
    model/constant-position-mobility-model.cc
    model/constant-velocity-helper.cc
    model/constant-velocity-mobility-model.cc
    model/gauss-markov-mobility-model.cc
    model/geocentric-constant-position-mobility-model.cc
    model/geographic-positions.cc
    model/hierarchical-mobility-model.cc
    model/mobility-model.cc
    model/position-allocator.cc
    model/random-direction-2d-mobility-model.cc
    model/random-walk-2d-mobility-model.cc
    model/random-waypoint-mobility-model.cc
    model/rectangle.cc
    model/steady-state-random-waypoint-mobility-model.cc
    model/waypoint-mobility-model.cc
    model/waypoint.cc
)

set(mobility_headers_list
    helper/group-mobility-helper.h
    helper/mobility-helper.h
    helper/ns2-mobility-helper.h
    model/box.h
    model/constant-acceleration-mobility-model.h
    model/constant-position-mobility-model.h
    model/constant-velocity-helper.h
    model/constant-velocity-mobility-model.h
    model/gauss-markov-mobility-model.h
    model/geocentric-constant-position-mobility-model.h
    model/geographic-positions.h
    model/hierarchical-mobility-model.h
    model/mobility-model.h
    model/position-allocator.h
    model/random-direction-2d-mobility-model.h
    model/random-walk-2d-mobility-model.h
    model/random-waypoint-mobility-model.h
    model/rectangle.h
    model/steady-state-random-waypoint-mobility-model.h
    model/waypoint-mobility-model.h
    model/waypoint.h
)

set(mobility_libraries_to_link ${libantenna} ${libnetwork})

# Add ML model if LibTorch is found
if(Torch_FOUND)
    message(STATUS "LibTorch found - adding netmob25 mobility model")
    list(APPEND mobility_sources_list model/netmob25-mobility-model.cc)
    list(APPEND mobility_headers_list model/netmob25-mobility-model.h)
    list(APPEND mobility_libraries_to_link ${TORCH_LIBRARIES})
    
    # Copy model.pt file to build directory
    if(EXISTS ${CMAKE_SOURCE_DIR}/model.pt)
        configure_file(${CMAKE_SOURCE_DIR}/model.pt ${CMAKE_BINARY_DIR}/model.pt COPYONLY)
        message(STATUS "Model file copied to: ${CMAKE_BINARY_DIR}/model.pt")
    endif()
endif()

build_lib(
  LIBNAME mobility
  SOURCE_FILES ${mobility_sources_list}
  HEADER_FILES ${mobility_headers_list}
  LIBRARIES_TO_LINK ${mobility_libraries_to_link}
  TEST_SOURCES
    test/box-line-intersection-test.cc
    test/geo-to-cartesian-test.cc
    test/geocentric-topocentric-conversion-test.cc
    test/mobility-test-suite.cc
    test/mobility-trace-test-suite.cc
    test/ns2-mobility-helper-test-suite.cc
    test/rand-cart-around-geo-test.cc
    test/rectangle-closest-border-test.cc
    test/steady-state-random-waypoint-mobility-model-test.cc
    test/waypoint-mobility-model-test.cc
)

# Additional LibTorch configuration if found
if(Torch_FOUND)
    # Use SYSTEM to suppress warnings from LibTorch headers
    # Use PRIVATE instead of PUBLIC to avoid CMake policy issues with source-tree paths
    target_include_directories(${libmobility} SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    target_compile_definitions(${libmobility} PUBLIC HAVE_LIBTORCH)
    target_compile_features(${libmobility} PUBLIC cxx_std_17)
    
    # Disable OpenMP to avoid conflicts with NS-3's precompiled headers
    # LibTorch may have been compiled with OpenMP, but we'll run single-threaded
    target_compile_definitions(${libmobility} PUBLIC AT_PARALLEL_OPENMP=0)
    
    # On macOS, filter out Linux-specific linker flags
    if(APPLE)
        set(FILTERED_FLAGS)
        foreach(flag ${TORCH_CXX_FLAGS})
            if(NOT flag MATCHES "-Wl,--")
                list(APPEND FILTERED_FLAGS ${flag})
            endif()
        endforeach()
        set(TORCH_CXX_FLAGS ${FILTERED_FLAGS})
    endif()
    
    # Apply LibTorch compile flags
    target_compile_options(${libmobility} PUBLIC ${TORCH_CXX_FLAGS})
endif()