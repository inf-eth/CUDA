#cmake_minimum_required(VERSION 3.20...3.30)
#project(simplePrintf LANGUAGES C CXX CUDA)
#find_package(CUDAToolkit REQUIRED)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(EXECUTABLE_NAME ${PROJECT_NAME})
add_executable(${EXECUTABLE_NAME} ../src/simplePrintf.cu)
target_compile_options(${EXECUTABLE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(${EXECUTABLE_NAME} PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${EXECUTABLE_NAME} PUBLIC ../include ../include/CUDA ../include ../include/common)
#target_link_libraries(${PROJECT_NAME} kernels)
#add_library(kernels ${PROJECT_SOURCE_DIR}/src/kernels.cu ${PROJECT_SOURCE_DIR}/include/kernels.cuh)
#target_include_directories(kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
#target_compile_features(kernels PUBLIC cxx_std_11)
#set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(kernels)

# Auto-select bitness based on platform
if( NOT BITNESS )
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(BITNESS 64)
    else()
        set(BITNESS 32)
    endif()
endif()

if( BITNESS EQUAL 64 )
    set(BITNESS_SUFFIX x86_64)
elseif( BITNESS EQUAL 32 )
    set(BITNESS_SUFFIX x86)
else()
    message( FATAL_ERROR "Bitness specified is invalid" )
endif()

# Set CMAKE_BUILD_TYPE (default = Release)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	set(CMAKE_BUILD_TYPE Debug)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "")
	set(CMAKE_BUILD_TYPE Release)
else()
	set(CMAKE_BUILD_TYPE Release)
endif()
# Set output directory to bin
if( MSVC )
	set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/${BITNESS_SUFFIX})
else()
	set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/${BITNESS_SUFFIX}/${CMAKE_BUILD_TYPE})
endif()

if( NOT MSVC )
add_custom_target(
    run
    COMMAND ${EXECUTABLE_OUTPUT_PATH}/${EXECUTABLE_NAME}
    DEPENDS ${EXECUTABLE_NAME}
    COMMENT "Running..."
)
endif()

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")