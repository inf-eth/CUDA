set(EXECUTABLE_NAME ${PROJECT_NAME})
add_executable(${EXECUTABLE_NAME} ../src/main.cpp ../src/kernels.cu ../include/kernels.h)
target_include_directories(${EXECUTABLE_NAME} PUBLIC ../include)
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