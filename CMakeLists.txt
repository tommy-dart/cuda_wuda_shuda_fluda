cmake_minimum_required (VERSION 3.0.2)
project(cuda_wuda_shuda_fluda LANGUAGES CXX)

#### Check Dependencies ####
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)



#### Directory definition ####
#set (PROJECT_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
set (PROJECT_SRC_DIR      "${PROJECT_SOURCE_DIR}/src")
#set (PROJECT_DEPS_DIR     "${PROJECT_SOURCE_DIR}/deps")


#### Adding dependencies ####

# Adding external libs
#add_subdirectory()


#### Source Files ####

#file(GLOB_RECURSE SOURCES "*.cpp" "*.c")
file(GLOB_RECURSE SOURCES_CUDA "${PROJECT_SRC_DIR}/*.cu")


#### Header Files ####

include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)


#### Compilation ####

message(${CUDA_LIBRARIES})
message(${OPENGL_LIBRARIES})
message(${GLUT_LIBRARIES})
message(${GLEW_LIBRARIES})

# Compilation Flags

set(COMPILE_FLAGS "${COMPILE_FLAGS} -W -Wall -std=c++11 -Wshadow -O2 -g ")
set(LINKER_FLAGS  "${LINKER_FLAGS} -lm")
# set(LINKER_FLAGS  "${LINKER_FLAGS} -lm -lglut -lGLU -lGL -lGLEW")

# Cmake compilation flags redefinition

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${LINKER_FLAGS}")

# findCUDA Options

set(CUDA_VERBOSE_BUILD ON)
set(CUDA_SEPARABLE_COMPILATION ON) # Allow multiple CUDA files compilation


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

cuda_add_executable(${PROJECT_NAME}
    ${SOURCES_CUDA}
)

target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBARIES}
    ${GLUT_LIBRARIES}
    ${GLEW_LIBRARIES}
    /usr/lib/x86_64-linux-gnu/libGLU.so
    /usr/lib/x86_64-linux-gnu/libGLEW.so
    /usr/lib/x86_64-linux-gnu/libGL.so
    /usr/lib/x86_64-linux-gnu/libcufft.so
)
