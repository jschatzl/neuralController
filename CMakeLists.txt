cmake_minimum_required(VERSION 3.22)

project(Neural_Network)

# generated library goes into the scripts folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/scripts")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

# create STATIC library for every other target
add_library(neuralController STATIC)
target_sources(neuralController PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralController.c
)
target_include_directories(neuralController PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "([xX]86|AMD64|amd64)")
    # create target library for python
    add_library(neuralControllerInterface SHARED)
    target_sources(neuralControllerInterface PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralController.c
    )
    target_include_directories(neuralControllerInterface PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
    )
    target_link_libraries(neuralControllerInterface PUBLIC 
        m
    )
    set_target_properties(neuralControllerInterface PROPERTIES PREFIX "")
endif()

if(DEFINED TEST)
    # add target test
    add_executable(test)
    target_sources(test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
    )
    target_include_directories(test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
    )
    target_link_libraries(test PRIVATE 
        neuralController 
        m
    )
endif()


