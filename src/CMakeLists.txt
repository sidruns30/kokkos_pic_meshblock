set(EXECUTABLE main.xc)

file(GLOB_RECURSE SOURCES "*.cpp")

add_executable(${EXECUTABLE} ${SOURCES})

target_include_directories(${EXECUTABLE}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${EXECUTABLE} Kokkos::kokkos)
