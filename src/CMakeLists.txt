# HH: see my comment in root CMakelists.txt
# 
add_subdirectory(grid)
add_subdirectory(particle_tracker)
add_subdirectory(utils)

# HH: no need to include the header files here.
#
# add_executable(main.xc main.cpp include/util.hpp)

add_executable(main.xc main.cpp)

# HH: you don't need to include one-by-one for each library. (see root CMakeLists.txt)
#
# target_include_directories(main.xc PUBLIC ../include)

target_link_libraries(main.xc util grid particle_tracker)
