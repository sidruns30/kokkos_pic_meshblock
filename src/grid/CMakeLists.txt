add_library(grid grid.cpp)

target_link_libraries(grid PUBLIC util)

# HH: you don't need to include one-by-one for each library. (see root CMakeLists.txt)

# target_include_directories(grid PUBLIC ../include)
