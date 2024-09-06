# Create a static library from utils.cpp
add_library(grid STATIC grid.cpp)
target_include_directories(grid PUBLIC ../include)
target_link_libraries(grid PUBLIC util)
target_link_libraries(grid PUBLIC particle_tracker)