add_library(particle_tracker particle_tracker.cpp)

target_link_libraries(particle_tracker PUBLIC grid util)

# HH: you don't need to include one-by-one for each library. (see root CMakeLists.txt)

# target_include_directories(particle_tracker PUBLIC ../include)
