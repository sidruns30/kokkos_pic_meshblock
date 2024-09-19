add_library(particle_tracker STATIC particle_tracker.cpp)
target_include_directories(particle_tracker PUBLIC ../include)
target_link_libraries(particle_tracker PUBLIC util)