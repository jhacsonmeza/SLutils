# Samples

add_executable(ps ps.cpp)
target_link_libraries(ps ${OpenCV_LIBS} SLutils)

add_executable(ps_gc ps+gc.cpp)
target_link_libraries(ps_gc ${OpenCV_LIBS} SLutils)

add_executable(cl cl.cpp)
target_link_libraries(cl ${OpenCV_LIBS} SLutils)
