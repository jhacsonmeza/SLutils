file(GLOB SOURCES *.cpp)
add_library(sl STATIC ${SOURCES})

target_include_directories(sl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(sl ${OpenCV_LIBS} slcalib)
