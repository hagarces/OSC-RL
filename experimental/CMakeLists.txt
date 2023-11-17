#cmake_minimum_required(VERSION 3.0.2) # 
cmake_minimum_required(VERSION 3.4)
project(franka_control_test)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  franka_gripper
  franka_hw
  pluginlib
  controller_interface
  message_generation
  controller_manager
  cv_bridge
  image_transport
)

# necesario?
find_package(Franka 0.9.0 QUIET)
if(NOT Franka_FOUND)
  find_package(Franka 0.8.0 REQUIRED)
endif()

# indicar que se importe el package de open cv
find_package(OpenCV 4 REQUIRED)

add_message_files(FILES
)


add_service_files(FILES
  SetPointXYZ.srv
  SetPointRL.srv
)



generate_messages(DEPENDENCIES
  std_msgs
)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES franka_control_test
  CATKIN_DEPENDS roscpp
                 franka_gripper
                 franka_hw
                 pluginlib
                 controller_interface
                 cv_bridge
                 image_transport
  DEPENDS Franka
)




add_library(franka_control_test
  src/robot_xyz_controller.cpp
  src/robot_OSC_controller.cpp
  src/utils.cpp
)

target_include_directories(franka_control_test PUBLIC
  include
)


target_link_libraries(franka_control_test PUBLIC
  ${Franka_LIBRARIES}
  ${catkin_LIBRARIES}
)


set(EXAMPLES
  franka_control_node2
)



foreach(example ${EXAMPLES})
  add_executable(${example} src/${example}.cpp)
  
  add_dependencies(${example} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  
  target_include_directories(${example} PUBLIC ${catkin_INCLUDE_DIRS})
  
  target_link_libraries(${example} PUBLIC ${catkin_LIBRARIES}) 
endforeach()

## Installation
install(TARGETS franka_control_test
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES franka_control_test_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

catkin_install_python(
  PROGRAMS scripts/RL_MOTION_PLANNER.py 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


# ------------------------------------------------------

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


## Tools
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/ClangTools.cmake OPTIONAL
  RESULT_VARIABLE CLANG_TOOLS
)
if(CLANG_TOOLS)
  file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  )
  add_format_target(franka_control_test FILES ${SOURCES} ${HEADERS})
  add_tidy_target(franka_control_test
    FILES ${SOURCES}
    DEPENDS franka_control_test square_rutine
  )
endif()

if(PEP_TOOLS)
  file(GLOB_RECURSE PYSOURCES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py)
  add_pyformat_target(franka_control_test FILES ${PYSOURCES})
endif()

