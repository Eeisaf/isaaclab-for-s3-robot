cmake_minimum_required(VERSION 3.8)
project(eyou_ethercat_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  urdf
  rviz
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/sim2real_policy_node.py
  scripts/test_target_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()