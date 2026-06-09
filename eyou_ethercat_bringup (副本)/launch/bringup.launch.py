from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory
import os
import xacro

def generate_launch_description():
    pkg = get_package_share_directory('eyou_ethercat_bringup')
    urdf_file = os.path.join(pkg, 'urdf', 'trunk_robot.ethercat.xacro')
    controllers_file = os.path.join(pkg, 'config', 'controllers.yaml')

    robot_description_content = xacro.process_file(urdf_file).toxml()

    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node', 
        parameters=[
            {'robot_description': robot_description_content},
            controllers_file,
        ],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content}],
        output='screen'
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager'
        ],
        output='screen'
    )

    trunk_group_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'trunk_group_controller',
            '--controller-manager', '/controller_manager'
        ],
        output='screen'
    )

    return LaunchDescription([
        control_node,
        robot_state_publisher,
        RegisterEventHandler(
            OnProcessStart(
                target_action=control_node,
                on_start=[joint_state_broadcaster_spawner],
            )
        ),
        RegisterEventHandler(
            OnProcessStart(
                target_action=joint_state_broadcaster_spawner,
                on_start=[trunk_group_controller_spawner],
            )
        ),
    ])