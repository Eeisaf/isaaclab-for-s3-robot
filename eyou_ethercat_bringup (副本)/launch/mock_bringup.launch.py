from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart, OnProcessExit
from ament_index_python.packages import get_package_share_directory
import os
import xacro

def generate_launch_description():
    pkg = get_package_share_directory('eyou_ethercat_bringup')
    
    # 关键修改：加载 MOCK 版本的 URDF，而不是 ethercat 版本
    urdf_file = os.path.join(pkg, 'urdf', 'trunk_robot.mock.xacro')
    controllers_file = os.path.join(pkg, 'config', 'controllers.yaml')
    rviz_config_file = os.path.join(pkg, 'rviz', 'mock.rviz')

    robot_description_content = xacro.process_file(urdf_file).toxml()

    # 启动控制器管理器 (Controller Manager)
    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node', 
        parameters=[
            {'robot_description': robot_description_content},
            controllers_file,
        ],
        output='screen'
    )

    # 启动机器人状态发布器 (Robot State Publisher)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content}],
        output='screen'
    )

    # 启动 RViz2 进行可视化
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    # 启动关节状态广播器
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager'
        ],
        output='screen'
    )

    # 启动轨迹控制器
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
        rviz_node,
        RegisterEventHandler(
            OnProcessStart(
                target_action=control_node,
                on_start=[joint_state_broadcaster_spawner],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[trunk_group_controller_spawner],
            )
        ),
    ])
