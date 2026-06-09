#!/usr/bin/python3
# 指定这是一个Python3可执行脚本

import rclpy # 导入ROS2的Python客户端库
from rclpy.node import Node # 导入ROS2的节点基类
from geometry_msgs.msg import PoseStamped # 导入带时间戳的位姿消息类型
import math

class TestTargetPublisher(Node):
    """
    测试目标发布节点：用于向策略节点发送安全的测试坐标点。
    """
    def __init__(self):
        # 初始化ROS2节点，节点名称为'test_target_publisher'
        super().__init__('test_target_publisher')
        
        # 创建发布器，向 '/target_pose' 话题发布 PoseStamped 消息，队列长度为10
        self.publisher_ = self.create_publisher(PoseStamped, '/target_pose', 10)
        
        # 创建定时器，每 2.0 秒触发一次 timer_callback 回调函数
        self.timer = self.create_timer(2.0, self.timer_callback) 
        
        # Test target coordinates (Safe target in front of the robot)
        # 设定一个安全的测试目标坐标 (位于机器人正前方，高度0.6米)
        self.target_x = 0.15
        self.target_y = 0.0
        self.target_z = 0.60
        
        # 打印初始化完成日志
        self.get_logger().info("Test Target Publisher Initialized. Sending safe targets...")

    def timer_callback(self):
        """
        定时器回调函数：负责打包并发布目标位姿消息。
        """
        msg = PoseStamped() # 创建位姿消息对象
        msg.header.stamp = self.get_clock().now().to_msg() # 打上当前时间戳
        msg.header.frame_id = "base_link" # 指定该坐标是相对于机器人的基座坐标系(base_link)
        
        # Safe position (填入安全的位置坐标)
        msg.pose.position.x = self.target_x
        msg.pose.position.y = self.target_y
        msg.pose.position.z = self.target_z
        
        # Default orientation (identity quaternion)
        # 填入默认的姿态四元数 (无旋转，即单位四元数)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        
        # 发布消息
        self.publisher_.publish(msg)
        # 打印已发布的日志信息
        self.get_logger().info(f"Published Target: x={self.target_x}, y={self.target_y}, z={self.target_z}")

def main(args=None):
    """
    节点入口函数。
    """
    rclpy.init(args=args) # 初始化ROS2环境
    node = TestTargetPublisher() # 实例化测试发布器节点
    try:
        rclpy.spin(node) # 阻塞并保持节点运行
    except KeyboardInterrupt:
        pass # 捕捉Ctrl+C中断信号
    finally:
        node.destroy_node() # 销毁节点
        rclpy.shutdown() # 关闭ROS2环境

if __name__ == '__main__':
    main() # 如果作为主程序运行，则调用入口函数