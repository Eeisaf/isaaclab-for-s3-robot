#!/usr/bin/python3
# 指定这是一个Python3可执行脚本

import rclpy # 导入ROS2的Python客户端库
from rclpy.node import Node # 导入ROS2的节点基类
from sensor_msgs.msg import JointState # 导入关节状态消息类型，用于接收真实电机的角度和速度
from geometry_msgs.msg import PoseStamped # 导入带时间戳的位姿消息类型，用于接收目标点坐标
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # 导入关节轨迹消息类型，用于向控制器下发动作
import tf2_ros # 导入ROS2的坐标变换(TF)库，用于计算末端执行器的真实位置
import torch # 导入PyTorch深度学习框架，用于运行强化学习策略网络
import numpy as np # 导入NumPy数学库，用于矩阵和向量计算
import os # 导入操作系统接口库，用于检查文件路径是否存在

# skrl imports (导入skrl强化学习库的相关组件)
from skrl.utils.model_instantiators.torch import gaussian_model, deterministic_model # 导入模型实例化工具，用于快速构建策略网络和价值网络
from skrl.agents.torch.ppo import PPO, PPO_CFG
import gymnasium as gym # 导入Gymnasium环境库，用于定义观测和动作空间

class DummyEnv:
    """
    定义一个伪环境类。因为skrl的PPO代理在初始化时需要知道环境的观测空间和动作空间，
    但在真实部署中我们没有仿真环境，所以用这个伪环境来“骗”过初始化检查。
    """
    def __init__(self):
        # 自动检测并使用GPU(cuda)，如果没有则使用CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 环境数量设为1，因为真实机器人只有一个
        self.num_envs = 1
        # 定义观测空间：25维的连续向量 (与训练时的设定保持一致)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,))
        # 定义动作空间：3维的连续向量，范围在[-1.0, 1.0]之间 (对应前三个关节的输出)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
        # 状态空间设为None即可
        self.state_space = None
        
    # skrl 新版需要环境有 device 属性，但它有时会读取 env.device.type，所以为了安全起见，
    # 我们直接把 device 属性设为字符串或者确保它有正确的行为。
    # 另外，skrl 的 compute_space_size 会尝试调用 flatdim，它需要空间对象正确配置。

class Sim2RealPolicyNode(Node):
    """
    Sim2Real策略部署节点：负责订阅真实机器人状态，运行神经网络，并发布控制指令。
    """
    def __init__(self):
        # 初始化ROS2节点，节点名称为'sim2real_policy_node'
        super().__init__('sim2real_policy_node')
        
        # 声明ROS2参数，允许在launch文件或命令行中修改
        self.declare_parameter('policy_path', '') # 策略权重文件(agent.pt)的路径
        self.declare_parameter('base_frame', 'chassis_base_link') # 机器人的基座坐标系名称
        self.declare_parameter('ee_frame', 'trunk_link4') # 机器人的末端执行器坐标系名称
        self.declare_parameter('control_rate', 60.0) # 控制频率(Hz)，需与训练时的控制频率一致
        
        # 获取上述声明的参数值
        self.policy_path = self.get_parameter('policy_path').value
        self.base_frame = self.get_parameter('base_frame').value
        self.ee_frame = self.get_parameter('ee_frame').value
        self.control_rate = self.get_parameter('control_rate').value
        
        # RL Setup (强化学习环境和代理设置)
        self.env = DummyEnv() # 实例化伪环境
        self.device = self.env.device # 获取计算设备(CPU/GPU)
        self.setup_rl_agent() # 调用自定义函数，初始化PPO代理并加载权重
        
        # TF2 Setup (坐标变换设置)
        self.tf_buffer = tf2_ros.Buffer() # 创建TF缓存，用于存储一段时间内的坐标系关系
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self) # 创建TF监听器，自动在后台接收TF消息并存入缓存
        
        # State variables (状态变量初始化)
        self.current_joint_pos = np.zeros(4) # 存储当前4个关节的位置(角度)
        self.current_joint_vel = np.zeros(4) # 存储当前4个关节的速度
        self.joint_states_received = False # 标记是否已经接收到了真实的关节状态
        self.target_pose = None # 存储目标位姿，格式为 [x, y, z, w, x, y, z]
        self.joint_names = ['trunk_joint1', 'trunk_joint2', 'trunk_joint3', 'trunk_joint4'] # 关节名称列表
        self.default_joint_pos = np.array([0.0, 0.0, 0.0, 0.0]) # 机器人的默认初始姿态(与训练时一致)
        
        # Subscribers (订阅器设置)
        # 订阅 '/joint_states' 话题，获取真实电机的状态，回调函数为 joint_callback
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        # 订阅 '/target_pose' 话题，获取外部下发的目标点，回调函数为 target_callback
        self.target_sub = self.create_subscription(PoseStamped, '/target_pose', self.target_callback, 10)
        
        # Publisher (发布器设置)
        # 创建发布器，向 '/trunk_group_controller/joint_trajectory' 话题发送关节轨迹指令，控制真实电机
        self.traj_pub = self.create_publisher(JointTrajectory, '/trunk_group_controller/joint_trajectory', 10)
        
        # Timer (定时器设置)
        # 创建定时器，按照设定的控制频率(例如60Hz，即每1/60秒)循环执行 control_loop 函数
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
        
        # 打印初始化完成日志
        self.get_logger().info("Sim2Real Policy Node Initialized. Waiting for /target_pose...")
        
    def setup_rl_agent(self):
        """
        初始化强化学习代理，构建网络结构并加载训练好的权重文件。
        """
        # 检查策略权重路径是否有效
        if not self.policy_path or not os.path.exists(self.policy_path):
            self.get_logger().error(f"Invalid policy_path: {self.policy_path}") # 路径无效则报错并退出
            return
            
        models = {} # 创建一个字典，用于存放策略网络和价值网络
        
        # Policy Network (构建策略网络/Actor)
        # 注意：这里如果报错 Missing key(s) in state_dict: "net_container.4.weight", "net_container.4.bias"
        # 说明训练时用的网络结构可能并不是 [64, 64]
        # 根据实际权重文件，网络结构是共享特征提取器，或者是只有两层隐藏层，并且输出层名为 policy_layer
        # 我们需要完全匹配训练时的网络结构
        models["policy"] = gaussian_model(
            observation_space=self.env.observation_space, # 传入观测空间
            action_space=self.env.action_space, # 传入动作空间
            device=self.device, # 指定计算设备
            clip_actions=False, # 不在网络层裁剪动作
            clip_log_std=True, # 限制对数标准差
            min_log_std=-2.0, # 最小对数标准差
            max_log_std=0.5, # 最大对数标准差
            initial_log_std=-0.5, # 初始对数标准差
            network=[
                {"name": "net", "input": "OBSERVATIONS", "layers": [64, 64], "activations": "elu"}
            ],
            output="ACTIONS" # 网络输出为动作
        )
        
        import torch
        # 覆盖 skrl 自动生成的网络层名字，以匹配权重文件
        # 注意：skrl 内部生成的 net_container 是一个 Sequential，我们不能直接 pop，这会破坏内部结构导致 shape 报错
        # 我们用一个新的 Sequential 替换它
        import torch.nn as nn
        models["policy"].net_container = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU()
        ).to(self.device)
        models["policy"].policy_layer = nn.Linear(64, self.env.action_space.shape[0]).to(self.device)
        models["policy"].value_layer = nn.Linear(64, 1).to(self.device)
        
        # 覆盖 compute 方法。训练配置中的网络输入是 OBSERVATIONS，因此这里读取 observations。
        def policy_compute(inputs, role):
            obs_input = inputs["observations"] if isinstance(inputs, dict) and "observations" in inputs else inputs
            features = models["policy"].net_container(obs_input)
            return models["policy"].policy_layer(features), {"log_std": models["policy"].log_std_parameter}
        models["policy"].compute = policy_compute
        
        # Value Network (构建价值网络/Critic)
        # 注意：即使在部署(推理)阶段只用策略网络，skrl在实例化PPO时仍要求提供价值网络
        models["value"] = deterministic_model(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
            clip_actions=False,
            network=[
                {"name": "net", "input": "OBSERVATIONS", "layers": [64, 64], "activations": "elu"}
            ],
            output="ONE" # 输出为一个标量(状态价值)
        )
        
        # 覆盖 skrl 自动生成的网络层名字，以匹配权重文件
        import torch.nn as nn
        models["value"].net_container = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU()
        ).to(self.device)
        models["value"].value_layer = nn.Linear(64, 1).to(self.device)
        models["value"].policy_layer = nn.Linear(64, self.env.action_space.shape[0]).to(self.device)
        # 增加一个假的 log_std_parameter，因为有些旧权重的 value 网络里也会带这个
        models["value"].log_std_parameter = torch.nn.Parameter(torch.zeros(self.env.action_space.shape[0])).to(self.device)
        
        # 覆盖 compute 方法。价值网络同样以 OBSERVATIONS 作为输入。
        def value_compute(inputs, role):
            obs_input = inputs["observations"] if isinstance(inputs, dict) and "observations" in inputs else inputs
            features = models["value"].net_container(obs_input)
            return models["value"].value_layer(features), {}
        models["value"].compute = value_compute
        
        # 获取PPO的默认配置
        cfg = PPO_CFG()
        # 配置观测预处理器：使用RunningStandardScaler对输入观测值进行标准化(减均值除以标准差)
        # 这是非常关键的一步！因为训练时用了状态标准化，部署时必须用同样的标准化器才能得到正确动作
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        cfg.observation_preprocessor = RunningStandardScaler
        cfg.observation_preprocessor_kwargs = {"size": self.env.observation_space, "device": self.device}
        
        # 实例化PPO代理
        self.agent = PPO(
            models=models, # 传入构建好的网络模型
            memory=None, # 推理阶段不需要经验回放池
            cfg=cfg, # 传入配置
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device
        )
        
        # 从指定的路径加载训练好的权重文件 (这也会一并加载状态预处理器的均值和方差)
        # 为了避免 optimizer 的状态字典不匹配，我们只加载模型权重
        import torch
        checkpoint = torch.load(self.policy_path, map_location=self.device)
        if "policy" in checkpoint:
            self.agent.policy.load_state_dict(checkpoint["policy"], strict=False)
        if "value" in checkpoint:
            self.agent.value.load_state_dict(checkpoint["value"], strict=False)
        if "state_preprocessor" in checkpoint and self.agent._observation_preprocessor is not None:
            self.agent._observation_preprocessor.load_state_dict(checkpoint["state_preprocessor"])
        # 将代理设置为评估(推理)模式，关闭探索噪声和梯度计算
        # skrl 新版没有 set_mode，直接通过设置 agent.training = False 或者不管它，
        # 因为我们在 act() 时可以传入 role="eval" 或类似参数，或者模型本身就处于 eval 模式
        # 这里我们手动把底层的 pytorch model 设置为 eval()
        self.agent.policy.eval()
        if hasattr(self.agent, "value") and self.agent.value is not None:
            self.agent.value.eval()
        self.get_logger().info(f"Loaded policy successfully on {self.device}") # 打印加载成功日志

    def joint_callback(self, msg: JointState):
        """
        关节状态回调函数：每当收到真实电机的状态时触发。
        """
        self.joint_states_received = True # 标记已收到状态
        # 遍历我们关心的4个关节名称
        for i, name in enumerate(self.joint_names):
            # 如果收到的消息中包含该关节
            if name in msg.name:
                # 找到该关节在消息列表中的索引
                idx = msg.name.index(name)
                # 更新当前关节位置
                self.current_joint_pos[i] = msg.position[idx]
                # 如果消息中包含速度信息，则更新当前关节速度
                if len(msg.velocity) > idx:
                    self.current_joint_vel[i] = msg.velocity[idx]
                    
    def target_callback(self, msg: PoseStamped):
        """
        目标点回调函数：接收外部下发的目标位姿。
        """
        # SAFE RL: 安全强化学习检查。如果目标点的Z坐标小于0.1米，可能会导致机器人砸向地面或基座
        if msg.pose.position.z < 0.1:
            self.get_logger().warn("Target Z is too low! Ignoring target to prevent ground collision.") # 打印警告
            return # 直接丢弃该危险目标，不予执行
            
        # 将ROS的Pose消息转换为IsaacLab训练时使用的NumPy数组格式: [x, y, z, w, x, y, z]
        # 注意：IsaacLab的四元数顺序通常是 [w, x, y, z]，这里严格按照训练时的格式拼接
        self.target_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.w, # 四元数实部 w
            msg.pose.orientation.x, # 四元数虚部 x
            msg.pose.orientation.y, # 四元数虚部 y
            msg.pose.orientation.z  # 四元数虚部 z
        ])
        # 打印接收到安全目标点的日志
        self.get_logger().info(f"Received new safe target pose: x={msg.pose.position.x:.2f}, z={msg.pose.position.z:.2f}")

    def control_loop(self):
        """
        主控制循环：以固定频率(如60Hz)运行，负责收集状态、推理动作并下发指令。
        """
        # 如果还没有收到过目标点，则直接返回，不执行任何动作
        if self.target_pose is None:
            return
            
        # SAFE RL: 确保在发送指令前，已经接收到了真实的关节状态
        # 如果还没收到任何关节状态，说明可能还没连上电机，此时乱发指令很危险
        if not self.joint_states_received:
            self.get_logger().warn("Waiting for /joint_states...", throttle_duration_sec=2.0) # 每2秒报一次警
            return
            
        try:
            # Get EE pose (获取末端执行器位姿)
            # 通过TF树查询从基座(base_frame)到末端(ee_frame)的最新坐标变换
            trans = self.tf_buffer.lookup_transform(self.base_frame, self.ee_frame, rclpy.time.Time())
            # 提取末端位置 [x, y, z]
            ee_pos = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            # 提取末端姿态四元数 [w, x, y, z] (注意顺序与IsaacLab对齐)
            ee_quat = np.array([
                trans.transform.rotation.w,
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z
            ])
            # 拼接成7维的末端位姿向量
            ee_pose = np.concatenate([ee_pos, ee_quat])
        except Exception as e:
            # 如果TF查询失败(例如TF树断裂)，则打印警告并跳过本控制周期
            self.get_logger().warn(f"TF Error: {e}")
            return
            
        # Compute position error (计算位置误差)
        # 目标位置减去当前末端位置
        pos_error = self.target_pose[:3] - ee_pos
        # 计算误差向量的欧氏距离(L2范数)
        error_norm = np.linalg.norm(pos_error)
        
        # SAFE RL: 到达目标即停止 (5cm 阈值)
        # 如果距离误差小于0.05米(5厘米)，认为已经到达目标
        if error_norm < 0.05:
            self.get_logger().info("Target reached (<5cm). Holding position.", throttle_duration_sec=2.0)
            # 将当前真实的关节位置作为目标下发，让机器人保持在原地不动，避免在目标点附近高频抖动
            self.publish_trajectory(self.current_joint_pos)
            return

        # Construct observation (构建输入给神经网络的观测向量)
        # 1. 计算相对关节位置 (当前位置 - 默认初始位置)
        joint_pos_rel = self.current_joint_pos - self.default_joint_pos
        # 2. 关节速度
        joint_vel_rel = self.current_joint_vel
        
        # 将所有观测项拼接成一个一维数组 (顺序必须与训练时 observations.py 中的顺序严格一致)
        obs_np = np.concatenate([
            joint_pos_rel, # 相对关节位置 (4维)
            joint_vel_rel, # 关节速度 (4维)
            ee_pose,       # 末端位姿 (7维)
            self.target_pose, # 目标位姿 (7维)
            pos_error      # 位置误差 (3维)
        ]) # 总计 25 维
        
        # 将NumPy数组转换为PyTorch张量，并增加一个Batch维度 (shape变为 [1, 25])，然后移至GPU/CPU
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Inference (神经网络推理)
        with torch.no_grad(): # 关闭梯度计算，加速推理
            # 调用策略网络输出动作。当前 skrl 的第一个参数是 observations，第二个参数才是 states。
            actions, _ = self.agent.act(obs_tensor, None, timestep=0, timesteps=0)
            
        # 将输出的动作张量转回CPU并转换为NumPy数组，提取第一个(也是唯一一个)样本
        action_np = actions.cpu().numpy()[0] # shape (3,)
        
        # SAFE RL: Action scale and limits (动作缩放与安全限制)
        scale = 0.05 # 动作缩放系数，与训练时 ActionsCfg 中的 scale 保持一致
        # 将网络输出的归一化动作 [-1, 1] 映射为实际的关节位置增量 (弧度)
        delta_pos = action_np * scale
        
        # SAFE RL: Velocity limit check (速度硬限幅检查)
        dt = 1.0 / self.control_rate # 计算单步时间周期，例如 1/60 秒
        max_vel = 1.0 # 允许的最大关节速度：1.0 rad/s (与 demo_learn_env_cfg.py 中一致)
        # 限制单步的位置增量，确保其除以 dt 后不会超过最大速度，防止电机瞬间暴走
        delta_pos = np.clip(delta_pos, -max_vel * dt, max_vel * dt)
        
        # 计算目标关节位置
        target_j_pos = self.current_joint_pos.copy() # 基于当前真实位置进行累加
        target_j_pos[:3] += delta_pos # 前3个关节加上网络计算出的位置增量
        target_j_pos[3] = 0.0 # 第4个关节在位置任务中被锁定为0.0 (与 direct_joint4 动作配置一致)
        
        # SAFE RL: Joint limits (关节物理限位检查)
        # 定义4个关节的物理运动范围 [-1.57, 1.57] 弧度 (约±90度)
        joint_limits = [(-1.57, 1.57), (-1.57, 1.57), (-1.57, 1.57), (-1.57, 1.57)]
        for i in range(4):
            # 将目标位置强制裁剪在物理限位内，防止机械结构发生硬碰撞
            target_j_pos[i] = np.clip(target_j_pos[i], joint_limits[i][0], joint_limits[i][1])
            
        # 调用发布函数，将计算好的安全目标位置发送给底层电机控制器
        self.publish_trajectory(target_j_pos)

    def publish_trajectory(self, target_j_pos):
        """
        将目标关节位置打包为 ROS2 的 JointTrajectory 消息并发布。
        """
        msg = JointTrajectory() # 创建轨迹消息对象
        msg.header.stamp = self.get_clock().now().to_msg() # 打上当前的时间戳
        msg.joint_names = self.joint_names # 填入关节名称列表
        
        pt = JointTrajectoryPoint() # 创建轨迹点对象
        pt.positions = target_j_pos.tolist() # 填入目标位置列表
        
        # 对于 ros2_control 的 joint_trajectory_controller，需要指定到达该点的时间(time_from_start)
        dt = 1.0 / self.control_rate # 单步时间周期
        pt.time_from_start.nanosec = int(dt * 1e9) # 将秒转换为纳秒填入
        
        msg.points.append(pt) # 将轨迹点加入消息中
        self.traj_pub.publish(msg) # 发布消息

def main():
    """
    节点入口函数。
    """
    rclpy.init() # 初始化ROS2环境
    node = Sim2RealPolicyNode() # 实例化我们编写的策略节点
    try:
        rclpy.spin(node) # 阻塞并保持节点运行，处理所有回调函数
    except KeyboardInterrupt:
        pass # 捕捉Ctrl+C中断信号，优雅退出
    finally:
        node.destroy_node() # 销毁节点
        rclpy.shutdown() # 关闭ROS2环境

if __name__ == '__main__':
    main() # 如果作为主程序运行，则调用入口函数