import math
import os
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg

from . import mdp

TRUNK_ROBOT_USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../trunk_robot/trunk_robot.usd"))

from isaaclab.actuators import ImplicitActuatorCfg

TRUNK_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRUNK_ROBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 是否禁用重力
            retain_accelerations=False,  # 是否在时间步之间保留加速度
            linear_damping=0.0,  # 线性阻尼，单位：无量纲或 N/(m/s)
            angular_damping=0.0,  # 角阻尼，单位：无量纲或 N·m/(rad/s)
            max_linear_velocity=1000.0,  # 最大线速度限制，单位：m/s
            max_angular_velocity=1000.0,  # 最大角速度限制，单位：rad/s
            max_depenetration_velocity=1.0,  # 最大抗穿透（去重叠）速度，用于解决刚体碰撞穿透，单位：m/s
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 是否允许机器人的不同连杆之间发生自碰撞
            solver_position_iteration_count=4,  # 物理引擎求解器：位置迭代次数，数值越大越精确但越耗时
            solver_velocity_iteration_count=0   # 物理引擎求解器：速度迭代次数
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # 机器人的初始三维空间坐标位置 (X, Y, Z)，单位：m
        joint_pos={".*": 0.0},  # 所有关节的初始角度/位置，".*"为正则匹配所有，0.0为初始值。如果是转动关节，单位是 rad(弧度)；移动关节单位是 m(米)
        joint_vel={".*": 0.0},  # 所有关节的初始速度。单位：rad/s 或 m/s
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # 指定该驱动器控制哪些关节，".*"代表所有
            effort_limit=100.0,  # 关节驱动的最大力/力矩限制，单位：N 或 N·m
            velocity_limit=10.0,  # 关节的最大速度限制，单位：rad/s 或 m/s
            stiffness=800.0,  # PD控制器的刚度系数（Kp），相当于弹簧的硬度，影响向目标位置移动的力度
            damping=40.0,  # PD控制器的阻尼系数（Kd），相当于阻力，用于抑制震荡
        ),
    },
)

@configclass
class DemoLearnSceneCfg(InteractiveSceneCfg):
    """Configuration for the trunk robot scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),  # 生成 100m x 100m 的物理地面，单位：m
    )

    # robot
    robot: ArticulationCfg = TRUNK_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),  # 生成穹顶光，color为(R,G,B)，intensity为光照强度
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    target_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name=".*", # we will override this logic or ignore it if not needed, but UniformPoseCommand requires it
        resampling_time_range=(5.0, 5.0),  # 指令多久重新采样一次（即多久下发一次新目标），单位：秒(s)。这里是固定每 5 秒换一次目标。
        debug_vis=True,  # 是否在可视化界面中显示目标位置的虚拟标记
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(-0.5, 0.5),         # X轴目标范围（前后移动），单位：m
            pos_y=(0.0, 0.0),          # Y轴固定为0。因为是平面机器人，必须限制在竖直 X-Z 平面内。
            pos_z=(0.5, 1),          # Z轴目标高度范围（上下移动），单位：m
            roll=(-math.pi, math.pi),           # 平面机器人不应该有 Roll (绕X轴) 旋转
            pitch=(-math.pi, math.pi), # 目标姿态 pitch (俯仰角，即绕Y轴的旋转)，平面内唯一自由的旋转角
            yaw=(0.0, 0.0),            # 平面机器人不应该有 Yaw (绕Z轴) 旋转
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # 策略网络输出的动作类型：关节位置控制 (JointPositionAction)
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=1.0  # scale: 对网络输出动作的缩放系数。网络输出一般在[-1,1]之间，会被乘以scale作为最终发给控制器的目标指令。单位与关节类型有关(rad 或 m)
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel) # 相对关节位置 (相对于默认状态)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel) # 相对关节速度
        end_effector_pose = ObsTerm(func=mdp.end_effector_pose, params={"asset_cfg": SceneEntityCfg("robot")}) # 末端执行器位姿
        target_pose = ObsTerm(func=mdp.target_pose) # 目标位姿

        def __post_init__(self) -> None:
            self.enable_corruption = False  # 是否为观测值加入随机噪声（用于模拟真实传感器的误差，做sim2real迁移时常打开）
            self.concatenate_terms = True   # 是否将上述所有的观测值拼接成一个大的一维向量传入神经网络

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",  # 触发时机：在环境 reset（回合重置）时触发
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (0.0, 0.0),  # 重置时对初始位置加入的随机扰动范围。0.0表示严格回到初始位置，不加扰动。单位：依赖底层函数的实现，通常是乘数因子
            "velocity_range": (0.0, 0.0),  # 重置时对初始速度加入的随机扰动范围。
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0) # 存活奖励：每存活一个步长给予 1.0 的奖励权重
    
    terminating = RewTerm(func=mdp.is_terminated, weight=-10.0) # 终止惩罚：如果因为失败条件触发终止（如提前摔倒），一次性扣除 10.0 分
    
    # Task specific rewards
    target_pos_tracking = RewTerm(
        func=mdp.end_effector_position_tracking,
        weight=10.0, # 追踪目标位置的奖励权重：10.0
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.1} # std: 计算高斯衰减函数的距离标准差，越小要求越严苛。单位：m
    )
    
    target_ori_tracking = RewTerm(
        func=mdp.end_effector_orientation_tracking,
        weight=5.0, # 追踪目标姿态的奖励权重：5.0
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.1} # std: 旋转四元数误差衰减的标准差
    )
    
    # Phase specific
    upright_penalty = RewTerm(
        func=mdp.upright_posture_penalty,
        weight=-2.0, # 偏离直立姿态的惩罚权重：-2.0
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    phase_joint_penalty = RewTerm(
        func=mdp.phase_based_penalty,
        weight=-5.0, # 高度达标后约束底层关节乱动的惩罚权重：-5.0
        params={"asset_cfg": SceneEntityCfg("robot"), "height_threshold": 0.05, "frozen_joints": [0, 1]} 
        # frozen_joints=[0, 1, 2] 表示：到达高度后，锁定前三个关节，专门留出最后一个关节去调姿态
    )

    # Joint limit penalty (random weight)
    joint_limit = RewTerm(
        func=mdp.joint_limit_penalty,
        weight=-random.uniform(1.0, 5.0), # 关节越界惩罚权重：每次运行环境时在 [-5.0, -1.0] 之间随机抽取（域随机化技术）
        params={"asset_cfg": SceneEntityCfg("robot"), "bounds": [
            (-math.pi / 2, math.pi / 2),   # 关节 0 的限制 (-90度 到 90度)，单位：rad
            (-2 * math.pi / 4, 2 * math.pi / 4),   # 关节 1 的限制 (-120度 到 120度)，单位：rad
            (-math.pi / 3, math.pi / 3),   # 关节 2 的限制 (-60度 到 60度)，单位：rad
            (-math.pi / 2, math.pi / 2)            # 关节 3 的限制 (-90度 到 90度)，单位：rad
        ]}
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True) # 超时终止：如果回合运行时间到达 episode_length_s 的上限，正常结束并重置
    
    target_reached = DoneTerm(
        func=mdp.reached_target_pose,
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "pos_threshold": 0.05, # pos_threshold: 位置到达的判定容差阈值，单位：m。距离目标小于 5cm 即视为到达。
            "quat_threshold": 0.1  # quat_threshold: 旋转到达的判定容差阈值（基于四元数差异计算的数值，无单位）。
        }
    )

@configclass
class DemoLearnEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DemoLearnSceneCfg = DemoLearnSceneCfg(num_envs=256, env_spacing=4.0) # num_envs: 并行模拟的机器人数量；env_spacing: 每个机器人互相隔开的距离，单位：m
    # Basic settings
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2 # 控制降采样率（跳帧数）：物理引擎每计算 decimation 步，神经网络才下发一次动作。单位：物理仿真步数
        self.episode_length_s = 10.0 # 单个回合(Episode)的最大存活时长，单位：秒(s)
        self.viewer.eye = (3.0, 3.0, 3.0) # 图形界面打开时，观察相机的初始位置 (X, Y, Z)，单位：m
        self.sim.dt = 1 / 120 # 物理引擎的底层仿真步长（每一帧流逝的时间），单位：秒(s)。此例中约等于 0.0083s
        self.sim.render_interval = self.decimation # 渲染间隔，通常设为和 decimation 一样，避免无意义的画面渲染浪费性能
