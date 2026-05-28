import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

TRUNK_ROBOT_USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../trunk_robot/trunk_robot.usd"))
END_EFFECTOR_BODY_NAME = "trunk_link4"
REACHABLE_TARGETS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "reachable_targets.pt"))


def ee_cfg() -> SceneEntityCfg:
    """Return the end-effector entity used consistently by observations and rewards."""
    return SceneEntityCfg("robot", body_names=[END_EFFECTOR_BODY_NAME])

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
            solver_velocity_iteration_count=0,  # 物理引擎求解器：速度迭代次数
            fix_root_link=True,  # 固定根链接，避免“机器人躺倒后目标随基座漂移”
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # 机器人的初始三维空间坐标位置 (X, Y, Z)，单位：m
        # 固定预弯曲姿态：避免从全零伸直/奇异姿态开始探索。
        joint_pos={
            "trunk_joint1": 0.8,
            "trunk_joint2": -1.3,
            "trunk_joint3": 1.2,
            "trunk_joint4": 0.0,
        },
        joint_vel={".*": 0.0},  # 所有关节的初始速度。单位：rad/s 或 m/s
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # 指定该驱动器控制哪些关节，".*"代表所有
            effort_limit=100.0,  # 关节驱动的最大力/力矩限制，单位：N 或 N·m
            velocity_limit=3.0,  # 关节的最大速度限制，单位：rad/s 或 m/s
            stiffness=4000.0,  # PD控制器的刚度系数（Kp），相当于弹簧的硬度，影响向目标位置移动的力度
            damping=100.0,  # PD控制器的阻尼系数（Kd），相当于阻力，用于抑制震荡
        ),
    },
)

@configclass
class DemoLearnSceneCfg(InteractiveSceneCfg):
    """Configuration for the trunk robot scene."""

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
    target_pose = mdp.AnnulusPoseCommandCfg(
        asset_name="robot",
        body_name=END_EFFECTOR_BODY_NAME,
        resampling_time_range=(30.0, 30.0),  # 指令多久重新采样一次；与30s episode对齐，避免目标中途变化。
        debug_vis=False,  # 训练时关闭目标可视化，避免联网加载 Isaac marker USD 失败
        # 圆环采样参数：在 base frame 的 X-Z 平面以(center_x, center_z)为圆心采样可达目标
        center_x=0.0,
        center_z=0.0,
        radius_min=0.25, # 避开靠近基座的奇异/高曲率区域
        radius_max=0.65, # 覆盖测试常用目标，例如 (x=0.15, z=0.60) 的半径约0.62m
        uniform_area=True,
        theta_min=0.35,  # 避免接近水平边界的困难目标
        theta_max=math.pi - 0.35,
        sample_reachable_poses=False, # 不再从当前末端集合采样，避免训练目标分布塌缩导致虚高成功率
        reachable_targets_path=REACHABLE_TARGETS_PATH, # 从离线FK点云采样，确保每个训练目标真实可达
        ranges=mdp.AnnulusPoseCommandCfg.Ranges(
            pos_x=(-0.2, 0.2),  # 占位参数：AnnulusPoseCommand 不使用 pos_x
            pos_y=(0, 0),       # Y轴固定为0（平面任务）
            pos_z=(0.5, 0.8),   # 占位参数：AnnulusPoseCommand 不使用 pos_z
            # 位置任务只使用 target_pose 的 xyz；姿态固定为单位四元数，避免观测混入无关随机量。
            roll=(0.0, 0.0),  # 目标姿态 roll (横滚角) 范围，单位：rad
            pitch=(0.0, 0.0), # 目标姿态 pitch (俯仰角) 范围，单位：rad
            yaw=(0.0, 0.0),   # 目标姿态 yaw (偏航角) 范围，单位：rad
        ),
    )
    target_joint4 = mdp.UniformJointCommandCfg(
        resampling_time_range=(30.0, 30.0),
        debug_vis=False,
        ranges=mdp.UniformJointCommandCfg.Ranges(
            joint_pos=(0.0, 0.0), # 位置任务先锁定第4关节，避免随机本体转角干扰前三关节到达目标
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # 前 3 个关节由网络输出关节位置增量；0.05 rad @ 60Hz 约等于 3 rad/s
    joint_position_delta = mdp.DeltaJointPositionActionCfg(
        asset_name="robot",
        joint_names=["trunk_joint1", "trunk_joint2", "trunk_joint3"], # 只控制前三个关节
        scale=0.05,
        joint_limits=[(-1.57, 1.57), (-1.57, 1.57), (-1.57, 1.57)],
    )

    # 第 4 关节直接读取指令，不经过神经网络
    direct_joint4 = mdp.DirectJointCommandActionCfg(
        asset_name="robot",
        joint_name="trunk_joint4",
        command_name="target_joint4",
        command_index=0
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel) # 相对关节位置 (相对于默认状态)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel) # 相对关节速度
        end_effector_pose = ObsTerm(func=mdp.end_effector_pose, params={"asset_cfg": ee_cfg()}) # 末端执行器位姿
        target_pose = ObsTerm(func=mdp.target_pose) # 目标位姿
        position_error = ObsTerm(func=mdp.position_error, params={"asset_cfg": ee_cfg()}) # 目标位置误差

        def __post_init__(self) -> None:
            self.enable_corruption = False  # 是否为观测值加入随机噪声（用于模拟真实传感器的误差，做sim2real迁移时常打开）
            self.concatenate_terms = True   # 是否将上述所有的观测值拼接成一个大的一维向量传入神经网络

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",  # 触发时机：在环境 reset（回合重置）时触发
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["trunk_joint1", "trunk_joint2", "trunk_joint3"],
            ),
            "position_range": (0.0, 0.0),  # 第一阶段固定初始姿态，先建立可验证的闭集目标分布。
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_joint4 = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["trunk_joint4"]),
            "position_range": (1.0, 1.0),  # 第4关节保持默认0位，由 DirectJointCommandAction 锁定
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 动作平滑惩罚（防止原地高频抽搐和抖动）
    action_rate_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.001 # 进一步降低动作变化惩罚，鼓励探索
    )

    episode_time_penalty = RewTerm(
        func=mdp.time_penalty,
        weight=-0.1 # 每秒持续扣分，避免策略在目标附近拖到超时
    )

    # Task specific rewards
    target_pos_error = RewTerm(
        func=mdp.end_effector_position_error,
        weight=-10.0, # 直接惩罚距离；不再让机器人靠近目标后按时间刷正奖励
        params={"asset_cfg": ee_cfg()}
    )

    target_pos_error_squared = RewTerm(
        func=mdp.end_effector_position_error_squared,
        weight=-20.0, # 大误差额外惩罚，帮助早期从远处拉回目标区域
        params={"asset_cfg": ee_cfg()}
    )

    target_pos_excess_error = RewTerm(
        func=mdp.end_effector_position_excess_error,
        weight=-30.0, # 只惩罚5cm成功圈外的剩余误差，强化最后十几厘米的进圈压力
        params={
            "asset_cfg": ee_cfg(),
            "pos_threshold": 0.05,
        }
    )

    mid_pos_tracking = RewTerm(
        func=mdp.end_effector_position_fine_tracking,
        weight=2.0, # 中距离引导：帮助增量控制策略先从20cm量级进入10cm量级
        params={"asset_cfg": ee_cfg(), "std": 0.15}
    )

    fine_pos_tracking = RewTerm(
        func=mdp.end_effector_position_fine_tracking,
        weight=3.0, # 小权重精度奖励：强化5cm附近梯度，但避免靠近目标刷分压过终止奖励
        params={"asset_cfg": ee_cfg(), "std": 0.05}
    )

    up_axis_error = RewTerm(
        func=mdp.end_effector_up_axis_error,
        weight=-2.0, # 约束末端局部+Z保持与基坐标+Z平行，消除三连杆位置任务的连续冗余解。
        params={"asset_cfg": ee_cfg()}
    )

    # 过程奖励：靠近目标奖励，远离目标惩罚
    approach_target = RewTerm(
        func=mdp.approach_target_reward,
        weight=1.0, # 保留靠近目标的方向引导，但避免速度项主导奖励
        params={"asset_cfg": ee_cfg()}
    )

    # 任务达成终极奖励
    target_reached_bonus = RewTerm(
        func=mdp.target_reached_bonus,
        weight=3000.0,
        params={
            "asset_cfg": ee_cfg(),
            "pos_threshold": 0.05,
            "up_axis_threshold": 0.98,
        }
    )

    # 成功率使用 Curriculum/all_env_pos_success 和 Episode_Termination/target_reached 监控；
    # 不再把 success_count 作为奖励项，避免它参与优化并混淆 TensorBoard 奖励曲线。

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True) # 超时终止：如果回合运行时间到达 episode_length_s 的上限，正常结束并重置

    # 目标到达终止：重新启用！并在Rewards中搭配了巨大的成功奖励。
    target_reached = DoneTerm(
        func=mdp.reached_target_pose,
        params={
            "asset_cfg": ee_cfg(),
            "pos_threshold": 0.05, # pos_threshold: 位置到达的判定容差阈值，单位：m。距离目标小于 5cm 即视为到达。
            "up_axis_threshold": 0.98, # 末端局部+Z与基坐标+Z夹角约小于11.5度。
        }
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # 注意：CurriculumManager 只在 reset env_ids 上触发。这里的函数会忽略 env_ids，
    # 统计所有并行环境的当前状态，避免 reset batch 成功率被误读成全局成功率。
    all_env_pos_success = CurriculumTerm(
        func=mdp.log_all_env_pos_success,
        params={
            "asset_cfg": ee_cfg(),
            "pos_threshold": 0.05,
        }
    )
    all_env_pos_error = CurriculumTerm(
        func=mdp.log_all_env_pos_error,
        params={
            "asset_cfg": ee_cfg(),
        }
    )

@configclass
class DemoLearnEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DemoLearnSceneCfg = DemoLearnSceneCfg(num_envs=4096, env_spacing=4.0) # num_envs: 并行模拟的机器人数量；env_spacing: 每个机器人互相隔开的距离，单位：m
    # Basic settings
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2 # 控制降采样率（跳帧数）：物理引擎每计算 decimation 步，神经网络才下发一次动作。单位：物理仿真步数
        self.episode_length_s = 30.0 # 单个回合(Episode)的最大存活时长，单位：秒(s)
        self.viewer.eye = (3.0, 3.0, 3.0) # 图形界面打开时，观察相机的初始位置 (X, Y, Z)，单位：m
        self.sim.dt = 1 / 120 # 物理引擎的底层仿真步长（每一帧流逝的时间），单位：秒(s)。此例中约等于 0.0083s
        self.sim.render_interval = self.decimation # 渲染间隔，通常设为和 decimation 一样，避免无意义的画面渲染浪费性能
