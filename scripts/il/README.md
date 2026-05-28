# IL Workflow for `Template-Demo-Learn-v0`

This folder contains an imitation-learning loop that stays inside Isaac Lab:

1. Generate a safe FK candidate point cloud.
2. Collect successful global-IK expert demonstrations on that point cloud.
3. Train a behavior-cloning MLP on `obs -> action`.
4. Play the BC policy in the same Gym environment.
5. Collect DAgger data by rolling out BC and relabeling visited states with global reachable-target IK.
6. Retrain BC on expert + DAgger datasets, then repeat if needed.

The current task action is `DeltaJointPositionAction`, so the dataset stores normalized delta actions:

```text
expert_action = clamp((q_next_expert - q_current) / 0.05, -1, 1)
```

## 1. Prepare Safe FK Candidate Point Cloud

First generate a safe FK candidate point cloud. Its joint sampling range defines which IK solutions are allowed:

```bash
python scripts/generate_reachable_targets.py \
  --joint4 0.0 \
  --min_up_dot 0.98 \
  --output source/DEMO_learn/DEMO_learn/tasks/manager_based/demo_learn/reachable_targets.pt \
  --headless
```

This produces FK pairs:

```text
target_position <-> q_goal
end_effector_local_z || base_z
```

By default the script samples `trunk_joint1/2/3` over `[-1.57, 1.57]`,
matching the environment action clamp. The wider third-joint range is important because fixed end-effector orientation
makes `trunk_joint3` compensate the upstream joint angles.

If the printed robot soft limit for `trunk_joint3` is still around `[-1.57, 1.57]`, update the USD physics joint limit
with Isaac Sim/Lab Python:

```bash
python scripts/set_usd_joint_limits.py \
  --usd source/DEMO_learn/trunk_robot/configuration/trunk_robot_physics.usd \
  --joint trunk_joint3 \
  --lower_rad -1.57 \
  --upper_rad 1.57
```

## 2. Collect Global IK Expert Data 收集全局IK反解专家轨迹

This is the preferred first expert for the current task. The environment samples `target_pose` normally. For each
sampled `target_pos`, the script solves global sampled IK over `reachable_targets.pt`, builds a time-indexed joint
reference, steps the Isaac Lab environment, and saves only successful trajectories:

```text
q_goal = argmin_q ||FK(q) - target_pos||^2 + w_up * up_error(q) + w_joint * ||q - q_current||^2
q_ref(t) = q_start + alpha(t) * (q_goal - q_start)
expert_action(t) = clamp((q_ref(t + 1) - q_current) / 0.05, -1, 1)
```

The script still steps the Isaac Lab environment, so observations, PD tracking, reset distribution, and termination are
from the same simulator as RL.

```bash
python scripts/il/collect_global_ik_dataset.py \
  --task Template-Demo-Learn-v0 \
  --num_envs 512 \
  --num_samples 200000 \
  --reachable_targets source/DEMO_learn/DEMO_learn/tasks/manager_based/demo_learn/reachable_targets.pt \
  --output datasets/demo_learn/global_ik_expert.pt \
  --traj_delta 0.025 \
  --hold_steps 60 \
  --headless
```

Use the same Python launcher/environment that you normally use for Isaac Lab scripts.

`--traj_delta` controls the nominal joint-space progress per policy step. `--hold_steps` keeps commanding `q_goal`
after interpolation ends so the PD controller can settle. The script saves the trajectory only when it terminates
through `target_reached`; timed-out trajectories are discarded.

### Optional: Paired FK Target Collection

`collect_reachable_joint_trajectory_dataset.py` directly samples paired `target_position <-> q_goal` rows from
`reachable_targets.pt`. It is useful for debugging the FK table, but for expert data matching the real environment
command distribution, prefer `collect_global_ik_dataset.py`.

## Diagnose Reachable Target Pairs

If `reachable_joint_expert` has a high timeout rate, first check whether the saved FK pairs still match the current
environment:

```bash
python scripts/il/diagnose_reachable_targets.py \
  --task Template-Demo-Learn-v0 \
  --num_envs 512 \
  --num_checks 4096 \
  --reachable_targets source/DEMO_learn/DEMO_learn/tasks/manager_based/demo_learn/reachable_targets.pt \
  --rollout_steps 1800 \
  --headless
```

Read the output as follows:

```text
DIRECT_FK_ERROR
  Directly writes q_goal into sim and compares FK(q_goal) with target_pos.
  If this is often > 0.05m, reachable_targets.pt does not match the current env/frame/joint order.

ROLLOUT_FAILED_LAST_JOINT_ERROR
  For failed rollouts, how far q was from q_goal near timeout.
  Large value means the controller/action limit/episode length did not reach q_goal.

ROLLOUT_FAILED_LAST_POS_ERROR
  For failed rollouts, how far the end-effector was from target near timeout.
  Small joint error but large pos error means q_goal and target_pos are inconsistent.
```

## 3. Train Behavior Cloning

```bash
python scripts/il/train_bc_policy.py \
  datasets/demo_learn/global_ik_expert.pt \
  --output logs/il/demo_learn/bc_policy.pt \
  --epochs 100 \
  --batch_size 4096
```

## 4. Play the BC Policy

```bash
python scripts/il/play_bc_policy.py \
  --task Template-Demo-Learn-v0 \
  --checkpoint logs/il/demo_learn/bc_policy.pt \
  --num_envs 64 \
  --headless
```

The playback script prints cumulative episode metrics:

```text
success_rate: episodes that terminated through target_reached
timeout_rate: episodes that reached the time limit
success_now: current-frame fraction inside the success radius
```

Use `success_rate`, not single-frame `success_now`, to evaluate closed-loop performance.

## 5. Collect DAgger Data With Global IK Relabeling

If BC playback is weak, collect states visited by the BC policy and relabel them with the global reachable-target IK
expert:

```bash
python scripts/il/collect_dagger_dataset.py \
  --task Template-Demo-Learn-v0 \
  --checkpoint logs/il/demo_learn/bc_policy.pt \
  --num_envs 512 \
  --num_samples 200000 \
  --reachable_targets source/DEMO_learn/DEMO_learn/tasks/manager_based/demo_learn/reachable_targets.pt \
  --output datasets/demo_learn/dagger_iter1.pt \
  --headless
```

This script executes the BC action in simulation, but saves the global IK expert action for the state BC actually
visited:

```text
visited_obs = obs_from_bc_rollout
q_goal = argmin_q ||FK(q) - target_pos||^2 + w_up * up_error(q) + w_joint * ||q - q_current||^2
label = clamp((q_goal - q_current) / 0.05, -1, 1)
```

This directly targets BC distribution shift: the policy learns recovery actions for states it reaches after its own
mistakes. The search is global over `reachable_targets.pt`, so it respects the joint ranges used when generating that
file instead of following a local Jacobian branch.

## 6. Retrain With Aggregated Data

Train a new BC policy using both the original successful expert trajectories and DAgger relabels:

```bash
python scripts/il/train_bc_policy.py \
  datasets/demo_learn/global_ik_expert.pt \
  datasets/demo_learn/dagger_iter1.pt \
  --output logs/il/demo_learn/bc_policy_dagger1.pt \
  --epochs 100 \
  --batch_size 4096
```

Then evaluate:

```bash
python scripts/il/play_bc_policy.py \
  --task Template-Demo-Learn-v0 \
  --checkpoint logs/il/demo_learn/bc_policy_dagger1.pt \
  --num_envs 64 \
  --headless
```

If the cumulative success rate is still poor, repeat:

```text
bc_policy_dagger1.pt -> collect_dagger_dataset.py -> dagger_iter2.pt
global_ik_expert.pt + dagger_iter1.pt + dagger_iter2.pt -> train_bc_policy.py
```

## Notes

- The first training stage fixes the reset distribution in `demo_learn_env_cfg.py`; widen it only after the global IK expert and BC playback are stable.
- If the global IK expert still times out often, inspect whether the saved `q_goal` uses the same joint names, joint limits, end-effector frame, and fixed `trunk_joint4` convention as the environment.
- Initial expert collection saves only successful expert trajectories; DAgger collection saves BC-visited states regardless of whether the BC episode eventually succeeds.
- The BC checkpoint is currently a standalone Torch policy for validation. Use it to verify that expert data and observation/action alignment are correct before wiring it into skrl PPO fine-tuning.
