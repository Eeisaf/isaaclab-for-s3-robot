import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_file = sys.argv[1]
ea = EventAccumulator(log_file)
ea.Reload()

tags_to_check = [
    "Reward / Total reward (mean)",
    "Episode_Reward/success_count",
    "Curriculum/pos_only_success",
    "Episode_Reward/target_pos_tracking",
    "Episode_Reward/target_reached_bonus",
    "Episode_Reward/terminating",
    "Episode_Reward/action_rate_penalty",
    "Episode_Reward/upright_penalty",
    "Episode_Reward/phase_joint_penalty",
    "Episode_Reward/joint_limit",
    "Episode / Total timesteps (mean)",
    "Loss / Value loss",
    "Loss / Policy loss",
    "Loss / Entropy loss"
]

print("--- Latest Metrics ---")
for tag in tags_to_check:
    try:
        events = ea.Scalars(tag)
        if events:
            latest = events[-1]
            print(f"{tag}: step={latest.step}, value={latest.value:.4f}")
    except Exception as e:
        print(f"{tag}: Not found")

print("\n--- Success Rate Trend ---")
for tag in ["Episode_Reward/success_count", "Curriculum/pos_only_success"]:
    try:
        events = ea.Scalars(tag)
        if events:
            # Print every 10th value to see the trend
            vals = [e.value for e in events]
            steps = [e.step for e in events]
            print(f"{tag} trend:")
            for i in range(0, len(vals), max(1, len(vals)//10)):
                print(f"  Step {steps[i]}: {vals[i]:.4f}")
            print(f"  Step {steps[-1]}: {vals[-1]:.4f}")
    except Exception as e:
        pass
