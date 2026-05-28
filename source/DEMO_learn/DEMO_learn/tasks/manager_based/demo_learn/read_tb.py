import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_file = sys.argv[1]
ea = EventAccumulator(log_file)
ea.Reload()

tags = ea.Tags()['scalars']
print("Available tags:", tags)

metrics_to_check = [
    "Reward / Total",
    "Reward / success_count",
    "Reward / pos_only_success",
    "Reward / ori_only_success",
    "Episode / Length",
    "Policy / Entropy",
    "Loss / Value"
]

print("\n--- Latest Metrics ---")
for tag in tags:
    if any(m in tag for m in ["Reward", "success", "Length", "Entropy", "Loss / Value", "KL"]):
        events = ea.Scalars(tag)
        if events:
            latest = events[-1]
            print(f"{tag}: step={latest.step}, value={latest.value:.4f}")
