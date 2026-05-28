import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "/home/sythoid/DEMO_learn/logs/skrl/cartpole_direct/2026-05-21_13-41-37_ppo_torch"
event_file = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))[0]
ea = EventAccumulator(event_file)
ea.Reload()

events = ea.Scalars("Episode_Reward/approach_target")
print("First 5 values for approach_target:")
for e in events[:5]:
    print(f"Step: {e.step}, Value: {e.value}")
