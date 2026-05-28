import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_file = sys.argv[1]
ea = EventAccumulator(log_file)
ea.Reload()

events = ea.Scalars("Reward / Total reward (mean)")
print(f"Number of data points: {len(events)}")
for e in events:
    print(f"Step: {e.step}, Value: {e.value}")

