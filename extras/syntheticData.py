import numpy as np
import matplotlib.pyplot as plt

MINUTES_PER_DAY = 24 * 60
DAYS = 7
TOTAL_POINTS = DAYS * MINUTES_PER_DAY

BASELINE_RPS = 25
WINDOW_SIZE = 30
HORIZON = 12

t = np.arange(TOTAL_POINTS)

baseline = np.full(TOTAL_POINTS, BASELINE_RPS)

daily_period = MINUTES_PER_DAY
seasonality_amplitude = 20  # you can tune this

seasonality = seasonality_amplitude * np.sin(
    2 * np.pi * t / daily_period
)

rps = baseline + seasonality

noise_std = 3  # small
noise = np.random.normal(0, noise_std, TOTAL_POINTS)

rps = rps + noise

spikes_per_day = 3
spike_duration = 20
spike_height = 50  # added RPS during spike

for day in range(DAYS):
    day_start = day * MINUTES_PER_DAY
    day_end = day_start + MINUTES_PER_DAY

    num_spikes = np.random.randint(
        spikes_per_day - 1,
        spikes_per_day + 2
    )

    for _ in range(num_spikes):
        spike_start = np.random.randint(
            day_start,
            day_end - spike_duration
        )
        spike_end = spike_start + spike_duration

        rps[spike_start:spike_end] += spike_height

rps = np.clip(rps, 0, None)        

plt.figure(figsize=(12, 4))
plt.plot(rps)
plt.title("Synthetic RPS — 7 Days")
plt.show()

np.save("rps_raw.npy", rps)
