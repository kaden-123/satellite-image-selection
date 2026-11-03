import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from Sim import run_simulation
from MetricsRecorder import MetricsRecorder


def summary_average_runs(config, policy, scoring, recorder, n) -> dict:
    run_seeds = [
        int(child.generate_state(1)[0])
        for child in np.random.SeedSequence(config.rand_seed).spawn(n)
    ]

    summaries = []
    for seed in run_seeds:
        config.rand_seed = seed
        run_simulation(config, policy, scoring, recorder)
        summary, _ = recorder.to_dataframes()
        summaries.append(summary.iloc[0].to_dict())
        recorder.reset()

    averages = pd.DataFrame(summaries).mean(numeric_only=True).to_dict()
    averages.pop("random_seed", None)
    averages["policy_name"] = recorder.policy_name
    return averages
    

def compare_policies(config, policies, scoring, bandwidths, n=10):
    """Plot raw delivered-image curves; return averaged summaries separately."""
    fig, axes = plt.subplots(len(bandwidths), 1, figsize=(9, 4 * len(bandwidths)),
                             squeeze=False)
    rows = []
    for i, bandwidth in enumerate(bandwidths):
        run_config = replace(config, bandwidth=bandwidth)
        ax = axes[i, 0]
        for policy in policies:
            recorder = MetricsRecorder(
                policy_name=policy.name,
                random_seed=run_config.rand_seed,
                available_downlink_mb_per_contact=(
                    bandwidth * run_config.contact_duration * 60 / 8
                ),
            )
            run_simulation(run_config, policy, scoring, recorder)
            _, images = recorder.to_dataframes()
            images = images[images["capture_time"] != 0]
            images = images[images["downlink_time"].notna()]
            images = images.sort_values("downlink_time")
            ax.plot(images["downlink_time"], images["delivered_value"].cumsum(),
                    label=recorder.policy_name)

            rows.append({"bandwidth": bandwidth, **summary_average_runs(
                run_config, policy, scoring, recorder, n
            )})

        ax.set(xlabel="Mission time (days)", ylabel="Cumulative delivered value",
               title=f"Bandwidth: {bandwidth} Mb/s")
        ax.legend()

    fig.tight_layout()
    return pd.DataFrame(rows), fig, axes