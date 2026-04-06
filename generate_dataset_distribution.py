from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    output_path = Path(__file__).resolve().parent / "assets" / "dataset_distribution.png"

    splits = ["TRAIN", "VAL", "TEST"]
    vehicle = [765919, 109819, 219870]
    person = [95917, 13910, 25931]
    traffic_sign = [239881, 34905, 68987]
    traffic_light = [186205, 26883, 52814]

    x = np.arange(len(splits))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 8))

    bars1 = ax.bar(x - 1.5 * width, vehicle, width, label="Vehicle", color="#3b7ddd")
    bars2 = ax.bar(x - 0.5 * width, person, width, label="Person", color="#ff7f0e")
    bars3 = ax.bar(x + 0.5 * width, traffic_sign, width, label="Traffic Sign", color="#22c55e")
    bars4 = ax.bar(x + 1.5 * width, traffic_light, width, label="Traffic Light", color="#ef4444")

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=16)
    ax.set_ylabel("Bounding Boxes", fontsize=16)
    ax.set_title("BDD100K Label Distribution After Four-Class Conversion", fontsize=22)
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=14)

    for bars in (bars1, bars2, bars3, bars4):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 5000,
                f"{height / 1000:.1f}k",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=12,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
