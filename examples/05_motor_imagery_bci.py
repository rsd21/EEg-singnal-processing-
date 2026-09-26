"""Motor imagery BCI: which channels respond to imagined left/right hand movement?

Imagining a hand movement suppresses the mu (8-13 Hz) and beta rhythms over
the opposite motor cortex (event-related desynchronisation, ERD). The script
ranks channels by ERD, by class separability and by single-channel decoding
accuracy.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import eegproc as ep  # noqa: E402
from eegproc.quality import decoding_accuracy, discriminability, erd_ers  # noqa: E402
from eegproc.viz import plot_erds_timecourse  # noqa: E402

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

if len(sys.argv) > 1:
    raw = ep.read_raw(sys.argv[1])
    raw.set_montage("standard_1010")
    conditions = sys.argv[2:4] if len(sys.argv) >= 4 else None
else:
    from eegproc.datasets import simulate_epochs_dataset

    raw = simulate_epochs_dataset("motor_imagery", n_trials=60, montage="32", seed=4)
    conditions = ["left", "right"]

raw = raw.filter(1.0, 40.0)
epochs = ep.Epochs(raw, tmin=-1.0, tmax=4.0, baseline=None, reject=200.0)
if conditions:
    epochs = epochs[conditions]
print(epochs)

for cond in epochs.event_id:
    r = erd_ers(epochs, band=(8, 13), baseline=(-1.0, 0.0), window=(0.5, 3.5), condition=cond)
    print(f"{cond:>6}: strongest mu change at {r.top(3)}")
    r.plot().savefig(f"{OUT}/erd_{cond}.png", bbox_inches="tight")

plot_erds_timecourse(epochs, ["C3", "C4"] if "C3" in epochs.ch_names else epochs.ch_names[:2]).savefig(
    f"{OUT}/erd_timecourse.png", bbox_inches="tight")

disc = discriminability(epochs, window=(0.5, 3.5))
print("\n" + disc.summary(max_rows=6))
dec = decoding_accuracy(epochs, window=(0.5, 3.5))
print("\n" + dec.summary(max_rows=6))
dec.plot().savefig(f"{OUT}/rank_decoding.png", bbox_inches="tight")
