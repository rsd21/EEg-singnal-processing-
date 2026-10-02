"""Which channel performs best? Several answers for several questions.

* signal quality (clean, well-connected electrodes)
* signal-to-noise ratio
* where each rhythm (delta ... gamma) is strongest
* which channels are network hubs (alpha-band wPLI)
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import eegproc as ep  # noqa: E402
from eegproc.quality import band_activity, dominant_bands  # noqa: E402
from eegproc.utils import format_table  # noqa: E402
from eegproc.viz import plot_band_topomaps, plot_connectivity, plot_sensors  # noqa: E402
from eegproc.viz._style import GRADE_STATUS  # noqa: E402

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

if len(sys.argv) > 1:
    raw = ep.read_raw(sys.argv[1])
    raw.set_montage("standard_1010")
else:
    raw = ep.simulate_eeg(duration=120, montage="32", bad_channels={"FC5": "noisy", "P8": "drift"}, seed=2)

# --- 1. Quality ranking
quality = ep.rank_channels(raw, by="quality")
print(quality.summary(max_rows=10))
quality.plot().savefig(f"{OUT}/rank_quality.png", bbox_inches="tight")

report = raw.assess_quality()
colors = {r["channel"]: GRADE_STATUS[r["grade"]] for r in report.rows}
plot_sensors(raw, colors=colors, legend=GRADE_STATUS, title="Electrode grades").savefig(
    f"{OUT}/sensor_grades.png", bbox_inches="tight")

# --- 2. SNR ranking
snr = ep.rank_channels(raw, by="snr")
print("\nBest SNR:", snr.top(5), "| worst:", snr.bottom(3))

# --- 3. Band activity (bad channels excluded)
raw.set_bads(report.bad_channels)
for band, ranking in band_activity(raw).items():
    print(f"{band:>6}: strongest at {', '.join(ranking.top(3))}")
print("\n" + format_table(dominant_bands(raw, picks=[c for c in raw.pick_names('eeg') if c not in raw.bads])[:10]))
plot_band_topomaps(raw).savefig(f"{OUT}/band_topomaps.png", bbox_inches="tight")
ep.rank_channels(raw, by="alpha").plot().savefig(f"{OUT}/rank_alpha.png", bbox_inches="tight")

# --- 4. Hubs of the alpha network
con = ep.compute_connectivity(raw, method="wpli", band=(8, 13))
print("\nStrongest alpha links:\n" + con.summary(5))
plot_connectivity(con).savefig(f"{OUT}/alpha_wpli.png", bbox_inches="tight")
print(f"\nFigures written to {OUT}/")
