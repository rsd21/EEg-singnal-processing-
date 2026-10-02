"""Event-related potentials: which channel shows the P300 best?

Uses an auditory oddball recording (rare "target" tones among "standard"
ones). With your own file, pass it and the names of two event types.

    python examples/04_erp_oddball.py
    python examples/04_erp_oddball.py my_file.vhdr "Stimulus/S  1" "Stimulus/S  2"
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import eegproc as ep  # noqa: E402
from eegproc.quality import decoding_accuracy, discriminability, erp_snr  # noqa: E402

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

if len(sys.argv) > 1:
    raw = ep.read_raw(sys.argv[1])
    raw.set_montage("standard_1010")
    conditions = sys.argv[2:4] if len(sys.argv) >= 4 else None
else:
    raw = ep.simulate_eeg(duration=300, montage="32", task="oddball", eog=True, seed=3)
    conditions = ["standard", "target"]

# Clean: band-pass for ERPs, remove blinks with ICA
raw = raw.filter(0.5, 30.0)
raw, ica = ep.remove_artifacts_ica(raw)
print("ICA removed components:", ica.exclude)

events, event_id = raw.find_events()
print("Events:", event_id, "| count:", len(events))
epochs = ep.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=(None, 0), reject=150.0)
if conditions:
    epochs = epochs[conditions]
print(epochs)

target = conditions[-1] if conditions else list(event_id)[-1]
evoked = epochs[target].average()
ch, latency, amplitude = evoked.get_peak(tmin=0.25, tmax=0.5, mode="pos")
print(f"Largest positive peak 250-500 ms: {ch} at {1000 * latency:.0f} ms, {amplitude:.1f} µV")
evoked.plot(picks=[ch]).savefig(f"{OUT}/erp_{target}.png", bbox_inches="tight")
evoked.plot_topomap(times=[0.1, 0.2, 0.35, 0.5]).savefig(f"{OUT}/erp_topomaps.png", bbox_inches="tight")

snr = erp_snr(epochs, condition=target)
print("\n" + snr.summary(max_rows=8))
snr.plot().savefig(f"{OUT}/rank_erp_snr.png", bbox_inches="tight")

if conditions and len(conditions) == 2:
    disc = discriminability(epochs, feature="erp", window=(0.1, 0.6))
    dec = decoding_accuracy(epochs, feature="erp", window=(0.1, 0.6))
    print("\n" + disc.summary(max_rows=5))
    print("\n" + dec.summary(max_rows=5))
    dec.plot().savefig(f"{OUT}/rank_erp_decoding.png", bbox_inches="tight")
