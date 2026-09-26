"""A standard cleaning pipeline, step by step.

filter -> notch -> detect & interpolate bad channels -> average reference
-> ICA ocular artifact removal -> resample -> save
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import eegproc as ep  # noqa: E402
from eegproc.preprocessing import annotate_artifacts, find_bad_channels  # noqa: E402

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

if len(sys.argv) > 1:
    raw = ep.read_raw(sys.argv[1])
    raw.set_montage("standard_1010")
else:
    raw = ep.simulate_eeg(duration=180, montage="32", bad_channels={"T8": "noisy", "O2": "flat"},
                          eog=True, seed=1)

# Zero-phase filters (Butterworth, applied forwards and backwards)
filtered = raw.filter(l_freq=1.0, h_freq=40.0).notch_filter("auto")   # 'auto' finds 50 or 60 Hz

# Bad channels: PREP criteria + quality score
prep = find_bad_channels(filtered, random_state=0)
print(prep)
quality = filtered.assess_quality()
bads = sorted(set(prep.bads) | set(quality.bad_channels))
print("Bad channels:", bads)
filtered.set_bads(bads)

# Repair them from their neighbours (spherical splines) and re-reference
repaired = filtered.interpolate_bads().set_reference("average")

# ICA: find components that look like eye blinks and remove them
ica = ep.ICA(n_components=0.99, random_state=0).fit(repaired)
eye, scores = ica.find_bads_eog(repaired)
ica.exclude = eye
print(ica.summary(repaired))
ica.plot_components().savefig(f"{OUT}/ica_components.png", bbox_inches="tight")
clean = ica.apply(repaired)

# Mark remaining artifact segments (they are skipped by PSDs and epoching)
clean = annotate_artifacts(clean, amplitude=True, muscle=True)

clean = clean.resample(128)
print(clean)
print("History:", " | ".join(clean.history))

clean.save(f"{OUT}/cleaned.edf")
before = raw.plot(duration=10, quality=quality)
before.savefig(f"{OUT}/before.png", bbox_inches="tight")
clean.plot(duration=10).savefig(f"{OUT}/after.png", bbox_inches="tight")
print(f"Saved {OUT}/cleaned.edf and before/after figures")

# The same thing as a reusable pipeline (see examples/pipeline.yaml)
pipe = ep.Pipeline([{"filter": {"l_freq": 1.0, "h_freq": 40.0}}, {"notch": {"freqs": "auto"}},
                    {"detect_bad_channels": {}}, {"interpolate_bad_channels": {}},
                    {"rereference": {"ref": "average"}}, {"ica": {"n_components": 0.99}},
                    {"resample": {"sfreq": 128}}])
same = pipe.run(raw, verbose=True)
