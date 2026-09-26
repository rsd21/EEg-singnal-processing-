import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from eegproc import Annotations, RawEEG  # noqa: E402
from eegproc.datasets import simulate_eeg  # noqa: E402

INJECTED = {"T8": "noisy", "O2": "flat", "F7": "drift", "P4": "line", "C3": "pop"}


@pytest.fixture(scope="session")
def injected():
    return dict(INJECTED)


@pytest.fixture(scope="session")
def clean_raw():
    return simulate_eeg(60, 256, "clinical19", seed=0, eog=True)


@pytest.fixture(scope="session")
def bad_raw():
    return simulate_eeg(90, 256, "clinical19", bad_channels=INJECTED, seed=1, eog=True)


@pytest.fixture(scope="session")
def oddball_raw():
    return simulate_eeg(240, 256, "32", task="oddball", seed=2, blinks=False)


@pytest.fixture(scope="session")
def mi_raw():
    from eegproc.datasets import simulate_epochs_dataset

    return simulate_epochs_dataset("motor_imagery", 40, seed=0)


@pytest.fixture
def small_raw():
    rng = np.random.default_rng(0)
    names = ["Fp1", "Fp2", "Cz", "Pz", "O1", "STI"]
    data = rng.normal(0, 20, (6, 256 * 10 + 100))
    data[5] = 0
    data[5, 300:310] = 3
    data[5, 1000:1005] = 7
    ann = Annotations([1.5, 3.25], [0.0, 0.5], ["target", "BAD blink"])
    raw = RawEEG(data, 256, names, ["eeg"] * 5 + ["stim"], annotations=ann)
    raw.set_montage("standard_1020")
    return raw


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
