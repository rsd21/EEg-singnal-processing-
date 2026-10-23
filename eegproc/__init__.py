"""eegproc: EEG signal processing from file to "which channel performs best".

Quick start::

    import eegproc as ep

    raw = ep.read_raw("recording.edf")          # any supported format
    raw.set_montage("standard_1010")             # electrode positions
    print(raw.describe())                        # channels, rate, events

    quality = raw.assess_quality()               # 0-100 score per channel
    print(quality.summary())
    quality.plot()                               # dashboard figure

    clean = (raw.filter(1.0, 40.0)
                .notch_filter(50)
                .set_bads(quality.bad_channels)  # in place, returns raw
                .interpolate_bads()
                .set_reference("average"))

    ep.generate_report(clean, "report.html")     # everything in one HTML file
"""

__version__ = "0.1.0"

from . import analysis, datasets, io, preprocessing, quality, viz  # noqa: E402
from .analysis import (BANDS, Spectrum, compute_connectivity, compute_psd, extract_features,  # noqa: E402
                       morlet_power)
from .core import (Annotations, Epochs, Evoked, Montage, RawEEG, find_events,  # noqa: E402
                   make_fixed_length_epochs, make_standard_montage)
from .datasets import simulate_eeg  # noqa: E402
from .io import read_raw, supported_formats, write_raw  # noqa: E402
from .pipeline import Pipeline  # noqa: E402
from .preprocessing import (ICA, filter_data, find_bad_channels, interpolate_bads,  # noqa: E402
                            remove_artifacts_ica)
from .quality import ChannelRanking, assess_channel_quality, rank_channels  # noqa: E402
from .report import generate_report  # noqa: E402
from .utils import set_log_level  # noqa: E402

__all__ = ["__version__", "analysis", "datasets", "io", "preprocessing", "quality", "viz", "BANDS", "Spectrum",
           "compute_connectivity", "compute_psd", "extract_features", "morlet_power", "Annotations", "Epochs",
           "Evoked", "Montage", "RawEEG", "find_events", "make_fixed_length_epochs", "make_standard_montage",
           "simulate_eeg", "read_raw", "supported_formats", "write_raw", "Pipeline", "ICA", "filter_data",
           "find_bad_channels", "interpolate_bads", "remove_artifacts_ica", "ChannelRanking",
           "assess_channel_quality", "rank_channels", "generate_report", "set_log_level"]
