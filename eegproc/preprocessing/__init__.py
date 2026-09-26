"""Preprocessing: filtering, referencing, resampling, bad channels, artifacts, ICA."""

from .artifacts import (annotate_amplitude, annotate_artifacts, annotate_muscle, bad_time_fraction,
                        find_blinks)
from .bad_channels import BadChannelResult, find_bad_channels, mark_bad_channels
from .filters import (design_fir, design_iir, detect_line_noise, filter_data, filter_raw,
                      notch_filter_data, notch_raw)
from .ica import ICA, fastica, remove_artifacts_ica
from .interpolation import interpolate_bads
from .reference import (DOUBLE_BANANA, TRANSVERSE, add_reference_channel, bipolar_reference,
                        laplacian, set_reference)
from .resample import resample_data, resample_raw

__all__ = ["annotate_amplitude", "annotate_artifacts", "annotate_muscle", "bad_time_fraction",
           "find_blinks", "BadChannelResult", "find_bad_channels", "mark_bad_channels",
           "design_fir", "design_iir", "detect_line_noise", "filter_data", "filter_raw",
           "notch_filter_data", "notch_raw", "ICA", "fastica", "remove_artifacts_ica",
           "interpolate_bads", "DOUBLE_BANANA", "TRANSVERSE", "add_reference_channel",
           "bipolar_reference", "laplacian", "set_reference", "resample_data", "resample_raw"]
