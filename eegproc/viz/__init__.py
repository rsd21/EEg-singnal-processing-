"""Plotting (matplotlib). Every function returns the Figure; pass ``show=True`` to display it."""

from ._style import CATEGORICAL, CMAP_DIV, CMAP_SEQ, GRADE_STATUS, STATUS, style
from .connectivity import plot_connectivity
from .erp import plot_erds_timecourse, plot_evoked, plot_evoked_topomaps
from .quality import (plot_quality_bars, plot_quality_heatmap, plot_quality_report, plot_quality_topomap,
                      plot_ranking)
from .spectra import plot_band_power, plot_psd, plot_spectrogram, plot_spectrum
from .timeseries import plot_raw
from .topomap import draw_head, plot_band_topomaps, plot_ica_components, plot_sensors, plot_topomap

__all__ = ["CATEGORICAL", "CMAP_DIV", "CMAP_SEQ", "GRADE_STATUS", "STATUS", "style", "plot_connectivity",
           "plot_erds_timecourse", "plot_evoked", "plot_evoked_topomaps", "plot_quality_bars",
           "plot_quality_heatmap", "plot_quality_report", "plot_quality_topomap", "plot_ranking",
           "plot_band_power", "plot_psd", "plot_spectrogram", "plot_spectrum", "plot_raw", "draw_head",
           "plot_band_topomaps", "plot_ica_components", "plot_sensors", "plot_topomap"]
