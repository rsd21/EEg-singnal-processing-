"""Synthetic datasets with known ground truth, and downloads of public recordings."""

from .openneuro import data_dir, fetch_openneuro, fetch_openneuro_recording, list_openneuro_files
from .simulate import BAD_KINDS, DEFAULT_BADS, MONTAGES, simulate_epochs_dataset, simulate_eeg

__all__ = ["BAD_KINDS", "DEFAULT_BADS", "MONTAGES", "simulate_eeg", "simulate_epochs_dataset",
           "data_dir", "fetch_openneuro", "fetch_openneuro_recording", "list_openneuro_files"]
