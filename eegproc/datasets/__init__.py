"""Synthetic datasets with known ground truth."""

from .simulate import BAD_KINDS, DEFAULT_BADS, MONTAGES, simulate_epochs_dataset, simulate_eeg

__all__ = ["BAD_KINDS", "DEFAULT_BADS", "MONTAGES", "simulate_eeg", "simulate_epochs_dataset"]
