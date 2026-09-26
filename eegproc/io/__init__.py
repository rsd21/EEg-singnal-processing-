"""Reading and writing EEG files.

:func:`read_raw` picks the reader from the file extension (or content) and
:func:`write_raw` the writer. Native readers need only numpy/scipy; a few
formats use optional packages (h5py, pyxdf, mne).
"""

from __future__ import annotations

import os
from typing import Callable

from ..core.raw import RawEEG
from .arrays import read_hdf5, read_matlab, read_numpy, write_hdf5, write_matlab, write_numpy
from .brainvision import read_brainvision, write_brainvision
from .edf import read_edf, read_edf_header, write_edf
from .mne_bridge import from_mne, read_with_mne, to_mne, write_fif
from .text import read_text, write_text
from .xdf import read_xdf

# extension -> (reader, description, backend)
READERS: dict[str, tuple[Callable[..., RawEEG], str, str]] = {
    ".edf": (read_edf, "European Data Format (EDF/EDF+)", "native"),
    ".bdf": (read_edf, "BioSemi Data Format (BDF/BDF+)", "native"),
    ".rec": (read_edf, "EDF variant (.rec)", "native"),
    ".vhdr": (read_brainvision, "BrainVision header", "native"),
    ".vmrk": (read_brainvision, "BrainVision marker file (reads the matching .vhdr)", "native"),
    ".eeg": (read_brainvision, "BrainVision data file (reads the matching .vhdr)", "native"),
    ".csv": (read_text, "Comma-separated values", "native"),
    ".tsv": (read_text, "Tab-separated values", "native"),
    ".txt": (read_text, "Plain text / OpenBCI export", "native"),
    ".dat": (read_text, "Whitespace-delimited text", "native"),
    ".asc": (read_text, "ASCII text", "native"),
    ".npy": (read_numpy, "NumPy array (sfreq required)", "native"),
    ".npz": (read_numpy, "NumPy archive", "native"),
    ".mat": (read_matlab, "MATLAB (v4-v7.3)", "native (+h5py for v7.3)"),
    ".set": (read_matlab, "EEGLAB dataset", "native"),
    ".h5": (read_hdf5, "HDF5", "h5py"),
    ".hdf5": (read_hdf5, "HDF5", "h5py"),
    ".xdf": (read_xdf, "Lab Streaming Layer XDF", "pyxdf"),
    ".fif": (read_with_mne, "MNE/Neuromag FIF", "mne"),
    ".gdf": (read_with_mne, "General Data Format", "mne"),
    ".cnt": (read_with_mne, "Neuroscan CNT", "mne"),
    ".mff": (read_with_mne, "EGI MFF", "mne"),
    ".egi": (read_with_mne, "EGI simple binary", "mne"),
    ".raw": (read_with_mne, "EGI raw", "mne"),
    ".lay": (read_with_mne, "Persyst", "mne"),
    ".data": (read_with_mne, "Nicolet", "mne"),
    ".nxe": (read_with_mne, "eXimia", "mne"),
    ".snirf": (read_with_mne, "SNIRF", "mne"),
    ".cdt": (read_with_mne, "Curry", "mne"),
    ".dap": (read_with_mne, "Curry", "mne"),
    ".ahdr": (read_with_mne, "BrainVision (alt)", "mne"),
    ".mefd": (read_with_mne, "MEF", "mne"),
}

WRITERS: dict[str, tuple[Callable[..., str], str]] = {
    ".edf": (write_edf, "EDF+ (16 bit)"),
    ".bdf": (lambda raw, path, **kw: write_edf(raw, path, bdf=True, **kw), "BDF+ (24 bit)"),
    ".vhdr": (write_brainvision, "BrainVision (float32)"),
    ".csv": (write_text, "CSV"),
    ".tsv": (write_text, "TSV"),
    ".txt": (write_text, "Text"),
    ".npz": (write_numpy, "NumPy archive"),
    ".mat": (write_matlab, "MATLAB"),
    ".h5": (write_hdf5, "HDF5"),
    ".hdf5": (write_hdf5, "HDF5"),
    ".fif": (write_fif, "FIF (needs mne)"),
}


def _sniff(path: str) -> str | None:
    """Guess the format from the first bytes when the extension is unknown."""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return None
    if head[:8] == b"\xffBIOSEMI":
        return ".bdf"
    if head[:8] == b"0       " and len(head) >= 256:
        return ".edf"
    if head.startswith(b"Brain Vision Data Exchange Header"):
        return ".vhdr"
    if head.startswith(b"\x93NUMPY"):
        return ".npy"
    if head.startswith(b"PK"):
        return ".npz"
    if head.startswith(b"\x89HDF"):
        return ".h5"
    if head.startswith(b"MATLAB"):
        return ".mat"
    if head.startswith(b"XDF:"):
        return ".xdf"
    try:
        head.decode("utf-8")
        return ".csv"
    except UnicodeDecodeError:
        return None


def read_raw(path: str | os.PathLike, fmt: str | None = None, **kwargs) -> RawEEG:
    """Read an EEG recording from any supported file.

    Parameters
    ----------
    path : str
        File to read. For BrainVision pass the ``.vhdr`` (``.eeg``/``.vmrk``
        work too).
    fmt : str, optional
        Force a format by extension (e.g. ``'edf'``, ``'csv'``).
    **kwargs
        Passed to the specific reader, e.g. ``sfreq=250`` for text/NumPy
        files, ``include=[...]`` or ``tmin/tmax`` for EDF.

    Examples
    --------
    >>> raw = read_raw("subject01.edf")                 # doctest: +SKIP
    >>> raw = read_raw("openbci_session.txt")           # doctest: +SKIP
    >>> raw = read_raw("signals.npy", sfreq=500)         # doctest: +SKIP
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.isdir(path) and path.lower().endswith(".mff"):
        ext = ".mff"
    else:
        ext = ("." + fmt.lower().lstrip(".")) if fmt else os.path.splitext(path)[1].lower()
    if ext not in READERS:
        guessed = _sniff(path)
        if guessed is None:
            try:
                return read_with_mne(path, **kwargs)
            except Exception as err:  # noqa: BLE001 - report the original problem
                raise ValueError(f"Unsupported file type {ext!r} for {path}") from err
        ext = guessed
    if ext == ".eeg" and not os.path.exists(os.path.splitext(path)[0] + ".vhdr"):
        return read_with_mne(path, **kwargs)  # Nihon Kohden also uses .eeg
    reader = READERS[ext][0]
    raw = reader(path, **kwargs)
    raw.history.append(f"read {os.path.basename(path)}")
    return raw


def write_raw(raw: RawEEG, path: str | os.PathLike, fmt: str | None = None, **kwargs) -> str:
    """Write a recording; the format follows the extension (or ``fmt``)."""
    path = os.fspath(path)
    ext = ("." + fmt.lower().lstrip(".")) if fmt else os.path.splitext(path)[1].lower()
    if ext not in WRITERS:
        raise ValueError(f"Cannot write {ext!r}; supported: {', '.join(sorted(WRITERS))}")
    if fmt and not path.lower().endswith(ext):
        path = path + ext
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    return WRITERS[ext][0](raw, path, **kwargs)


def supported_formats() -> list[dict]:
    """Table of readable/writable formats and the backend each one uses."""
    rows = []
    for ext, (_, desc, backend) in sorted(READERS.items()):
        rows.append({"extension": ext, "description": desc, "read": backend,
                     "write": "yes" if ext in WRITERS else ""})
    return rows


__all__ = ["read_raw", "write_raw", "supported_formats", "read_edf", "read_edf_header", "write_edf",
           "read_brainvision", "write_brainvision", "read_text", "write_text", "read_numpy",
           "write_numpy", "read_matlab", "write_matlab", "read_hdf5", "write_hdf5", "read_xdf",
           "read_with_mne", "from_mne", "to_mne", "write_fif", "READERS", "WRITERS"]
