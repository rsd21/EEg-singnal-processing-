"""Download recordings from OpenNeuro, a free archive of public (mostly CC0) EEG datasets.

Browse datasets at https://openneuro.org (filter by modality "EEG"); each has an
ID such as ``ds004362``. Files are read from OpenNeuro's public S3 bucket and
cached, so a file is only downloaded once.
"""

from __future__ import annotations

import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

from ..utils import logger

BUCKET = "https://s3.amazonaws.com/openneuro.org"


def data_dir(path: str | os.PathLike | None = None) -> Path:
    """Cache folder: ``path``, else ``$EEGPROC_DATA``, else ``~/eegproc_data``."""
    return Path(path or os.environ.get("EEGPROC_DATA") or Path.home() / "eegproc_data")


def list_openneuro_files(dataset: str, prefix: str = "") -> list[tuple[str, int]]:
    """``(path, size_bytes)`` of the files in a dataset whose path starts with ``prefix``.

    Example: ``list_openneuro_files("ds004362", "sub-001/eeg/")``.
    """
    files, token = [], None
    while True:
        query = {"list-type": "2", "prefix": f"{dataset}/{prefix}"}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(query)}", timeout=60) as resp:
            xml = resp.read().decode()
        keys = re.findall(r"<Key>(.*?)</Key>", xml)
        sizes = re.findall(r"<Size>(\d+)</Size>", xml)
        files += [(k[len(dataset) + 1:], int(s)) for k, s in zip(keys, sizes)]
        match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not match:
            return files
        token = match.group(1)


def fetch_openneuro(dataset: str, files: str | Iterable[str], path: str | os.PathLike | None = None) -> list[Path]:
    """Download dataset files (paths relative to the dataset root) and return the local paths.

    Example: ``fetch_openneuro("ds004362", "sub-001/eeg/sub-001_task-motion_run-1_eeg.set")``.
    Already downloaded files are reused.
    """
    files = [files] if isinstance(files, str) else list(files)
    root = data_dir(path) / dataset
    out = []
    for rel in files:
        target = root / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"{BUCKET}/{dataset}/{urllib.parse.quote(rel)}"
            logger.info("Downloading %s", url)
            partial = target.with_name(target.name + ".part")
            with urllib.request.urlopen(url, timeout=120) as resp, open(partial, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            partial.replace(target)
        out.append(target)
    return out


def fetch_openneuro_recording(dataset: str, file: str, path: str | os.PathLike | None = None) -> Path:
    """Download one recording with the files it needs and return the path to pass to ``read_raw``.

    Fetches the BrainVision ``.vmrk``/``.eeg`` next to a ``.vhdr``, the ``.fdt`` of a
    two-file EEGLAB ``.set`` (if there is one) and the BIDS ``*_channels.tsv``
    that gives channel types and bad channels.
    Example: ``read_raw(fetch_openneuro_recording("ds004362", "sub-001/eeg/sub-001_task-motion_run-1_eeg.set"))``.
    """
    main = fetch_openneuro(dataset, file, path)[0]
    stem, ext = file.rsplit(".", 1)
    required = [f"{stem}.vmrk", f"{stem}.eeg"] if ext.lower() == "vhdr" else []
    optional = [f"{stem}.fdt"] if ext.lower() == "set" else []
    if stem.endswith("_eeg"):
        optional.append(stem[:-4] + "_channels.tsv")
    fetch_openneuro(dataset, required, path)
    for rel in optional:
        try:
            fetch_openneuro(dataset, rel, path)
        except urllib.error.HTTPError as err:
            if err.code not in (403, 404):  # S3 answers 403 for missing keys in a public bucket
                raise
    return main
