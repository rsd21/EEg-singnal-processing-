"""OpenNeuro download helper, against a fake bucket (no network)."""

import io

import eegproc.datasets.openneuro as on


class FakeBucket:
    def __init__(self):
        self.calls = []

    def __call__(self, url, timeout=None):
        self.calls.append(url)
        if "list-type=2" in url and "continuation-token" not in url:
            body = ("<ListBucketResult><Contents><Key>ds1/sub-01/eeg/a.set</Key><Size>10</Size></Contents>"
                    "<IsTruncated>true</IsTruncated><NextContinuationToken>tok</NextContinuationToken>"
                    "</ListBucketResult>")
        elif "list-type=2" in url:
            body = "<ListBucketResult><Contents><Key>ds1/sub-01/eeg/b.tsv</Key><Size>3</Size></Contents></ListBucketResult>"
        else:
            body = "file:" + url.rsplit("/", 1)[-1]
        return io.BytesIO(body.encode())


def test_list_follows_pagination(monkeypatch):
    fake = FakeBucket()
    monkeypatch.setattr(on.urllib.request, "urlopen", fake)
    assert on.list_openneuro_files("ds1", "sub-01/") == [("sub-01/eeg/a.set", 10), ("sub-01/eeg/b.tsv", 3)]
    assert len(fake.calls) == 2


def test_fetch_downloads_once(monkeypatch, tmp_path):
    fake = FakeBucket()
    monkeypatch.setattr(on.urllib.request, "urlopen", fake)
    monkeypatch.setenv("EEGPROC_DATA", str(tmp_path))
    first = on.fetch_openneuro("ds1", "sub-01/eeg/a.set")
    again = on.fetch_openneuro("ds1", ["sub-01/eeg/a.set"])
    assert first == again == [tmp_path / "ds1" / "sub-01" / "eeg" / "a.set"]
    assert first[0].read_text() == "file:a.set" and len(fake.calls) == 1
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_recording_gets_companion_files(monkeypatch, tmp_path):
    import urllib.error

    def bucket(url, timeout=None):
        if url.endswith(".fdt"):
            raise urllib.error.HTTPError(url, 403, "missing", None, None)
        return io.BytesIO(b"x")

    monkeypatch.setattr(on.urllib.request, "urlopen", bucket)
    main = on.fetch_openneuro_recording("ds1", "sub-01/eeg/sub-01_task-x_eeg.set", path=tmp_path)
    assert main.name == "sub-01_task-x_eeg.set"
    assert (main.parent / "sub-01_task-x_channels.tsv").exists() and not (main.parent / "sub-01_task-x_eeg.fdt").exists()
    vhdr = on.fetch_openneuro_recording("ds1", "sub-01/eeg/sub-01_task-y_eeg.vhdr", path=tmp_path)
    assert {p.suffix for p in vhdr.parent.glob("sub-01_task-y_eeg.*")} == {".vhdr", ".vmrk", ".eeg"}
