"""The Kaggle upload parts must re-join into a byte-identical zip (the notebook relies on it)."""
import hashlib

from flymatch.package import split


def test_split_rejoins_identically(tmp_path):
    z = tmp_path / "flymatch_kaggle.zip"
    z.write_bytes(bytes(range(256)) * 9000)                  # ~2.2 MB
    dest = split(z, 1)
    parts = sorted(dest.glob("flymatch_kaggle.zip.part*"))
    assert [p.name[-2:] for p in parts] == ["01", "02", "03"]
    assert all(p.stat().st_size <= 2**20 for p in parts)
    joined = b"".join(p.read_bytes() for p in parts)
    assert joined == z.read_bytes()
    expected = (dest / "flymatch_kaggle.zip.sha256").read_text().split()[0]
    assert hashlib.sha256(joined).hexdigest() == expected


def test_split_replaces_old_parts(tmp_path):
    z = tmp_path / "flymatch_kaggle.zip"
    z.write_bytes(b"x" * (3 * 2**20))
    split(z, 1)
    z.write_bytes(b"y" * 10)
    dest = split(z, 1)
    assert [p.name for p in sorted(dest.glob("*.part*"))] == ["flymatch_kaggle.zip.part01"]
