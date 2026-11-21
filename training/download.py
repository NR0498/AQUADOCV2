"""Download the external training inputs with pinned checksums.

    python -m training.download

Files land in data/external/ (git-ignored). Re-running skips files whose SHA-256
already matches; a mismatch stops with an error rather than training on changed data.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

# GLORIA: Lehmann et al. (2023) Sci. Data 10:100; data DOI 10.1594/PANGAEA.948492, CC-BY-4.0.
GLORIA_URL = "https://download.pangaea.de/dataset/948492/files/GLORIA-2022.zip"
GLORIA_SHA256 = "f5ad5106d5fd74f0d874085a23e4189302e7409aed9f82d629ae6cdb47d21f14"

# Sentinel-2 spectral response functions, ESA COPE-GSEG-EOPG-TN-15-0007 v4.0 (2024),
# as text files mirrored by the ACOLITE project.
SRF_BASE = "https://raw.githubusercontent.com/acolite/acolite/main/data/RSR"
SRF_FILES = {
    "S2A_MSI_V4.txt": "d8f2039e14b78c12e8ca647b3262f68d03a1fcf8be13ffb01d9bc4a045fce182",
    "S2B_MSI_V4.txt": "5a9558c1f5138801790a960cce08d774661799ffc299272cc01b2706bd4c467c",
    "S2C_MSI_V4.txt": "c9b3b99e29a54dd7d42c40dd478ab901454bdb84fc011715fd0f21b5a6a75f42",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == expected_sha256:
        print(f"ok      {destination.relative_to(PROJECT_ROOT)}")
        return destination
    print(f"fetch   {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "aquadoc-training/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    actual = sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink()
        raise RuntimeError(f"Checksum mismatch for {url}: expected {expected_sha256}, got {actual}")
    temporary.replace(destination)
    return destination


def download_all(external_dir: Path = EXTERNAL_DIR) -> None:
    archive = fetch(GLORIA_URL, external_dir / "gloria" / "GLORIA-2022.zip", GLORIA_SHA256)
    extracted = external_dir / "gloria" / "extracted"
    if not (extracted / "GLORIA_2022" / "GLORIA_Rrs.csv").exists():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
        print(f"extract {extracted.relative_to(PROJECT_ROOT)}")
    for name, digest in SRF_FILES.items():
        fetch(f"{SRF_BASE}/{name}", external_dir / "s2_srf" / name, digest)


if __name__ == "__main__":
    try:
        download_all()
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
