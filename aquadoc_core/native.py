"""Native-library shims for serverless Linux runtimes.

GDAL inside the rasterio wheel needs ``libexpat.so.1`` from the operating system. Some Vercel runtime
instances lack it, so a vetted copy is shipped in ``vendor/`` (see ``vendor/README.md``) and loaded
globally before rasterio is imported. The dynamic loader then finds a library with the SONAME
``libexpat.so.1`` already loaded and uses it to satisfy GDAL's dependency.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

VENDORED_EXPAT = Path(__file__).with_name("vendor") / "libexpat.so.1"


def ensure_libexpat() -> str:
    """Return "system", "vendored" or "not-needed"; raises OSError if Linux has neither."""
    if not sys.platform.startswith("linux"):
        return "not-needed"  # Windows/macOS wheels bundle everything
    try:
        ctypes.CDLL("libexpat.so.1", mode=ctypes.RTLD_GLOBAL)
        return "system"
    except OSError:
        ctypes.CDLL(str(VENDORED_EXPAT), mode=ctypes.RTLD_GLOBAL)
        return "vendored"
