"""Render analysis rasters as small PNG images for the dashboard.

A PNG is a signature plus chunks (IHDR size/format, IDAT zlib-compressed scanlines,
IEND), each with a CRC. Writing it directly with zlib avoids an imaging dependency
in the deployed bundle.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

# Perceptually uniform colour ramps (anchor points sampled from matplotlib's maps).
COLORMAPS: dict[str, list[tuple[float, float, float]]] = {
    "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "algae": [(215, 249, 208), (142, 204, 129), (64, 159, 64), (25, 112, 42), (17, 54, 25)],
    "sediment": [(253, 237, 176), (241, 184, 109), (219, 128, 70), (167, 79, 45), (95, 40, 20)],
    "cdom": [(255, 246, 216), (226, 196, 132), (175, 134, 70), (115, 80, 39), (60, 40, 20)],
    "clarity": [(25, 39, 70), (37, 82, 140), (47, 129, 184), (109, 190, 215), (210, 240, 245)],
    "debris": [(240, 240, 240), (252, 187, 161), (251, 106, 74), (203, 24, 29), (103, 0, 13)],
}


def encode_png(rgba: np.ndarray) -> bytes:
    """Encode an (h, w, 4) uint8 array as PNG bytes."""
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError("encode_png expects an (h, w, 4) uint8 array")
    height, width = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[row].tobytes() for row in range(height))  # filter type 0 per row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def _ramp(values: np.ndarray, name: str) -> np.ndarray:
    stops = np.asarray(COLORMAPS[name], dtype=float)
    positions = np.linspace(0, 1, len(stops))
    v = np.clip(values, 0, 1)
    return np.stack([np.interp(v, positions, stops[:, channel]) for channel in range(3)], axis=-1)


def colorize(values: np.ndarray, mask: np.ndarray, vmin: float, vmax: float, colormap: str = "viridis",
             log: bool = False) -> bytes:
    """Map ``values`` to colours where ``mask`` is True; transparent elsewhere."""
    data = np.asarray(values, dtype=float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            data, vmin, vmax = np.log10(data), np.log10(vmin), np.log10(vmax)
    scaled = (data - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(data)
    rgb = _ramp(np.nan_to_num(scaled), colormap)
    alpha = np.where(mask & np.isfinite(data), 255, 0)
    return encode_png(np.dstack([rgb, alpha]).astype(np.uint8))


def true_color(red: np.ndarray, green: np.ndarray, blue: np.ndarray, gain: float = 3.5) -> bytes:
    """Natural-colour image from surface reflectance with a simple gain + gamma stretch."""
    rgb = np.dstack([red, green, blue]).astype(float)
    valid = np.all(np.isfinite(rgb), axis=2)
    stretched = np.clip(np.nan_to_num(rgb) * gain, 0, 1) ** (1 / 1.6)
    alpha = np.where(valid, 255, 0)
    return encode_png(np.dstack([stretched * 255, alpha]).astype(np.uint8))


def class_map(layers: list[tuple[np.ndarray, tuple[int, int, int, int]]], shape: tuple[int, int]) -> bytes:
    """Paint boolean layers in order (later layers on top) with RGBA colours."""
    image = np.zeros((*shape, 4), dtype=np.uint8)
    for mask, colour in layers:
        image[mask] = colour
    return encode_png(image)
