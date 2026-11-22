# Vendored native library: libexpat

`libexpat.so.1` is the XML parser GDAL (bundled inside the rasterio wheel) links against. The rasterio
Linux wheels expect it from the operating system, but some Vercel production runtime instances do not
have it, which made every analysis fail with `ImportError: libexpat.so.1: cannot open shared object file`.

`aquadoc_core/native.py` loads this copy only on Linux, and only when the system has none.

| | |
| --- | --- |
| Source | Debian 11 package `libexpat1_2.2.10-2+deb11u5_amd64.deb` (security-patched) |
| Package SHA256 | `5744040c4735bcdd51238aebfa3e402b857244897f1007f61154982ebe5abbd7` (matches Debian's bullseye `Packages` index) |
| Library SHA256 | `1a3ace30e2c877e3dc13de57cd0ecc8327acf0f15e6d5ec3608f4b36873687b6` (`libexpat.so.1.6.12`, SONAME `libexpat.so.1`) |
| Requires | glibc ≥ 2.25 (Vercel runs Amazon Linux 2023, glibc 2.34) |
| Licence | MIT, see `EXPAT_COPYRIGHT.txt` |

Debian 12's build was not used: it requires glibc 2.36.
