# GlyphMatics Hardware Report

Status: **VERIFIED** for directly readable fields; permission-limited fields are **BLOCKED**.  
Observed: 2026-11-12/13, America/Chicago, directly inside Termux on the target phone.

## Identity

- Product/model: `SM-A146U`
- Device/board: `a14xm`
- SoC: MediaTek `MT6833V/NZA` (`ro.soc.model=MT6833`)
- Android: 15, API 35
- ABI: `arm64-v8a, armeabi-v7a, armeabi`
- Kernel: `6.6.50-android15-8-maybe-dirty-4k`, aarch64
- Build fingerprint: `samsung/a14xmsq/a14xm:15/AP3A.240905.015.A2/A146USQUBEYF5:user/release-keys`

## CPU and memory

- 8 online cores: 6 Cortex-A55 (`0-5`), 2 Cortex-A76 (`6-7`)
- A55 range: 500 MHz–2.0 GHz
- A76 range: 725 MHz–2.203 GHz
- Features: ASIMD/NEON, FP16, dot product, AES, SHA1/SHA2, CRC32, atomics
- Physical RAM: 3,688,308 kB
- Swap/zram: 4,194,300 kB
- Initial observed `MemAvailable`: 1,093,844 kB
- Internal data volume: 48 GB, 42 GB available before v4 artifacts

## Environment

- Termux: 0.119.0-beta.3, F-Droid, aarch64
- Clang: 21.1.8 (`aarch64-unknown-linux-android24`)
- CMake: 4.4.5
- Python: 3.14.10
- Git: 2.56.1
- Packaged llama.cpp: 0.9.0, commit unknown
- Source build: llama.cpp `5676dfbe01e7ebe0387913680eee0ed39bf4a363`

## Blocked fields

- **BLOCKED:** Android thermal service requires `android.permission.DUMP`.
- **BLOCKED:** battery service/sysfs access is denied to the Termux UID.
- **BLOCKED:** CPU governor names and policy maximum writes are permission denied; no settings were changed.
- **BLOCKED:** `/storage` and `/mnt/media_rw` directory enumeration is denied, so microSD presence is unverified.
- Android thermal, memory, firmware, and security protections were not modified.
