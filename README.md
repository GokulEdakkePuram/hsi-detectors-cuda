# hsi-detectors-cuda

[![ci](https://github.com/GokulEdakkePuram/hsi-detectors-cuda/actions/workflows/ci.yml/badge.svg)](https://github.com/GokulEdakkePuram/hsi-detectors-cuda/actions/workflows/ci.yml)

Real-time target and anomaly detection for hyperspectral video, written for the
Jetson AGX Orin and validated on discrete NVIDIA GPUs.

Four detectors, chosen because they sit on opposite sides of the roofline:

| detector | what it finds | needs | per-pixel cost | bound by |
|----------|---------------|-------|---------------:|----------|
| **SAM** — spectral angle mapper | known targets | a signature | ~2B FLOP | DRAM |
| **CEM** — constrained energy minimization | known targets | signature + scene stats | ~2B FLOP | DRAM |
| **RX** — Reed-Xiaoli | anomalies, no prior | scene stats | ~B² FLOP | compute |
| **ACE** — adaptive cosine estimator | known targets | signature + scene stats | ~B² FLOP | compute |

At 113 bands the second pair is around 28 FLOP/byte against a ridge point near
38 on an RTX 3090 and near 26 on an AGX Orin, so SAM/CEM are bandwidth problems
and RX/ACE are arithmetic problems. Optimising them pulls in opposite
directions, which is most of what makes this interesting.

SAM here is the spectral angle mapper, not Segment Anything.

Given a stream of hyperspectral frames and a library of target signatures, the
pipeline scores every pixel against every target, thresholds the result and
returns a compacted list of detections — at video rate, inside a fixed frame
budget.

## Why the spectral angle

For a pixel spectrum `x` and a reference spectrum `r`:

```
theta(x, r) = acos( <x, r> / (||x|| * ||r||) )
```

It is the angle between the two spectra, which makes it invariant to a
per-pixel multiplicative gain. That invariance is the entire reason to use it.
Illumination, shadow and viewing geometry scale a spectrum without changing its
shape, so SAM keeps matching a target through a passing cloud or into shade,
where a Euclidean distance would lose it. Reflectance is non-negative, so
angles lie in `[0, pi/2]` and smaller is a better match.

## Layout

```
include/hsi/     public headers
src/             core (host) and CUDA kernels, pipeline, CLI
bench/           kernel-level correctness and throughput harness
tests/           host-side test suite, no framework dependency
scripts/         dataset fetch, build, profiling
docs/            design notes and measured results
```

## Build

CUDA is optional. Without a toolkit the host-side library and its tests still
build, which is what makes the ENVI reader, the demosaicer and the CPU
reference developable on a laptop.

```sh
scripts/build.sh          # fat binary, Turing through Hopper, plus sm_87
scripts/build.sh 87       # AGX Orin only, much faster to compile
cd build && ctest         # host-side tests; adds a GPU check if CUDA was found
```

## Run

```sh
# Synthetic video, no data needed - the generator supplies its own signatures.
build/src/hsi_detect --frames=300

# Real cubes. prepare_hyperblood.py writes cleaned 113-band cubes and a
# matching library; the raw 128-band cubes will not line up with it.
scripts/fetch_hyperblood.sh
python3 scripts/prepare_hyperblood.py
build/src/hsi_detect --source=envi \
  --envi=data/hyperblood_prepared/F_1.hdr \
  --library=data/hyperblood_targets.csv --threshold=0.10

# CPU reference scorer - no GPU needed, and what a GPU run gets diffed against.
build/src/hsi_score --cube=data/hyperblood_prepared/F_1.hdr \
  --library=data/hyperblood_targets.csv --target=blood --dump-score=/tmp/F_1.f32

# Snapshot-mosaic hyperspectral video (HOT-style).
build/src/hsi_detect --source=hot --hot-dir=<frames/> --mosaic=4 --bands=16 \
  --library=<targets.csv>

build/src/hsi_detect --help
```

On the Orin, add `--memory=zerocopy`. On a discrete GPU, do not — see below.

## The kernels

SAM moves `bands * 4` bytes per pixel against roughly two FLOPs per band per
target. At one target that is 0.5 FLOP/byte, which is far below the ridge point
of every GPU this runs on: the kernel is bound by DRAM bandwidth, not by
arithmetic. Every variant is therefore an experiment in moving fewer bytes.

| variant     | what it does                                                      |
|-------------|-------------------------------------------------------------------|
| `baseline`  | one thread per pixel, scalar loads, targets from global memory     |
| `optimized` | `float4` loads, targets broadcast from constant memory, up to eight targets scored per read of the cube |
| `half`      | as above with the cube in fp16, accumulating in fp32               |
| `bip`       | the baseline over a BIP cube, to measure the coalescing penalty    |

Three decisions carry most of the performance:

**BSQ, not BIP.** With one thread per pixel and a band-sequential cube,
neighbouring threads read neighbouring addresses on every band step and the
loads coalesce. The same kernel over a band-interleaved-by-pixel cube reads at
a stride of `bands`. `bip` exists so that cost is a measured number rather than
an assertion, and a device-side tiled transpose is provided for sensors that
deliver BIP.

**Targets in constant memory.** On any band step every thread in a warp wants
the same target element. That is the broadcast case the constant bank serves at
register speed.

**One `acosf` per pixel, not per target.** Candidates are ranked on cosine,
which is monotonically decreasing in the angle, and converted once at the end.

## Zero-copy on Jetson

The AGX Orin's CPU and GPU share one physical LPDDR5 pool. An H2D copy there
moves bytes from memory to the same memory — at ~160 MB per frame, the largest
single cost in the pipeline and pure waste. `--memory=zerocopy` allocates the
staging buffer pinned and device-mapped and hands the kernels a pointer into
it, so the copy disappears.

The same flag on a discrete GPU is a trap: every kernel read becomes a PCIe
transaction. `hsi_detect` prints whether the device is integrated for exactly
this reason.

Frame sources take a plane stride to write at, so the staging buffer is already
in the padded layout the kernels need for aligned vector loads. Without that,
zero-copy would need a repacking pass and would stop being zero-copy.

## Data

**[HyperBlood](https://doi.org/10.5281/zenodo.3984905)** (CC-BY-4.0) — 14
hyperspectral cubes of blood beside substances chosen to look like it:
artificial blood, tomato concentrate, beetroot juice, poster paint. 113 usable
bands, per-pixel ground truth. It is the right dataset here precisely because
colour cannot separate those classes and the spectral angle can, and because
the annotations make detection quality measurable rather than eyeballed.
`scripts/fetch_hyperblood.sh` downloads it.

**HOT-style snapshot mosaic video** — the hyperspectral object tracking
benchmarks (XIMEA SSM 4×4 VIS, 16 bands over 470–620 nm, 25 fps) are the best
public hyperspectral *video*, but the archive is behind a request form at
[hsitracking.com](https://www.hsitracking.com/). The loader is written from the
published description and tested against synthetic mosaics of the same
geometry — not against the real files. Check that the first frame's band images
look like a scene before trusting it, and set `--mosaic`/`--bands` to match the
set (4×4/16 for VIS, 4×4/15 for RedNIR, 5×5/25 for NIR).

**Synthetic** — a mixed background with target blobs moving through it, at any
resolution and band count. It exists because the public datasets that ship with
ground truth are single scenes rather than video, and because throughput has to
be measured at sensor geometries no dataset happens to provide. It reports the
mask it stamped, which makes it self-validating.

## Does it work

### Speed — RTX 3090, 512×512×128, 4 targets

Median of five passes after the clock settles; spread under 3%.

| variant | ms | % of peak bandwidth | vs baseline |
|---------|---:|--------------------:|------------:|
| `baseline`  | 0.665 | 86% | 1.00× |
| `optimized` | 0.174 | 82% | 3.82× |
| `half`      | 0.098 | 73% | 6.77× |
| `bip`       | 1.820 | 32% | 0.37× |

Three findings worth knowing before reading the code.

**The baseline was already at the roofline** — 86% of peak. At *one* target the
optimized kernel beats it by 1.04×, so `float4` loads and constant-memory
broadcast together buy 4%. The entire ~4× win is amortising the cube read
across targets. Coalescing is what matters; vectorising already-coalesced
access does almost nothing.

**Getting the layout wrong costs 2.7×** — `bip` runs at 32% of peak and worse
as the spectrum spans more cache lines (14% at 224 bands). That is the one
mistake that is expensive.

**Constant-memory reads are the second wall.** Scoring stays memory-bound at
84–85% of peak up to four targets, then drops to ~60% at eight, because the
inner loop issues one constant read per target per vector load. The GPU says so
itself: the SM clock *rises* from 765 MHz to 1440 MHz as bandwidth falls, which
is what a kernel that has stopped waiting on DRAM looks like.

End to end on a discrete GPU the pipeline is PCIe-bound and the kernel is
invisible: 35 ms of upload against 0.18 ms of SAM, ~83 fps regardless of
variant. That is the configuration this was *not* built for — the Orin's
unified memory is the point, and is still unmeasured.

### Accuracy

On HyperBlood's `F_1` — blood beside ketchup, artificial blood, beetroot juice,
poster paint, tomato concentrate and acrylic paint, all red — with the target
signature taken from that scene:

| | |
|---|---|
| ROC AUC (blood vs rest) | 0.9945 |
| precision @ 0.10 rad | 1.0000 — **0 false positives in 320 000 negatives** |
| recall @ 0.10 rad | 0.7073 |

Blood averages 0.0916 rad from the signature; the closest single pixel of the
nearest confuser, beetroot juice, sits at 0.1623. That margin is the whole
argument for using the spectral angle.

Across all 14 scenes, mean AUC is 0.879 with an on-scene signature and 0.798
with one global signature. The split that matters is not the signature but the
scene: the controlled-background "frame" images reach 0.98–0.999 AUC at ~0.9
recall, while the cluttered mock-up scenes sit at 0.70–0.81, because SAM has no
spatial context to fall back on when the background overlaps the target in
spectral direction.

The operational caveat, measured rather than assumed: **ranking transfers
between scenes, the threshold does not.** Precision at a fixed 0.10 rad swings
from 1.00 to 0.004 across scenes with the same library. A deployed detector
needs its threshold set per scene, or derived from the frame's own angle
distribution. Full tables in [docs/results.md](docs/results.md).

## Profiling

```sh
scripts/profile_gpu.sh      # discrete GPU
scripts/profile_jetson.sh   # AGX Orin
```

Both check every kernel against the double-precision CPU reference before
reporting any timing, and both pin clocks first where they can — an unpinned
run measures the governor rather than the code. The Jetson script additionally
sets MAXN, logs tegrastats alongside, and prints the peak temperatures reached,
because a thermal ceiling hit mid-run invalidates every number after it.

Measured results: [docs/results.md](docs/results.md).
Design notes and the performance model: [docs/design.md](docs/design.md).

## License

The code is MIT — see [LICENSE](LICENSE).

The datasets are not redistributed here; the scripts download them from the
original sources, and their own terms apply:

* **HyperBlood** — CC-BY-4.0. Romaszewski, Głomb, Cholewa and Sochan, Institute
  of Theoretical and Applied Informatics, Polish Academy of Sciences.
  [doi:10.5281/zenodo.3984905](https://doi.org/10.5281/zenodo.3984905)
* **HOT** — obtained separately under the terms at
  [hsitracking.com](https://www.hsitracking.com/). No HOT data is included.
