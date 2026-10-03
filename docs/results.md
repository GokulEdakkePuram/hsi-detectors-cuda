# Results

Kernel and pipeline figures measured on an **RTX 3090** (sm_86, 82 SMs,
936.1 GB/s peak, CUDA 13.0, driver 580.142) in a rented container. Detection
quality measured on the CPU reference against real HyperBlood data.

Every kernel figure is the median of five timed passes taken after the clock
has settled, with the peak-to-peak spread and the SM clock under load beside
it. That instrumentation matters more than it sounds — see
[Reading these numbers](#reading-these-numbers).

**Not measured on the Orin.** Everything about zero-copy and unified memory is
still a prediction.

## What to run

```sh
scripts/build.sh 86              # one arch; the default builds seven
scripts/profile_gpu.sh           # discrete GPU
scripts/profile_jetson.sh        # AGX Orin
```

Correctness runs first and gates everything after it.

## Reading these numbers

The card is capped at **220 W of a 350 W board**. Under load its SM clock
collapses from 1695 MHz to as little as 285 MHz while the memory clock never
moves off 9501 MHz. Two consequences run through everything below.

**The SM clock column is a diagnostic, not a footnote.** Across every
measurement taken, clock and achieved bandwidth are *inversely* correlated:

| SM clock under load | achieved bandwidth | what it means |
|---------------------|-------------------|---------------|
| 285–870 MHz  | 80–86% of peak | memory-bound; the GPU spends its power budget on DRAM |
| 1300–1860 MHz | 14–73% of peak | SM-bound; DRAM is idle enough that clock goes to the SM instead |

A kernel running at 1800 MHz and 26% of peak is not fast, it is stalled.

**Percent of peak can exceed 100%.** It divides useful bytes by DRAM peak, and
some of those bytes come from L2 rather than DRAM. See the 16-band row.

## Correctness

### GPU vs the CPU reference

| variant | 96×128×24, 4 targets | 520×696×113, 8 targets | verdict |
|---------|---------------------:|-----------------------:|---------|
| `baseline`  | 1.80e-05 | 5.80e-05 | PASS |
| `optimized` | 1.80e-05 | 5.80e-05 | PASS |
| `half`      | 1.72e-04 | 1.27e-04 | PASS |
| `bip`       | 1.80e-05 | 5.80e-05 | PASS |

Tolerance is 2e-3 rad for the fp32 paths and 5e-3 for fp16. Zero significant
label mismatches anywhere. The fp32 paths land ~35× inside tolerance and fp16
~40× inside its own, so the thresholds are not doing any work — the kernels
simply agree with double precision. Accumulating in fp32 while storing in fp16
costs about 1e-4 rad, three orders of magnitude below any usable detection
threshold.

### Host side

Cross-checked against an independent numpy implementation on real HyperBlood
cubes:

| check | result |
|-------|--------|
| BIL→BSQ conversion vs numpy    | bit-exact (max diff 0.0) |
| angle map vs float64 numpy SAM | max 6.5e-7 rad, mean 1.1e-7 rad |
| winning target index           | 0 disagreements in 361 224 pixels |

## Kernel throughput

512×512×128 (134.2 MB), 4 targets:

| variant | ms | GB/s | % of peak | vs baseline | spread | SM MHz |
|---------|---:|-----:|----------:|------------:|-------:|-------:|
| `baseline`  | 0.665 | 807.1 | 86% | 1.00× | 5.5% |  945 |
| `optimized` | 0.174 | 771.4 | 82% | **3.82×** | 2.9% | 1140 |
| `half`      | 0.098 | 682.5 | 73% | **6.77×** | 0.1% | 1440 |
| `bip`       | 1.820 | 295.1 | 32% | 0.37× | 1.0% | 1560 |

BIP→BSQ transpose: 0.429 ms at 625.8 GB/s counting read and write.

### Scaling with target count

At 128 bands. This is where the design either pays off or does not.

| targets | baseline | optimized | half | opt speedup | opt % peak | opt SM MHz |
|--------:|---------:|----------:|-----:|------------:|-----------:|-----------:|
|  1 | 0.174 | 0.168 | 0.095 | 1.04× | 85% |  390 |
|  2 | 0.352 | 0.168 | 0.096 | 2.10× | 85% |  705 |
|  4 | 0.708 | 0.171 | 0.098 | 4.14× | 84% |  765 |
|  8 | 1.386 | 0.244 | 0.203 | 5.67× | **59%** | **1440** |
| 16 | 2.843 | 0.481 | 0.406 | 5.91× | **60%** | **1455** |

`optimized` is flat at 84–85% of peak and 390–765 MHz through four targets —
textbook memory-bound. At eight it drops to 59% *and the clock jumps to
1440 MHz*. It has stopped being memory-bound.

### Scaling with band count

4 targets:

| bands | MB | baseline | optimized | half | bip | opt % peak |
|------:|---:|---------:|----------:|-----:|----:|-----------:|
|  16 |  16.8 | 0.044 | 0.030 | 0.026 | 0.073 | 59% |
|  32 |  33.6 | 0.170 | 0.050 | 0.038 | 0.377 | 71% |
|  64 |  67.1 | 0.345 | 0.089 | 0.058 | 0.875 | 80% |
| 113 | 118.5 | 0.611 | 0.150 | 0.088 | 2.164 | 84% |
| 128 | 134.2 | 0.699 | 0.170 | 0.098 | 1.825 | 85% |
| 224 | 234.9 | 1.232 | 0.315 | 0.167 | 7.020 | 80% |

`baseline` at 16 bands reports **164% of peak**, which is real rather than
broken. Its target loop is outermost *per thread*, so each thread re-reads its
own spectrum four times back to back; the resident working set
(`~126k threads × 16 bands × 4 B ≈ 8 MB`) is close enough to the 6 MB L2 that
most of the re-reads hit cache. At 128 bands the same figure is 64 MB, nothing
is retained, and the number falls back to true DRAM bandwidth.

`bip` at 16 bands reaches **98% of peak**, its best result anywhere, for the
same kind of reason: a 16-band spectrum is 64 bytes, exactly one cache line, so
the strided access costs nothing. It degrades monotonically as the spectrum
spans more lines — 38% at 32 bands, 23% at 113, 14% at 224.

## Where the optimized kernel stops being memory-bound

The inner loop issues `TT` constant-memory reads per vector load, where `TT` is
the targets scored in one pass and the load covers `PPT` pixels. Varying `PPT`
varies that ratio directly. At 128 bands:

| targets | const reads per 16-byte load | px=2 | px=4 | px=8 |
|--------:|-----------------------------:|-----:|-----:|-----:|
|  1 | 0.5 – 2 | 81% | 83% | 83% |
|  2 | 1 – 4   | 79% | 82% | 81% |
|  4 | 2 – 8   | 76% | 81% | 79% |
|  8 | 4 – 16  | **26%** | **60%** | 67% |
| 16 | 4 – 16  | **26%** | **59%** | 66% |

Collecting by ratio rather than by configuration:

| const reads per 16 B | achieved | SM clock |
|---------------------:|---------:|---------:|
| ≤ 4  | 76–85% |  570–1200 MHz |
| 8    | 59–60% |  1455 MHz |
| 16   | 26%    |  1860 MHz |

The knee sits between 4 and 8, and the clock rises exactly as the bandwidth
falls. **Keeping constant-memory reads at or below four per 16-byte load keeps
the kernel memory-bound**; past that it becomes constant-cache-bound and the
DRAM goes idle. Eight pixels per thread at eight targets restores the 4:1 ratio
and recovers 7 points (60% → 67%), which is consistent but not a full recovery.

Confirming the mechanism needs `ncu`, and the rented container blocked
performance counters (`ERR_NVGPUCTRPERM`).

### The band-count notch

On top of the ratio effect, at eight targets the geometries where **the band
count is a multiple of 128** lose a further ~15 points:

| bands | 8 targets | target stride | stride mod 512 B |
|------:|----------:|--------------:|-----------------:|
|  96 | 81% | 384 B | 384 |
| 112 | 74% | 448 B | 448 |
| **128** | **60%** | **512 B** | **0** |
| 144 | 79% | 576 B | 64 |
| 160 | 69% | 640 B | 128 |
| **256** | **68%** | **1024 B** | **0** |
| 288 | 77% | 1152 B | 128 |
| 320 | 73% | 1280 B | 256 |
| **384** | **65%** | **1536 B** | **0** |

The per-target stride in constant memory is `bands × 4` bytes. At a multiple of
128 bands that is a multiple of 512, and the hypothesis is that every target of
a band step then maps into one constant-cache set, where eight of them thrash a
four-way set. Four targets fit, and the same geometries show no loss at four
targets (128 bands, 4 targets: 85%). The prediction that 256 and 384 would dip
while 288, 320 and 256-at-four-targets would not was made before those five
were run, and all five held.

**A fix was attempted and reverted.** Padding the constant-memory stride by one
float bought 2 points at 128 bands and cost 17 at 144 and 16 at 288 — including
geometries whose constant memory was byte-for-byte unchanged. Passing the
stride as its own runtime parameter instead of reusing `bands` loses the
compiler the relationship between the loop bound and the address stride. A
template parameter carrying the pad would keep the stride expressed as `bands`
plus a constant; untested.

Real sensor geometries mostly avoid this: HyperBlood is 113 bands, HOT VIS is
16, AVIRIS is 224. But 128 is exactly the round number someone would configure.

## Pipeline, end to end

512×512×128, 4 targets, 3 streams, `--memory=copy`, 200 frames prefilled:

| variant | fps | p50 ms | p99 ms | upload | sam | detect | download |
|---------|----:|-------:|-------:|-------:|----:|-------:|---------:|
| `baseline`  | 83.0 | 35.88 | 40.01 | 35.20 | 0.671 | 0.023 | 0.029 |
| `optimized` | 83.6 | 35.80 | 36.39 | 35.43 | 0.182 | 0.021 | 0.029 |
| `half`      | 82.4 | 36.37 | 36.82 | 35.80 | 0.351 | 0.021 | 0.028 |

**The pipeline is PCIe-bound and the kernel is invisible** — 35.4 ms of upload
against 0.18 ms of SAM, so the kernel is 0.5% of the frame and all three
variants land within 1.5% of each other. 83 fps × 134.2 MB is 11.2 GB/s, a
PCIe Gen3 ×16 link running flat out (confirmed by querying the link *under
load*; the idle query reports Gen1 because the link power-manages down).

### Streams

| streams | fps | p50 ms |
|--------:|----:|-------:|
| 1 | 83.8 | 11.89 |
| 2 | 86.4 | 23.07 |
| 3 | 81.4 | 36.75 |
| 4 | 80.2 | 49.90 |
| 6 | 79.8 | 50.15 |

Throughput is flat while latency grows linearly — the signature of one
serialised resource. Extra streams queue behind the same copy engine instead of
overlapping with anything. **On a discrete GPU `--streams=1` is strictly
better**: same throughput at a third of the latency. The default of 3 is wrong
for this path and should only pay off on the Orin, where there is no copy to
serialise on.

## The predictions, scored

Written into this file before any of them were measured.

| # | prediction | verdict |
|---|------------|---------|
| 1 | zero-copy cuts memory traffic ~3× on Orin | **untested** — needs the Orin |
| 2 | `optimized` approaches 8× over `baseline` at 8 targets | **no** — 5.7×, capped by the constant-cache knee |
| 3 | BIP costs ~8× in load efficiency | **directionally right** — 32% vs 82% of peak; the sector metric needs `ncu` |
| 4 | fp16 ≈2× at the kernel, ≈nothing end to end | **confirmed**, both halves |

Prediction 2 was closer than 5.7× suggests. Scaling is clean to four targets
(4.14× at 4), and what stops it at eight is the constant-cache knee rather than
anything about amortisation. At band counts away from a multiple of 128 the
same case reaches 7.5–7.7×.

### Four explanations that the measurements killed

Recorded because each one cost real GPU time and each is the obvious first
guess:

| explanation | how it died |
|-------------|-------------|
| register pressure / occupancy | `sam_opt_kernel<8>` is 55 registers, 75% occupancy — and `sam_half_kernel<4>` sits at the same 75% without the cliff. The `<8,8>` variant uses 104 registers and is *faster*. Two pixels per thread, with the fewest registers, is 3× the slowest. |
| power throttling as the cause | cold vs hot costs `optimized` 13% and `baseline` 6% — real, but not the 40% being explained. |
| plane-stride aliasing | widths 512, 520, 544 and 576 all give 61–63% at the affected geometry. |
| padding the constant stride (the fix) | cost 17 points on geometries whose data layout did not change. Reverted. |

### One that was an instrumentation bug

The SM clock was first sampled *after* `cudaEventSynchronize`, by which point
the GPU is idle and ramping back to boost. It reported 1425 MHz for the slowest
configuration and 480 MHz for the fastest — exactly backwards. Sampling it
while the launches are still queued inverted the reading and turned the column
into the most useful diagnostic here.

## What the Orin still has to answer

Everything above is a discrete GPU behind a PCIe link — the configuration this
project was *not* built for.

* whether zero-copy removes the upload stage entirely, as designed
* whether, with the copy gone, the variant choice becomes visible end to end;
  here it was buried under a 35 ms transfer
* whether more than one stream helps once there is no copy engine to serialise
* what fraction of a 40 ms frame budget survives detection
* whether the constant-cache knee bites harder at 15–60 W than it does at 220 W

## Detection quality on HyperBlood

Measured on the CPU reference, so these hold regardless of which GPU runs them.

```sh
scripts/fetch_hyperblood.sh
python3 scripts/prepare_hyperblood.py

# --target=blood matters: with a full library the angle map holds the best
# match over all targets, which is not the same quantity as blood-likeness.
build/src/hsi_score --cube=data/hyperblood_prepared/A_1.hdr \
  --library=data/hyperblood_targets.csv --target=blood \
  --threshold=0.10 --dump-score=/tmp/A_1_blood.f32

python3 scripts/eval_detection.py /tmp/A_1_blood.f32 \
  data/hyperblood_prepared/A_1_gt.u8 --width=696 --height=520 --target=1
```

All 14 scenes, blood as the target, threshold 0.10 rad. Two signature regimes,
because they answer different questions:

* **on-scene** — the signature is the mean of that scene's own labelled blood.
  This is the operator-in-the-loop mode: mark target pixels in the current
  frame, then detect.
* **global** — one signature averaged over all 14 scenes, carried in from
  elsewhere. This is the ship-it-in-a-config-file mode.

| scene | on-scene AUC | prec | recall | global AUC | prec | recall |
|-------|-------------:|-----:|-------:|-----------:|-----:|-------:|
| A_1   | 0.7019 | 0.9924 | 0.1804 | 0.5979 | 0.8097 | 0.2355 |
| B_1   | 0.7662 | 0.8791 | 0.0662 | 0.5516 | 0.6893 | 0.0530 |
| C_1   | 0.7891 | 0.0826 | 0.0036 | 0.7218 | 0.8006 | 0.1315 |
| D_1   | 0.8104 | 0.9027 | 0.7274 | 0.9979 | 0.9997 | 0.1446 |
| E_1   | 0.7296 | 0.5793 | 0.0813 | 0.6894 | 0.3385 | 0.0590 |
| E_7   | 0.8096 | 0.2083 | 0.1009 | 0.6020 | 0.0039 | 0.0013 |
| E_21  | 0.7426 | 0.0501 | 0.0613 | 0.6721 | 0.0227 | 0.0051 |
| **F_1**  | **0.9945** | **1.0000** | 0.7073 | 0.6940 | 0.3351 | 0.0388 |
| F_1a  | 0.9930 | 0.6865 | 0.9438 | 0.9586 | 0.7250 | 0.2864 |
| F_1s  | 0.9964 | 0.8660 | 0.8790 | 0.9805 | 0.5377 | 0.2038 |
| F_2   | 0.9991 | 0.9275 | 0.9774 | 0.9412 | 0.5559 | 0.1913 |
| F_7   | 0.9838 | 0.9655 | 0.9076 | 0.9301 | 0.5992 | 0.1709 |
| F_21  | 0.9979 | 0.9847 | 0.9380 | 0.9191 | 0.3439 | 0.0694 |
| F_2k  | 0.9959 | 0.9161 | 0.9517 | 0.9122 | 0.1609 | 0.0431 |
| **mean** | **0.8793** | | | **0.7977** | | |

### What the table says

**The scene split matters more than the signature.** The `F_*` scenes — the
"frame" images, blood on a controlled background — reach 0.98–0.999 AUC with
recall around 0.9 when calibrated on scene. The `A`–`E` mock-up scenes, with
cluttered natural backgrounds, sit at 0.70–0.81. Same kernel, same signature
procedure; the difference is entirely how much the background overlaps the
target in spectral direction. SAM has no spatial context to fall back on when
it does.

**On-scene calibration is worth about 0.08 AUC on average**, and much more than
that at a fixed operating point. F_1 goes from 0.694 AUC with the global
signature to 0.9945 with its own, and its precision at 0.10 rad from 0.34 to
1.00.

### F_1 in detail — blood against six red lookalikes

The case this dataset exists for: blood alongside ketchup, artificial blood,
beetroot juice, poster paint, tomato concentrate and acrylic paint, all red,
none separable by colour. On-scene signature, 0.10 rad:

| metric | value |
|--------|------:|
| ROC AUC | 0.9945 |
| precision | 1.0000 — **0 false positives in 320 000 negatives** |
| recall | 0.7073 |

| class | mean angle | min angle |
|-------|-----------:|----------:|
| **blood** (target) | **0.0916** | 0.0216 |
| beetroot_juice     | 0.1857 | 0.1623 |
| poster_paint       | 0.2062 | 0.1724 |
| acrylic_paint      | 0.2144 | 0.1120 |
| ketchup            | 0.2374 | 0.1901 |
| artificial_blood   | 0.2795 | 0.1865 |
| tomato_concentrate | 0.3744 | 0.2324 |
| background         | 0.4573 | 0.1606 |

The margin is not luck: blood averages 0.0916 rad while the *closest single
pixel* of the nearest confuser, beetroot juice, sits at 0.1623 — above the
threshold. That separation is the entire argument for the spectral angle, and
it is what the rest of this project exists to compute quickly.

### The operational caveat

Ranking transfers between scenes; the threshold does not. Global-signature AUC
stays respectable almost everywhere, so the angle keeps ordering blood ahead of
the rest — but precision at a *fixed* 0.10 rad swings from 1.00 to 0.004
depending on the scene. A deployed detector cannot ship one constant threshold
with one library. It needs the threshold set per scene, or derived from the
angle distribution of the frame itself.

None of this is a claim about the state of the art. A single mean endmember is
the weakest reasonable signature, and published results that do better on this
dataset use per-scene signatures, spatial context or learned features — none of
which this pipeline provides. What it claims is that the angle is computed
correctly and fast, and that where the signature fits, the calls it makes are
right.

## Notes on interpretation

* An unpinned run measures the governor, not the code. Both scripts pin clocks
  where they have the privileges; if they warn that they could not, the numbers
  are a lower bound of unknown tightness.
* On the Orin, check the peak temperatures the script prints at the end. A
  thermal ceiling reached mid-run invalidates everything measured after it.
* Use `--prefill` to measure the GPU and omit it to measure the whole system.
  The synthetic generator writes tens of millions of floats per frame and can
  easily be the bottleneck; both numbers are worth having, but they answer
  different questions.
