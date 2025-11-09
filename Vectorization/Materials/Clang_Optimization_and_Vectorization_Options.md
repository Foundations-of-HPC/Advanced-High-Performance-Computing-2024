# Clang Optimization & Vectorization Options — Small _excursus_ to the GCC-equivalent options

The aim of the followin is to give a glance on the Clang/LLVM options for optimization/vectorization, contrasted with GCC’s ones. I try to map common GCC flags to their Clang counterparts, highlight **behavioral differences**, and gives recipes to be used on real HPC/vector code.

> Scope: Clang/LLVM 12–18+ on x86-64 and AArch64 (Linux/macOS). Some flags exist earlier or later; where semantics are version-sensitive, I note it. 
> This is a summary of experience and man pages: test on your code, check man pages, let me know about inconsistencies
>
> [Link to the LLVM relevant page](https://llvm.org/docs/Vectorizers.html).

---

## 1) Target CPU & ISA: `-march`, `-mtune`, `-mcpu`, `-m*`

### Roughly equivalent options

* **GCC**: `-march=<cpu>` / `-mtune=<cpu>`
* **Clang**:
  * x86: **same**: `-march=<cpu>` / `-mtune=<cpu>`
  * AArch64/ARM: prefer `-mcpu=<cpu+features>` and optionally `-mtune=<cpu>`
  * “Host” detection: `-march=native` (x86) and `-mcpu=native` (AArch64) work on most builds.

Examples:

```bash
# x86
clang -O3 -march=native -mtune=native file.c
# AArch64 (Neoverse V1)
clang -O3 -mcpu=neoverse-v1 file.c
```

### Feature toggles (both compilers)

* `-mavx2`, `-mavx512f`, `-mfma`, `-msse4.2`, `-mf16c`, etc.
* On AArch64: `-march=armv8.5-a+dotprod+fp16` or `-mcpu=neoverse-n2+fp16`

### Important difference: “prefer vector width”

* **GCC** has `-mprefer-vector-width={128|256|512}`.
* **Clang** doesn’t expose the same UI. Options:

  1. **Disable AVX-512** when you want 256-bit: `-mavx2 -mno-avx512f` (common avoid down-clock, when needed).
  2. Use internal LLVM switches (not ABI-stable): `-mllvm -prefer-vector-width=256` or `-mllvm -force-vector-width=256`. These work, but treat them as **expert toggles** and re-validate on upgrades.

---

## 2) Optimization levels: `-O0/1/2/3/s/Ofast`

* **Same surface** as GCC:
  * `-O0`, `-O1`, `-O2`, `-O3`, `-Os`, `-Oz` (Clang also has `-Oz` often smaller than `-Os`)
  * `-Ofast` = **unsafe FP** (enables fast-math family; see §4).
  
* **Vectorizer defaults**:
* Clang’s **Loop Vectorizer** and **SLP Vectorizer** are reliably enabled at **`-O3`** (and under `-Ofast`). At `-O2`, LLVM’s defaults have changed over time; for portable habits, **assume `-O3`** for consistent vectorization unless you explicitly enable knobs (see §3).

---

## 3) Turning vectorization On/Off & Reporting

### Enabling/disabling vectorizers

* GCC: `-ftree-vectorize` (on at `-O3`/`-Ofast`).
* **Clang** equivalents:
  * Loop vectorizer: `-fvectorize` (usually **on at `-O3`**; you can enable is at lower levels).
  * SLP vectorizer: `-fslp-vectorize` (on at `-O3`; same idea).
    _note: about SLP vectorization: [link](https://web.eecs.umich.edu/~mahlke/courses/583f21/lectures/Nov29/Group5_paper.pdf)_

```bash
# Force at O2
clang -O2 -fvectorize -fslp-vectorize file.c
```

### Vectorization diagnostics

* GCC: `-fopt-info-vec*` family.
* **Clang**: the **`-Rpass`** family and optimization records.

  * `-Rpass=loop-vectorize -Rpass=slp-vectorize` — what vectorized
  * `-Rpass-missed=loop-vectorize` — why it *didn’t*
  * `-Rpass-analysis=loop-vectorize` — analysis notes
  * Persist complete records: `-fsave-optimization-record` (YAML/JSON) and optionally `-foptimization-record-file=...`

Example:

```bash
clang -O3 -march=native \
  -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize \
  -fsave-optimization-record \
  -c kernel.c
```

---

## 4) Floating-Point Flags: fast-math, errno, trapping, FMA

### Fast-math umbrella

* **Both** compilers: `-ffast-math` and `-Ofast` imply a bundle of relaxed FP assumptions.
* **Clang extras**: fine-grained model knob `-ffp-model=`

  * `-ffp-model=strict|precise|fast|except|value` (variants depend on version).
  * `fast` is similar in spirit to `-ffast-math` but is **not identical**; `-Ofast` still flips more switches.

### Finer-grained switches to use in Clang, similarly than in gcc

* `-fno-math-errno` — don’t preserve `errno` from libm.
* Clang is generally **more willing by default** to treat math as builtins and not set errno; to be explicit and portable across versions, pass **`-fno-math-errno`** for vector kernels.
* `-fno-trapping-math` — FP exceptions are not observed; enables reassociation/hoisting that helps vectorization.
* `-ffp-contract=fast|on|off` — Clang has the **extra `on`** mode: contract if it **doesn’t change** semantics; `fast` allows more aggressive FMA fusion.
* `-fno-signed-zeros`, `-fno-honor-nans`, `-fno-honor-infinities` — useful when your math allows it.
* `-freciprocal-math` — permits `1/x` → `rcp(x)` style transforms (and Newton steps).
* Denormals: `-fdenormal-fp-math=ieee|preserve-sign|positive-zero` (and `-fdenormal-fp-math-f32=`) to control flushing and cost model.

**Pragmatic “safe-ish” Clang recipe** comparable to your GCC middle ground:

```bash
clang -O3 -march=native \
  -fno-math-errno -fno-trapping-math -ffp-contract=fast
```

**“I accept relaxed FP semantics”:**

```bash
clang -Ofast -march=native \
  -ffp-contract=fast -freciprocal-math
```

---

## 5) Loop transforms

* Unrolling/peeling/unswitching:

  * GCC: `-funroll-loops`, `-fpeel-loops`, `-funswitch-loops`
  * **Clang**: front-end flags are limited; rely on LLVM passes at `-O3`.

    * You *can* nudge: `-mllvm -unroll-threshold=<N>`, `-mllvm -unroll-count=<N>`, `-mllvm -simple-loop-unswitch-threshold=<N>` (expert mode; version-sensitive).
    * For portability, prefer **source hints** (see pragmas below).

* Alias/`restrict`

  * Same as GCC: **C99 `restrict`** and `const` help a ton. Clang’s alias analysis is strong; give it clean IR.

* Data layout

  * Same truths: favor **SoA** over AoS for vector lanes, align arrays, and state alignment in directives.

---

## 6) OpenMP, SIMD, and Pragmas

* **OpenMP**

  * Clang: `-fopenmp` (needs `libomp`/`libiomp5`)
  * SIMD-only mode: **`-fopenmp-simd`** (like GCC) to enable `#pragma omp simd` without runtime.

* **Clang loop pragmas** (very handy!)
  * `#pragma clang loop vectorize(enable|disable)`
  * `#pragma clang loop interleave_count(N)`
  * `#pragma clang loop vectorize_width(W)`
  * `#pragma clang loop unroll_count(N)` or `unroll(disable)`
    These are **frontend-stable** ways to guide LLVM without resorting to `-mllvm` options.
  
* **OpenMP SIMD clauses** are portable across GCC/Clang:
* `#pragma omp simd safelen(W) aligned(ptr:64) reduction(+:sum)` etc.

---

## 7) LTO & PGO

### Link-Time Optimization

* GCC: `-flto` (+ plugin).
* **Clang**: **two modes**:

  * **ThinLTO** (scalable): `-flto=thin`
  * **Full LTO**: `-flto`
* Use `lld` for speed and clean diagnostics: `-fuse-ld=lld`

Example:

```bash
clang -O3 -flto=thin -march=native -c *.c
clang -O3 -flto=thin -fuse-ld=lld -o app *.o
```

### Profile-Guided Optimization

Clang has **two** PGO styles:

1. **Instrumentation PGO** (like GCC):

```bash
# build & train
clang -O3 -fprofile-generate -c *.c
clang -O3 -fprofile-generate -o app *.o
./app <inputs>
# use
clang -O3 -fprofile-use -c *.c
clang -O3 -fprofile-use -o app *.o
```

2. **Sample PGO** (use perf/AutoFDO samples):

```bash
clang -O3 -fprofile-sample-use=code.prof file.c
```

Sample PGO is powerful for large apps when you can’t run instrumented binaries in production.

---

## 8) Vector Math Libraries

* **GCC** often relies on platform ifuncs or SLEEF/short-vector libm via distro choices.
* **Clang** provides a clean switch: `-fveclib=`
  * `-fveclib=SVML` — generate calls to Intel SVML for vectorized transcendentals (needs linking availability).
  * `-fveclib=MassV` (on IBM), `-fveclib=Accelerate` (Apple), etc., platform permitting.
* If your loops use `sin/cos/exp/log` and won’t vectorize, try **`-fveclib=SVML`** on Intel + `-ffast-math` family.

---

## 9) Assembly & Inspection

* Emit annotated assembly: `-S -fverbose-asm`
* Optimization records (machine-readable): `-fsave-optimization-record` (YAML/JSON)
* Pass pipeline introspection (expert): `-mllvm -debug-pass-manager` (new PM) or `-mllvm -print-before-all` (noisy).

---

## 10) Behavior differences that matter in practice

1. **errno & builtins:** Clang tends to treat many `libm` calls as builtins earlier than GCC. If your code **observes `errno`**, you must compile with **`-fmath-errno`** to force preservation. For HPC, you typically want the opposite: **`-fno-math-errno`**.

2. **FMA contraction policy:** Clang’s `-ffp-contract=on` means “only if it doesn’t change the value,” while `fast` allows value-changing fusion. GCC only has `fast|off`. If you rely on exact rounding, pick `on` or `off`.

3. **AVX-512 downclock:** GCC gives you `-mprefer-vector-width=*`. Clang lacks that UX; the **robust** approach on Intel is to **disable AVX-512** (`-mno-avx512f`) when 512-bit hurts wall-time.

4. **Pragmas vs. flags:** LLVM favors **source pragmas** for loop hints (`#pragma clang loop …`). They age better than `-mllvm` “backdoor” switches.

5. **Vectorizer defaults:** Count on **`-O3`** for consistent vectorization on Clang. If you insist on `-O2`, force **`-fvectorize -fslp-vectorize`** and verify with `-Rpass`.

6. **Denormals handling:** Clang exposes it via `-fdenormal-fp-math=*`. This can move the needle on some CPUs (flush-to-zero vs. IEEE), but verify numerical acceptability.

---

## 11) Quick and incremental recipes for Clang

### 11.1 Strong, portable baseline

```bash
clang -O3 -march=native -flto=thin -fuse-ld=lld \
  -ffp-contract=fast \
  -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -fsave-optimization-record
```

### 11.2 Host-tuned FP-tolerant kernel

```bash
clang -O3 -march=native -flto=thin -fuse-ld=lld \
  -fno-math-errno -fno-trapping-math -ffp-contract=fast \
  -fvectorize -fslp-vectorize
```

### 11.3 Max throughput (accept unsafe FP)

```bash
clang -Ofast -march=native -flto=thin -fuse-ld=lld \
  -fopenmp-simd \
  -fveclib=SVML \
  -Rpass=loop-vectorize -Rpass=slp-vectorize
```

### 11.4 Avoid AVX-512 downclock, keep AVX2

```bash
clang -O3 -march=x86-64 -mavx2 -mno-avx512f -mfma
```

### 11.5 Sample PGO on top

```bash
clang -O3 -march=native -flto=thin -fuse-ld=lld \
  -fprofile-sample-use=code.prof
```

---

## 12) Option “Rosetta Stone” (GCC → Clang)

| Goal                    | GCC option                           | Clang counterpart                                            | Notes                                             |      |      |
| ----------------------- | ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- | ---- | ---- |
| Target host ISA         | `-march=native`                      | `-march=native` (x86) / `-mcpu=native` (AArch64)             | Same idea; AArch64 prefers `-mcpu`                |      |      |
| Tune scheduler          | `-mtune=<cpu>`                       | `-mtune=<cpu>`                                               | Works on both                                     |      |      |
| Enable vectorizer       | `-ftree-vectorize`                   | `-fvectorize` (Loop), `-fslp-vectorize` (SLP)                | At `-O3` both are on by default                   |      |      |
| Vector width preference | `-mprefer-vector-width=256`          | *(no direct)* → `-mno-avx512f` or `-mllvm -prefer-vector-width=256` | Disabling AVX-512 is the robust path              |      |      |
| Reports (vector)        | `-fopt-info-vec*`                    | `-Rpass[,-missed,-analysis]=loop-vectorize`                  | Plus `-fsave-optimization-record`                 |      |      |
| LTO                     | `-flto`                              | `-flto` or `-flto=thin`                                      | Use `-fuse-ld=lld` with Clang                     |      |      |
| PGO (instr)             | `-fprofile-generate/use`             | same                                                         | Clang also has **sample PGO**                     |      |      |
| Fast-math umbrella      | `-ffast-math`, `-Ofast`              | same                                                         | Implicit sub-flags differ in details              |      |      |
| No math errno           | `-fno-math-errno`                    | same                                                         | Clang often already behaves this way; be explicit |      |      |
| No trapping FP          | `-fno-trapping-math`                 | same                                                         | Unlocks more vector transforms                    |      |      |
| FMA contraction         | `-ffp-contract=fast`                 | `-ffp-contract=fast                                                 | on                                                | off` | `on` is Clang-specific middle ground              |      |      |
| FP model                | *(n/a consolidated)*                 | `-ffp-model=strict                                                  | precise                                           | fast` | Useful high-level knob                            |      |      |
| Flush/denormals         | GCC has `-fno-finite-math-only` etc. | `-fdenormal-fp-math=*`                                       | Helps model costs/behavior                        |      |      |
| Vector libm             | platform-dependent                   | `-fveclib=SVML                                                      | Accelerate                                        | MassV` | Great for `sin/cos/exp/log` loops                 |      |      |
| OpenMP SIMD             | `-fopenmp-simd`                      | `-fopenmp-simd`                                              | Same semantics                                    |      |      |

---

## 13) Practical Triage on Clang when a Loop Won’t Vectorize

1. Compile with `-O3 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -fsave-optimization-record` and **read the notes**.
2. Add `restrict`, `const`, and clean up aliasing.
3. Nudge with pragmas:

   ```c
   #pragma clang loop vectorize(enable) interleave_count(4)
   #pragma omp simd safelen(8) aligned(a,b:64) reduction(+:sum)
   ```
4. Loosen FP observation if acceptable: `-fno-math-errno -fno-trapping-math -ffp-contract=fast`.
5. If AVX-512 hurts, **turn it off**: `-mno-avx512f`.
6. Try a veclib for transcendentals: `-fveclib=SVML` with `-Ofast`.
7. For small trip counts, add `#pragma clang loop unroll_count(4)` to expose patterns.
8. Profile. If branchy: leverage **PGO** (instrumentation or sample).

---

## 14) Minimal, Copy-Paste Checklists

**Fast & safe-ish**

```bash
clang -O3 -march=native -flto=thin -fuse-ld=lld -ffp-contract=fast \
  -Rpass=loop-vectorize -Rpass-missed=loop-vectorize
```

**Relaxed FP**

```bash
clang -Ofast -march=native -flto=thin -fuse-ld=lld \
  -fno-math-errno -fno-trapping-math -freciprocal-math \
  -fveclib=SVML
```

**AVX-512 avoidance**

```bash
clang -O3 -mavx2 -mno-avx512f -mfma
```

**O2 + force vectorizers**

```bash
clang -O2 -fvectorize -fslp-vectorize -Rpass=loop-vectorize
```

---

## Final

* Clang’s **pragmas** are first-class levers; prefer them over unstable `-mllvm` switches.
* For frequency-sensitive Intel targets, “prefer 256-bit” often means **“disable AVX-512.”**
* `-Ofast` is a bulldozer. If your numerics allow it, it buys you easy wins; if not, assemble a targeted combo: `-O3 + -fno-math-errno + -fno-trapping-math + -ffp-contract=fast`.
* When in doubt, **measure** with LTO/PGO enabled; vectorization is a team sport across the pipeline.
