# SIMD Without Drama: `-fno-math-errno` and `-fno-trapping-math` for HPC

This note explains what **`-fno-math-errno`** and **`-fno-trapping-math`** do, why they’re routinely used in high‑performance computing, and how they affect **auto‑vectorization** on CPU compilers (GCC/Clang/ICX). It includes small, copy‑paste examples and verification steps.

> **Bottom line:** These two options remove *legality blockers* that often prevent compilers from vectorizing loops. They do **not** (by themselves) change algebraic identities or rounding modes. Use them as your default for SIMD‑oriented hot loops; add `-ffast-math` only if you accept broader semantic changes (reassociation, ignoring NaNs/INFs, etc.).

---

## 1) What the flags declare

### `-fno-math-errno`
Tells the compiler: *“Treat math functions as if they don’t set `errno`; nobody will read it.”*  
Effect: calls to `sqrtf`, `sinf`, `logf`, etc. can be **inlined** or replaced with fast builtins / vector library calls without preserving `errno`. This enables loop vectorization across libm calls and more aggressive code motion.

### `-fno-trapping-math`
Tells the compiler: *“Assume floating‑point operations do not raise observable traps or affect the FP environment; speculative evaluation is safe.”*  
Effect: permits **if‑conversion** (compute both sides, select by a mask) and other speculative FP transformations required by masked SIMD. Without it, compilers must avoid evaluating FP ops in lanes that would have been skipped by a branch, blocking vectorization.

> Defaults (GCC): `-fmath-errno` **ON**, `-ftrapping-math` **ON**.  
> Bundles: `-ffast-math` / `-Ofast` imply both **OFF**. Use the two flags independently for a surgical, SIMD‑friendly baseline that preserves IEEE order/association.

---

## 2) Why they matter for vectorization

Auto‑vectorizers need two green lights:
1) **Legality** — the transformation must not change defined behavior.
2) **Profitability** — it should be faster.

These flags attack legality blockers:

- **Function calls in loops** (e.g., `sqrtf`, `hypotf`, `expf`): with `-fno-math-errno`, the compiler can inline/replace the calls and treat them as *pure*, enabling **SLP** (Superword Level Parallelism) or **loop** vectorization and call‑out to vector libm (e.g., glibc’s `libmvec`) when available.
- **Data‑dependent control flow** (branches that skip FP ops): with `-fno-trapping-math`, evaluating the FP ops in *all* lanes and then selecting results by a **mask** is legal. That unlocks **vectorization by masking** (predication), central to SIMD Mandelbrot, soft thresholding, piecewise transforms, etc.

---

## 3) Minimal examples

### 3.1 Libm in a loop (needs `-fno-math-errno`)
```c
// build: gcc -O3 -march=native -fno-math-errno -fno-trapping-math \
//               -fopt-info-vec-optimized -fopt-info-vec-missed sqrt_sum.c -lm
float sqrt_sum(const float* __restrict a, int n){
    float s = 0.0f;
    for (int i=0; i<n; ++i)
        s += sqrtf(a[i]);    // may vectorize to a vector-math call or inline sequence
    return s;
}
```

**Why it helps:** if `errno` must be preserved, the call is side‑effecting and harder to vectorize or move. Disabling `math-errno` lets the compiler use vector variants or inline sequences.

---

### 3.2 If‑conversion / masking (needs `-fno-trapping-math`)
```c
// build: gcc -O3 -march=native -fno-math-errno -fno-trapping-math \
//               -fopt-info-vec-optimized -fopt-info-vec-missed thresh.c
void transform(float* __restrict y, const float* __restrict x,
               float t, int n)
{
    for (int i=0; i<n; ++i) {
        // If-convertible pattern: branch -> masked select
        y[i] = (x[i] > t) ? (1.0f / x[i]) : 0.0f;  // possible div-by-zero in masked-off lanes
    }
}
```
**Why it helps:** with trapping math *enabled*, speculatively computing `1/x[i]` in lanes where `x[i]==0` could raise a trap, so the compiler keeps the branch and often **won’t vectorize**. With `-fno-trapping-math`, it may compute unconditionally in all lanes and then blend results, producing SIMD code.

---

### 3.3 Mandelbrot, per‑row masked SIMD
```c
// Sketch of the pattern; real code uses AVX2/AVX-512 intrinsics or OpenMP simd.
for (int iter=0; iter<maxIters; ++iter) {
    // compute zr2, zi2, etc. for all lanes
    // active = (zr2 + zi2) < 4
    if (none(active)) break;     // early exit when all lanes escaped
    // masked updates: only active lanes change
    zr = select(active, new_zr, zr);
    zi = select(active, new_zi, zi);
    iters += active;
}
```
**Why it helps:** the compiler is allowed to evaluate FP operations even in lanes that will be discarded by the subsequent `select`, because traps are assumed absent. That’s exactly what `-fno-trapping-math` buys you.

---

## 4) Verification workflow

### See what the compiler thinks
```bash
# Show defaults and effects of flags (GCC)
gcc -Q --help=optimizers | grep -E 'trapping-math|math-errno'
gcc -O3 -Q --help=optimizers | grep -E 'trapping-math|math-errno'
gcc -ffast-math -Q --help=optimizers | grep -E 'trapping-math|math-errno'

# Build with vectorization reports
gcc -O3 -march=native -fno-math-errno -fno-trapping-math \
    -fopt-info-vec-optimized -fopt-info-vec-missed -c yourfile.c
```
Look for: *“loop vectorized”*, *“if-converted”*, or the disappearance of *“possible dependence due to function call / may trap”* style warnings.

### Measure it (portable perf set)
```bash
taskset -c 0 perf stat -e cycles,instructions,branches,branch-misses,cache-misses ./your_exe
```
Expect fewer branches, similar or higher instruction counts (masked SIMD can do more work per cycle), and better time‑to‑solution on typical data.

---

## 5) Safety and scope (what these flags do **not** do)

- They **do not** enable reassociation or algebraic transformations that change results (that’s `-ffast-math`, `-fassociative-math`, etc.).
- They **do not** change rounding mode, subnormal handling, or FTZ/DAZ settings.
- They **do** assume **you won’t** read `errno` or FP exception flags, and won’t rely on SIGFPE traps from masked‑off operations.

If you rely on `fenv.h` (`fetestexcept`, `feclearexcept`, `#pragma STDC FENV_ACCESS ON`) or inspect `errno` for math calls, **don’t** use these flags for those translation units.

---

## 6) Quick decision chart

| Requirement | Can you use `-fno-math-errno`? | Can you use `-fno-trapping-math`? |
|---|---|---|
| You never read `errno` after math | ✅ Yes | – |
| You read FP exception flags / want SIGFPE traps | ❌ No | ❌ No |
| You want masked SIMD (predication) | ✅ Recommended | ✅ Required |
| You want vector calls to libm | ✅ Often required | – |
| You need strict IEEE & deterministic sums | ✅ Yes (no change) | ✅ Usually fine (no reassociation) |

---

## 7) Recommended “SIMD‑friendly but safe” flag sets

**Teaching / default HPC hot loop:**  
```bash
-O3 -march=native -fno-math-errno -fno-trapping-math
```
**When you accept broader numeric relaxations:**  
```bash
-Ofast    # implies -ffast-math, which also implies -fno-math-errno -fno-trapping-math
```
Pair with: `-fopt-info-vec-optimized -fopt-info-vec-missed` (GCC) or `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize` (Clang) to keep students focused on the compiler’s reasoning.

---

## 8) One-page takeaway

- `-fno-math-errno` → lets compilers treat libm as pure and vectorizable.  
- `-fno-trapping-math` → legalizes masked SIMD and if‑conversion by allowing speculative FP eval.  
- Together they **unblock auto‑vectorization** in many real kernels without changing arithmetic identities. Keep them on by default in numeric TUs unless you genuinely use `errno` or the FP environment.
