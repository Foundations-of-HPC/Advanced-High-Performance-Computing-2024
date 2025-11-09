# GCC Optimization & Vectorization Options

# — A Practical Guide —

This document wnats to be a reference for getting **predictable speed-ups** from GCC, with an emphasis on **vectorization** and **numerics**. 
It starts with the cornerstone first (architecture, `-O` levels, LTO, PGO), then digs into vectorization controls, floating-point flags (and their correctness trade-offs), diagnostics, and loop transforms.

> All examples use GCC syntax as of GCC 10–14 era and remain relevant on newer versions unless noted. 
> **However: check the manual, as always**
> When options interact with correctness (floating-point, aliasing), this is explicitly stated.
>
> GCC’s auto-vectorization is poorly documented:
> [GCC’s page on auto-vectorization](https://gcc.gnu.org/projects/tree-ssa/vectorization.html).
> [RedHat’s paper on GCC auto-vectorization](https://developers.redhat.com/articles/2023/12/08/vectorization-optimization-gcc#warnings)

---

## 1) Cornerstone: Whole-Program Performance

### 1.1 Target the CPU: `-march` / `-mtune` / `-m*`

- `-march=<cpu>`  
  *Enables* instruction sets and ABI details for the chosen micro-architecture. Allows autovectorization to use AVX2/AVX-512/NEON, etc.  
  Common x86 values: `native`, `haswell`, `skylake`, `cascadelake`, `icelake-client`, `znver2/3/4`, `raptorlake`, `alderlake`.  
  On ARM: `armv8.2-a`, `armv9-a`, `neoverse-n1`, etc.

- `-mtune=<cpu>`  
  Tunes scheduling, prefetching, and inlining heuristics *without* changing available instructions. Typically set to the same as `-march`; use `-mtune=native` if distributing binaries compiled with a conservative `-march`.

- `-march=native` / `-mtune=native`  
  **Preferred for us: always target the machine you are using** 
  Detect host CPU and optimize for it.

- Selected x86 width preferences:
  
  - `-mprefer-vector-width={128|256|512}`: guides preferred SIMD width when multiple are available (useful on AVX-512 parts to curb downclock or suit cache/latency).
  
  - `-mavx2`, `-mavx512f`, `-mfma`, etc.: explicitly enable features if you’re not using `-march` that implies them.

### 1.2 Optimization Levels: `-O0/1/2/3/s/Ofast`

- `-O0`: no optimization; fastest compile; good for debugging.

- `-O1`: light optimization.

- `-O2`: **default for production** in many projects; enables most speed-beneficial optimizations without risky FP transforms.

- `-O3`: adds more aggressive inlining and vectorization, loop transforms. **Often faster** for numeric/HPC kernels. Test carefully for code size and potential regressions.

- `-Ofast`: roughly `-O3` + **unsafe** math (`-ffast-math` and friends). Highest performance on many FP kernels **at the cost of strict IEEE semantics**. Use when you can tolerate re-associations and value-changing behavior. Check the discussion in the slides. 

### 1.3 Link-Time Optimization (LTO): `-flto`

- `-flto` across compile and link steps lets GCC optimize *between* translation units (cross-file inlining, constant propagation, dead code elimination).

- For large projects: consider **thin LTO** (`-flto=auto` or distro default) to balance memory/time.

### 1.4 Profile-Guided Optimization (PGO): `-fprofile-*`

- `-fprofile-generate` to build an instrumented binary. Run representative workloads to collect profiles.

- Rebuild with `-fprofile-use` (optionally `-fprofile-correction`) to feed real edge frequencies into inliner, vectorizer, and layout decisions.

- PGO can be **transformative** for complex codes where “hot path” isn’t obvious statically.

---

## 2) Vectorization: Controls, Hints, and Proofs

### 2.1 Turn it on (and know when it’s on)

- `-ftree-vectorize`  
  Autovectorizer for loops; **enabled by default at `-O3` and `-Ofast`**. You can also enable it at `-O2` manually.

- `-fopt-info-vec`, `-fopt-info-vec-optimized`, `-fopt-info-vec-missed`  
  Emit reports describing which loops vectorized, and *why* others didn’t. Add `-fopt-info-vec-missed=miss.txt` to get a to-do list.

**Quick check**

```bash
gcc -O3 -march=native -fopt-info-vec-optimized -c file.c
# Look for "vectorized 1 loops" in the output
```

### 2.2 Vector width and cost model

- `-fvect-cost-model={unlimited|cheap|dynamic}`  
  Heuristic for when vectorization “pays off.” `dynamic` (default) is balanced; `unlimited` pushes harder; `cheap` is conservative.

- `-mprefer-vector-width=...` (x86)  
  Nudges GCC to pick 128/256/512-bit vectors when multiple are available, which can help avoid AVX-512 downclock or reduce pressure for memory-bound kernels.

### 2.3 Loop idioms and enabling conditions

- `-fno-trapping-math` and `-fno-math-errno`  
  Tell GCC that FP exceptions aren’t observed and `errno` from libm calls isn’t required. **Frees the vectorizer to combine operations and use vector math sequences**. See §4 for correctness details.

- `-funsafe-math-optimizations`, `-ffast-math`  
  Allow reassociation and contraction of FP operations; can unlock vectorization by making reductions/accumulations and common sub-expressions vector-friendly.

- `-ffp-contract=fast|off`  
  `fast` allows FMA contraction when profitable; `off` prohibits contraction (more strict semantics; can reduce vectorization opportunities).

### 2.4 Loop transforms that help vectorization

- `-funroll-loops`  
  Helps peel/prologues and expose patterns to the vectorizer, especially if trip counts are small or residue handling hides opportunities. Test; can increase code size.
  Enabled at `-O3` (some since `-O2`)

- `-fpeel-loops`, `-funswitch-loops`  
  Peel removes edge iterations; unswitch hoists invariant conditions out. Both may make a loop easier to vectorize.

- `-fstrict-aliasing`  
  Lets GCC assume standard C aliasing rules; improves vectorization and scheduling. If your code violates aliasing rules, either fix it or compile with `-fno-strict-aliasing` and expect reduced optimization headroom.

- `-fno-tree-slp-vectorize` / `-ftree-slp-vectorize`  
  SLP (superword-level parallelism) vectorizes straight-line code (common in small kernels).

### 2.5 Directives that cooperate with the vectorizer

- `#pragma GCC ivdep`  
  Instructs the compiler to ignore assumed loop dependencies (you assert no harmful deps). Use carefully.

- `#pragma omp simd` (requires `-fopenmp-simd` or `-fopenmp`)  
  Standard way to **assert** independence and guide vector lengths with `safelen`, `aligned`, `reduction`, etc.  
  Paired with `-fopenmp-simd`, you get just the SIMD semantics without full OpenMP runtime.

---

## 3) Floating-Point Options and Their Effect on Correctness & Speed

### 3.1 The individually useful pair

- `-fno-math-errno`  
  Promises that your code won’t check `errno` after libm calls. Frees GCC to inline/transform FP math (e.g., approximate vector sequences) and reduces stalls.

- `-fno-trapping-math`  
  Asserts that FP exceptions (like invalid/overflow) aren’t observed. Allows reordering and vectorization across potential exception points.

**Why they matter for vectorization**  
Many transformations (FMA contraction, loop interchange, reassociation) are *not* allowed if a single‐precision operation could change globally observable state like `errno` or FP exception flags. Disabling these observations removes the barrier and typically yields more vector code.

### 3.2 The bigger lever: `-ffast-math`

**Check the slides for this**

- Expands to a set of flags (including `-fno-math-errno`, `-funsafe-math-optimizations`, `-fno-signed-zeros`, `-fno-trapping-math`, `-ffinite-math-only`, etc.).

- 
  
- **Pros:** Often the largest single boost for FP kernels, unlocking reassociation and contraction and enabling vector intrinsics.

- **Cons:** Can change results (NaN/Inf propagation, signed zero behavior, rounding, associativity). Only use when your algorithm tolerates this.
  Typically, you constatntly check the correctness of your code in "debug mode" (enabling trap at CPU level, assert, ...), and in producvtion you may consider to activate this options.

### 3.3 Fine-grained math knobs

- `-funsafe-math-optimizations` — allows algebraic simplifications that may break strict IEEE.

- `-fno-signed-zeros` — treat `+0.0` and `-0.0` as equivalent; helps value-numbering and vectorization.

- `-fno-rounding-math` — assume default rounding and that rounding mode isn’t changed dynamically.

- `-ffp-contract=fast|off` — allow/prohibit FMA fusion.

- `-ffinite-math-only` — assume no NaN/Inf; enables stronger transforms in loops.

**Pragmatic recipe**  
If you need a middle ground between strict and `-Ofast`:

```
-O3 -march=<cpu> -fno-math-errno -fno-trapping-math -ffp-contract=fast
```

Add `-fno-signed-zeros` and `-fno-rounding-math` **only** if you’ve verified numerics.

---

## 4) Diagnostics: Know What the Compiler Did (and Didn’t)

- **Vectorization reports**
  
  - `-fopt-info-vec` — summary of vectorized loops.
  
  - `-fopt-info-vec-missed` — *why* a loop didn’t vectorize (deps, aliasing, unknown trip count, etc.).
  
  - `-fopt-info-vec-optimized` — details on successful vectorization.

- **General optimization reports**
  
  - `-fopt-info` or `-fopt-info-optimized` — broader (inlining, unrolling, etc.).
  
  - `-fopt-info-rtl-all`, `-fopt-info-ipa-all` — deeper dives into low-level or interprocedural passes.

- **Assembly inspection**
  
  - `-S -fverbose-asm` — emit annotated assembly.
  
  - Combine with `objdump -d -Mintel` on the final binary when LTO/PGO is used.

---

## 5) Memory, Aliasing, and Data Layout (Big Enablers)

- `restrict` qualifiers (C99): promise non-aliasing of pointers. This single keyword often unlocks vectorization.

- Align data explicitly when profitable; communicate alignment with OpenMP SIMD clauses (`aligned(ptr: 64)`).

- Structure of Arrays (SoA) often vectorizes better than Array of Structures (AoS).

- Use `const` where appropriate; enables better load hoisting and alias analysis.

---

## 6) Loop & Inlining Heuristics

- `-finline-functions` (on at `-O3`) — can expose vectorizable loops.

- `-fpartial-inlining`, `-fipa-cp` (constant propagation), `-fipa-sra` (scalar replacement of aggregates) — interprocedural wins, especially with LTO/PGO.

- `-funroll-loops`, `-fpeel-loops`, `-funswitch-loops` — use surgically; measure code size vs. speed.

- `-fno-tree-vectorize` — explicitly disable if diagnosing miscompiles or performance regressions from vectorization (rare but useful to isolate).

---

## 7) OpenMP & SIMD

- `-fopenmp` — enables OpenMP pragmas (parallel + SIMD).

- `-fopenmp-simd` — enables only the SIMD subset (no runtime), letting you use `#pragma omp simd` to assert independence, handle reductions, and specify `safelen` and alignment.

- `#pragma omp declare simd` on functions enables vector versioning of callsites used inside loops.

---

## 8) Debugging vs. Optimization

- Debug builds (`-O0 -g`) may hide race conditions that show up with reordering at `-O3`.

- For numerical debugging of optimized builds:
  
  - Build with `-O2 -g -fno-omit-frame-pointer` to keep stack traces readable.
  
  - Temporarily disable specific transforms (e.g., `-ffp-contract=off`) to isolate sensitivity.

---

## 9) “Recipes” You Can Start From

### 9.1 Conservative but strong baseline (portable)

```bash
gcc -O3 -march=$your_target_platform -mtune=generic \
    -fopt-info-vec-missed=miss.txt -fopt-info-vec-optimized=vec.txt \
    -ffp-contract=fast
```

### 9.2 Host-tuned HPC build (numerics tolerant)

```bash
gcc -O3 -march=native -mtune=native -flto \
    -fno-math-errno -fno-trapping-math -ffp-contract=fast \
    -fopt-info-vec-missed=miss.txt -fopt-info-vec-optimized=vec.txt
```

### 9.3 Maximum throughput (you accept IEEE relaxations)

```bash
gcc -Ofast -march=native -flto \
    -fopenmp-simd \
    -funroll-loops -fpeel-loops -funswitch-loops \
    -fopt-info-vec-optimized=vec.txt
```

### 9.4 Add PGO on top (recommended for large codes)

**Build & train:**

```bash
gcc -O3 -march=native -flto -fprofile-generate -c *.c
gcc -O3 -march=native -flto -fprofile-generate -o app *.o
./app <representative_inputs>
```

**Rebuild using profiles:**

```bash
gcc -O3 -march=native -flto -fprofile-use -fprofile-correction -c *.c
gcc -O3 -march=native -flto -fprofile-use -o app *.o
```

---

## 11) Option Reference (Vectorization-Relevant)

| Option                             | What it does                                | When to use                        | Correctness risk  |
| ---------------------------------- | ------------------------------------------- | ---------------------------------- | ----------------- |
| `-march=<cpu>`                     | Enables ISA features (AVX2/AVX-512/NEON)    | Always                             | None              |
| `-mtune=<cpu>`                     | Tunes scheduling/heuristics                 | Always                             | None              |
| `-O3`                              | Aggressive inlining/loop opts/vectorization | Numeric kernels                    | Low               |
| `-Ofast`                           | `-O3` + unsafe math suite                   | Max perf FP kernels                | **High**          |
| `-flto`                            | Cross-file optimization                     | Medium-large code                  | None              |
| `-fprofile-generate/use`           | Profile-guided optimization                 | Apps with complex hot paths        | None              |
| `-ftree-vectorize`                 | Loop vectorizer                             | On at `-O3/Ofast`                  | None              |
| `-fopt-info-vec*`                  | Vectorization reports                       | Always during tuning               | None              |
| `-fvect-cost-model=*`              | Cost heuristic                              | When vectorization is timid        | None              |
| `-mprefer-vector-width=*`          | Prefer 128/256/512                          | Tame AVX-512 or tune bandwidth     | None              |
| `-fno-math-errno`                  | Ignore libm `errno`                         | FP kernels                         | Very low          |
| `-fno-trapping-math`               | Ignore FP exceptions                        | FP kernels                         | Low               |
| `-ffp-contract=fast`               | FMA fusion                                  | FMA-capable CPUs                   | Low-Moderate      |
| `-funsafe-math-optimizations`      | Algebraic FP liberties                      | If numerics tolerant               | Moderate          |
| `-fno-signed-zeros`                | Treat ±0 as same                            | If algorithm ok                    | Moderate          |
| `-fno-rounding-math`               | Assume default rounding                     | If no dynamic rounding             | Moderate          |
| `-funroll-loops`                   | Unroll to expose patterns                   | Hot small loops                    | None–Low          |
| `-fpeel-loops`, `-funswitch-loops` | Peel/unswitch                               | Complex loop bounds/guards         | None              |
| `-fopenmp-simd`                    | Enable `#pragma omp simd`                   | Declarative SIMD                   | None              |
| `-fno-strict-aliasing`             | Disable alias assumptions                   | Only if code violates aliasing     | None (but slower) |
| `-fstrict-aliasing`                | Enable alias rules                          | Default at `-O2`; keep code strict | None              |

---

## 11) Practical Triage if your hot loop does not vectorize

1. **Read the report:** compile with `-fopt-info-vec-missed=miss.txt`.

2. **Remove aliasing ambiguities:** use `restrict`, add `const`, separate arrays (SoA).

3. **Clarify loop bounds:** make trip counts known/large enough; hoist invariants; avoid complicated control flow inside the loop.

4. **Loosen FP observations if safe:** add `-fno-math-errno -fno-trapping-math`, consider `-ffp-contract=fast`.

5. **Assert independence:** try `#pragma omp simd reduction(+:sum)` and alignment hints.

6. **Adjust width:** try `-mprefer-vector-width=256` on AVX-512 machines.

7. **Declare the alignment** : use `__builtin_assume_aligned__`

8. **Use PGO:** sometimes the vectorizer is conservative until profiles reveal hot paths.

---

## 12) Verification & Reproducibility (VERY IMPORTANT)

When using non-strict FP flags:

- Keep a **reference build** (`-O2`, strict FP) and a **fast build**.

- Validate numerics with problem-specific norms (e.g., max relative error, ULP budgets).

- Log compiler version and flags in your benchmarking scripts.

---

## 13) Minimal Checklists

**“I just want it fast and safe-ish”**

- `-O3 -march=native -mtune=native -flto`

- `-ffp-contract=fast`

- `-fopt-info-vec-missed=miss.txt`

**“I accept relaxed FP semantics”**

- Previous list **plus** `-fno-math-errno -fno-trapping-math`

- Consider `-Ofast` if validated

**“It still won’t vectorize”**

- Add `#pragma omp simd` with proper clauses

- Try `-fvect-cost-model=unlimited` and `-funroll-loops`

- Re-layout data (SoA), add `restrict`, align arrays

---

### General Hints

- Start with **`-O3 -march=native -flto`**, then layer **diagnostics**, then **math relaxations** if permitted, then **PGO**.

- Use **reports** and **small, controlled experiments**; don’t guess.

- Document choices that affect correctness (`-ffast-math` family). Make the fast path intentional, not accidental.

























