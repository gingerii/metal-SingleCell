---
name: metal-shading-language
description: Apple Metal Shading Language (MSL) facts for writing mx.fast.metal_kernel code — atomics, SIMD-group reductions, threadgroup memory, barriers, memory orders, fast math. Consult before designing any custom Metal kernel.
---

# Metal Shading Language — distilled reference for this project

Source: `resources/Metal-Shading-Language-Specification.pdf` (383 pp, Metal 4.x, dated 2026-06-04).
Extract with `pymupdf` (`fitz`) — `pdftoppm`/`pdftotext` are NOT installed; the Read tool can't
render this PDF. Pattern: `fitz.open(path)[i].get_text()`. Section/page map at the bottom.

These are the facts that matter for `mx.fast.metal_kernel` (the kernel body is MSL; MLX wraps it).
MLX gives the kernel: scalar inputs as `const constant T*` pointers (deref `N[0]`), array inputs as
`device`/`constant`, **outputs as zero-initialized `device` buffers** (so output buffers are the place
for atomics). SIMD-group width on Apple silicon = **32 lanes**.

## Atomics (§6.16) — the load-bearing facts
- **`atomic_float`: ADD and SUB only** (no max/min/and/or/xor for float). **Device memory always**
  (Metal 3+); **threadgroup `atomic_float` add/sub needs Metal 4.1+**. Verified on this M3:
  `device atomic_float` + `atomic_fetch_add_explicit` compiles, runs, **exactly correct** (1M
  contended adds → 1e6), fast (5M adds = 0.5ms distributed / 4.67ms all-to-one). `fetch_max` on
  float → compile error (matches spec). **The old "no Metal float atomics" claim was WRONG.**
- **`atomic_int` / `atomic_uint`: ALL ops** — add, sub, and, or, xor, **max, min**, exchange,
  compare_exchange_weak. Both device and threadgroup. So integer/fixed-point hashes CAN use
  threadgroup atomics (quantize float weights to fixed-point uint if a TG float hash is needed
  pre-Metal-4.1).
- **`atomic_ulong`: 64-bit** add/max/min/and/or/xor/exchange (Metal 2.4+, Apple silicon, device only).
- **Fetch atomics support `memory_order_relaxed` ONLY.** No acquire/release/seq_cst on fetch ops.
  (`atomic_thread_fence` does support seq_cst; store/load support relaxed.) This is why ordered
  multi-step atomic protocols are out — design around relaxed-only.
- Usage in an MLX kernel: `device atomic_float* a = (device atomic_float*)out;
  atomic_fetch_add_explicit(a + idx, w, memory_order_relaxed);`

## SIMD-group functions (§6.10) — warp reductions WITHOUT atomics (cuGraph's primitive)
macOS Metal 2.1+ (always on visionOS). `T` = scalar/vector int or float (NOT bool/bfloat/long/ulong).
- **Reductions (broadcast result to all lanes):** `simd_sum(T)`, `simd_max(T)`, `simd_min(T)`,
  `simd_product`, `simd_and/or(Ti)`. **`simd_max`/`simd_min` work for FLOAT** (unlike float atomics!).
- **Prefix scans:** `simd_prefix_exclusive_sum/_inclusive_sum` (and product) — for compaction/segmented
  work within a SIMD-group.
- **Shuffles:** `simd_shuffle(data,lane)`, `simd_shuffle_xor(v,mask)` (butterfly reduction),
  `simd_shuffle_up/down(data,delta)`, `_rotate_*`, `_and_fill_*`, `simd_broadcast(data,lane)`,
  `simd_broadcast_first(data)`.
- **Votes:** `simd_ballot(expr)→simd_vote`, `simd_all/any(bool)`, `simd_active_threads_mask()`,
  `simd_is_first()`, `simd_is_helper_thread()`.
- **Implication for clustering:** a SIMD-group (32 lanes) per vertex can aggregate neighbor-community
  weights cooperatively via `simd_sum`/`simd_shuffle_xor` — no threadgroup atomics needed, and float
  reductions are first-class. This is the route to retire the O(d²) per-vertex move kernel + host tail.

## Barriers & memory (§6.10.1, §6.16.2)
- `threadgroup_barrier(mem_flags)` / `simdgroup_barrier(mem_flags)`: execution + memory barrier; ALL
  threads in the group must reach it (incl. inside conditionals/loops). Apple silicon: an ended thread
  no longer blocks the barrier.
- `mem_flags`: `mem_none` (exec only), `mem_device`, `mem_threadgroup`, `mem_texture`, `mem_threadgroup_imageblock`.
- **No grid-wide barrier across threadgroups** (only within a threadgroup/SIMD-group) — confirmed; a
  whole-graph synchronization within one kernel dispatch is impossible. Multi-pass = multi-dispatch.
- `thread_scope` variants of barriers exist in Metal 4.1+ (threadgroup/simdgroup/device scope).

## Threadgroup memory
- **~32 KB** of threadgroup memory budget (tile functions explicitly 32 KB, p132). A per-vertex hash
  bigger than that must fall back (the reason high-degree super-vertices can't use a TG hash).

## Fast math (§6.x, p209)
- `metal::fast::` and `metal::precise::` nested namespaces select fast/precise variants of
  distance/length/normalize and math funcs; `-ffast-math` compiler flag (MLX may set its own).

## Page map (for re-extraction)
- Atomics: types intro p39, full §6.16.4 **p298–310** (store p302, fetch ops + Table 6.27 **p307**, 64-bit p307).
- SIMD-group & sync: §6.10 **p215–225** (barriers p215–217, permute/shuffle p218–221, reductions Table 6.15 **p221–222**).
- Math functions: p205–210. Address spaces: p11–13, 21. Data types (half/bfloat): p26+.

## What this unlocked / next ideas
- Confirmed float atomics → reopened coloring-free clustering (shipped, see [[graph-clustering]]).
- **SIMD-group-per-vertex aggregation** (simd_sum / simd_shuffle_xor) is the clean O(d) replacement
  for the O(d²) move kernel + host tail — no threadgroup-atomic dependency, float reductions native.
- For a TG float hash pre-Metal-4.1: use `atomic_uint` fixed-point (quantize weights), then convert.
