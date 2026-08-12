"""Exact geometric predicates for Delaunay, on integer coordinates.

The orientation and in-circle tests are the correctness lynchpin of any Delaunay
algorithm: every insertion and every flip is decided by the *sign* of a determinant, and
a single sign that rounds the wrong way does not produce a slightly-wrong triangulation
— it produces an inconsistent one, which then hangs the flipping loop or emits inverted
triangles.

The usual answer is Shewchuk's adaptive floating-point predicates: a fast filter with an
a-priori error bound, falling back to exact expansion arithmetic. That machinery exists
because the inputs are arbitrary doubles. Ours are not, and Metal has no fp64 anyway, so
we take a different route that is exact by construction:

1. **Condition the points onto an integer lattice** (:func:`condition_points`). The scale
   is chosen so one nearest-neighbour spacing spans ``TARGET_UNITS`` lattice units, i.e.
   the quantisation error is ~0.1% of the distance between adjacent cells/spots — orders
   of magnitude below the precision of any spatial assay.
2. **Evaluate the determinants in 64-bit integers.** Metal's ``long`` is a true exact
   64-bit type on Apple silicon (verified against NumPy right up to the wrap boundary),
   so with bounded operands there is no rounding at all: the sign is the true sign, ties
   are true ties, and there is no tolerance to tune.

The catch is operand width. In-circle is a degree-4 determinant, so it only stays inside
int64 while the coordinate differences are bounded — see :data:`SAFE_ABS`. Both predicates
therefore return an ``unsafe`` mask alongside the signs, and callers must not silently
ignore it. On the CPU side we re-evaluate those in Python's arbitrary-precision integers;
:func:`incircle` never returns a value it cannot stand behind.

Degeneracy is a genuine zero here, not a near-zero: four exactly cocircular lattice points
give ``incircle == 0``. We deliberately do **not** apply simulation of simplicity. The
flipping loop treats 0 as "already locally Delaunay" and leaves the edge alone, which is a
valid Delaunay triangulation (they all are, on a tie) and — unlike a symbolic tiebreak —
cannot cycle. Square lattices, the one input where ties are common (32.8% of edges), are
exactly where a cycling flip loop would otherwise be a real risk.
"""

from __future__ import annotations

import numpy as np

#: Largest coordinate *difference* for which the in-circle determinant is guaranteed to
#: fit in a signed 64-bit integer. With all six translated differences bounded by ``M``
#: the determinant is bounded by ``12 * M**4``; at ``M = 2**14`` that is ``2**59.6``,
#: comfortably inside ``2**63``, and every intermediate product (largest ``2 * M**3``)
#: fits too. The true limit is ``M < 29650``; we round down to a power of two.
SAFE_ABS = 1 << 14

#: Orientation is only degree 2 — the determinant is bounded by ``2 * M**2`` — so it
#: tolerates far wider operands than in-circle does.
SAFE_ABS_ORIENT = 1 << 30

#: Hard ceiling on a conditioned coordinate, enforced by :func:`condition_points`. With
#: differences bounded by ``2**30`` the in-circle determinant is bounded by ``12 * M**4 =
#: 2**123.6``, which the 128-bit path evaluates exactly. This is what makes the GPU
#: predicate total: there is no input for which it has to return "ask the host".
MAX_COORD = 1 << 30

#: Lattice units per nearest-neighbour spacing, chosen by :func:`condition_points`. 1024
#: puts quantisation ~1000x below the point spacing while leaving room for a triangle to
#: span ~16 spacings before it exceeds :data:`SAFE_ABS`.
TARGET_UNITS = 1024


def _median_spacing(coords):
    """Median nearest-neighbour distance, from a sample when the cloud is large."""
    from scipy.spatial import cKDTree

    n = len(coords)
    rng = np.random.default_rng(0)
    sample = coords if n <= 20000 else coords[rng.choice(n, 20000, replace=False)]
    tree = cKDTree(coords)
    d, _ = tree.query(sample, k=2)                 # k=1 is the point itself
    d = d[:, 1]
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.0


def condition_points(coords, target_units: int = TARGET_UNITS):
    """Snap ``coords`` (n, 2) onto an integer lattice fine enough to be lossless in practice.

    Returns ``(ipts, info)`` where ``ipts`` is an ``int64`` (n, 2) array and ``info``
    records the affine map back to the input frame plus the diagnostics a caller needs to
    decide whether the conditioning was benign: how many distinct points survived, and how
    far any point moved.

    The scale is a power of two, so the snap is the only lossy step — the multiply itself
    is exact in floating point.
    """
    pts = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected an (n, 2) coordinate array, got {pts.shape}")
    n = len(pts)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64), {"scale": 1.0, "offset": np.zeros(2),
                                                  "n_distinct": 0, "max_shift": 0.0,
                                                  "spacing": 0.0}

    offset = pts.min(axis=0)
    spacing = _median_spacing(pts)
    if spacing <= 0:                                # every point identical
        scale = 1.0
    else:
        span = float(np.max(pts.max(axis=0) - offset))
        # power of two, and never so large that a coordinate exceeds MAX_COORD — past
        # that the 128-bit determinant is no longer guaranteed to fit
        scale = 2.0 ** np.floor(np.log2(target_units / spacing))
        max_scale = 2.0 ** np.floor(np.log2(MAX_COORD / max(span, 1e-12)))
        scale = min(scale, max_scale)

    ipts = np.rint((pts - offset) * scale).astype(np.int64)
    shift = float(np.max(np.abs((pts - offset) - ipts / scale))) if n else 0.0
    n_distinct = len(np.unique(ipts, axis=0))
    return ipts, {"scale": scale, "offset": offset, "spacing": spacing,
                  "n_distinct": n_distinct, "max_shift": shift}


# --------------------------------------------------------------------------- CPU exact


def orient2d(pts, a, b, c):
    """Sign of the orientation determinant of triangles ``(a, b, c)``, exactly.

    ``pts`` is the conditioned ``int64`` (n, 2) array; ``a``/``b``/``c`` are index arrays.
    Returns ``int8`` signs: ``+1`` counter-clockwise, ``-1`` clockwise, ``0`` collinear.
    """
    pts = np.asarray(pts, dtype=np.int64)
    a, b, c = (np.asarray(i, dtype=np.int64) for i in (a, b, c))
    bx = pts[b, 0] - pts[a, 0]
    by = pts[b, 1] - pts[a, 1]
    cx = pts[c, 0] - pts[a, 0]
    cy = pts[c, 1] - pts[a, 1]
    unsafe = (np.maximum(np.maximum(np.abs(bx), np.abs(by)),
                         np.maximum(np.abs(cx), np.abs(cy))) > SAFE_ABS_ORIENT)
    det = bx * cy - by * cx
    sign = np.sign(det).astype(np.int8)
    if unsafe.any():
        sign[unsafe] = _orient_bigint(pts, a[unsafe], b[unsafe], c[unsafe])
    return sign


def incircle(pts, a, b, c, d):
    """Sign of the in-circle determinant, exactly.

    For a counter-clockwise triangle ``(a, b, c)``: ``+1`` if ``d`` lies strictly inside
    the circumcircle, ``0`` if the four points are exactly cocircular, ``-1`` outside.
    The caller is responsible for passing a counter-clockwise triangle — this is a raw
    determinant, not the "is this edge legal" test.

    Entries whose operands exceed :data:`SAFE_ABS` are recomputed in Python integers, so
    the result is exact regardless of how far apart the points are.
    """
    pts = np.asarray(pts, dtype=np.int64)
    a, b, c, d = (np.asarray(i, dtype=np.int64) for i in (a, b, c, d))
    ax = pts[a, 0] - pts[d, 0]
    ay = pts[a, 1] - pts[d, 1]
    bx = pts[b, 0] - pts[d, 0]
    by = pts[b, 1] - pts[d, 1]
    cx = pts[c, 0] - pts[d, 0]
    cy = pts[c, 1] - pts[d, 1]
    widest = np.abs(np.stack([ax, ay, bx, by, cx, cy])).max(axis=0)
    unsafe = widest > SAFE_ABS

    aa = ax * ax + ay * ay
    bb = bx * bx + by * by
    cc = cx * cx + cy * cy
    det = (ax * (by * cc - bb * cy)
           - ay * (bx * cc - bb * cx)
           + aa * (bx * cy - by * cx))
    sign = np.sign(det).astype(np.int8)
    if unsafe.any():
        sign[unsafe] = _incircle_bigint(pts, a[unsafe], b[unsafe], c[unsafe], d[unsafe])
    return sign


def _orient_bigint(pts, a, b, c):
    out = np.empty(len(a), dtype=np.int8)
    for k in range(len(a)):
        (axp, ayp), (bxp, byp), (cxp, cyp) = (
            (int(pts[i, 0]), int(pts[i, 1])) for i in (a[k], b[k], c[k]))
        det = (bxp - axp) * (cyp - ayp) - (byp - ayp) * (cxp - axp)
        out[k] = (det > 0) - (det < 0)
    return out


def _incircle_bigint(pts, a, b, c, d):
    out = np.empty(len(a), dtype=np.int8)
    for k in range(len(a)):
        dxp, dyp = int(pts[d[k], 0]), int(pts[d[k], 1])
        axp, ayp = int(pts[a[k], 0]) - dxp, int(pts[a[k], 1]) - dyp
        bxp, byp = int(pts[b[k], 0]) - dxp, int(pts[b[k], 1]) - dyp
        cxp, cyp = int(pts[c[k], 0]) - dxp, int(pts[c[k], 1]) - dyp
        aa = axp * axp + ayp * ayp
        bb = bxp * bxp + byp * byp
        cc = cxp * cxp + cyp * cyp
        det = (axp * (byp * cc - bb * cyp)
               - ayp * (bxp * cc - bb * cxp)
               + aa * (bxp * cyp - byp * cxp))
        out[k] = (det > 0) - (det < 0)
    return out


# ------------------------------------------------------------------------------ Metal


#: 128-bit signed integer arithmetic, from pairs of 64-bit words. Metal has no wide
#: integer type and no ``mulhi`` for 64-bit operands, so the 64x64 -> 128 product is
#: assembled from four 32-bit partial products. Only the in-circle path needs this, and
#: only for the minority of quads whose operands exceed ``SAFE_ABS``; those are real
#: geometry (a hull edge bridging a gap between tissue regions is thousands of times the
#: local spacing), not a pathological input, so the branch has to be exact rather than
#: approximate. Everything is two's complement, matching the 64-bit path's semantics.
_I128_HEADER = """
#include <metal_stdlib>
using namespace metal;

struct i128 { ulong lo; ulong hi; };

inline i128 i128_from(long v)      { return {(ulong)v, v < 0 ? ~0UL : 0UL}; }
inline bool i128_neg(i128 a)       { return (a.hi >> 63) != 0; }
inline bool i128_zero(i128 a)      { return a.lo == 0UL && a.hi == 0UL; }

inline i128 i128_negate(i128 a) {
    ulong lo = ~a.lo + 1UL;
    ulong hi = ~a.hi + (lo == 0UL ? 1UL : 0UL);
    return {lo, hi};
}

inline i128 i128_add(i128 a, i128 b) {
    ulong lo = a.lo + b.lo;
    return {lo, a.hi + b.hi + (lo < a.lo ? 1UL : 0UL)};
}

inline i128 i128_sub(i128 a, i128 b) { return i128_add(a, i128_negate(b)); }

// magnitude of a 128-bit signed value, as an unsigned 128
inline i128 i128_abs(i128 a) { return i128_neg(a) ? i128_negate(a) : a; }

// exact 64x64 -> 128 unsigned product, via 32-bit limbs
inline i128 u64_mul(ulong a, ulong b) {
    ulong a0 = a & 0xFFFFFFFFUL, a1 = a >> 32;
    ulong b0 = b & 0xFFFFFFFFUL, b1 = b >> 32;
    ulong p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    ulong mid = (p00 >> 32) + (p01 & 0xFFFFFFFFUL) + (p10 & 0xFFFFFFFFUL);
    ulong lo  = (p00 & 0xFFFFFFFFUL) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return {lo, hi};
}

// unsigned 128 x 64 -> 128, truncating. Callers guarantee the result fits.
inline i128 u128_mul_u64(i128 a, ulong b) {
    i128 r = u64_mul(a.lo, b);
    r.hi += a.hi * b;
    return r;
}

inline i128 i128_mul_i64(i128 a, long b) {
    bool neg = i128_neg(a) != (b < 0);
    i128 r = u128_mul_u64(i128_abs(a), (ulong)abs(b));
    return neg ? i128_negate(r) : r;
}

inline i128 i64_mul(long a, long b) {
    i128 r = u64_mul((ulong)abs(a), (ulong)abs(b));
    return ((a < 0) != (b < 0)) ? i128_negate(r) : r;
}

inline char i128_sign(i128 a) { return i128_zero(a) ? 0 : (i128_neg(a) ? -1 : 1); }
"""


_PREDICATE_KERNEL_SOURCE = """
    uint t = thread_position_in_grid.x;
    if (t >= (uint)nquad[0]) return;

    int ia = quads[4 * t + 0], ib = quads[4 * t + 1];
    int ic = quads[4 * t + 2], id = quads[4 * t + 3];

    long ax = (long)px[ia] - (long)px[id], ay = (long)py[ia] - (long)py[id];
    long bx = (long)px[ib] - (long)px[id], by = (long)py[ib] - (long)py[id];
    long cx = (long)px[ic] - (long)px[id], cy = (long)py[ic] - (long)py[id];

    long m = max(max(max(abs(ax), abs(ay)), max(abs(bx), abs(by))),
                 max(abs(cx), abs(cy)));

    if (m <= (long)SAFE) {                 // the common case: exact in plain 64-bit
        long aa = ax * ax + ay * ay;
        long bb = bx * bx + by * by;
        long cc = cx * cx + cy * cy;
        long det = ax * (by * cc - bb * cy)
                 - ay * (bx * cc - bb * cx)
                 + aa * (bx * cy - by * cx);
        sign[t] = (char)((det > 0) ? 1 : ((det < 0) ? -1 : 0));
        return;
    }

    // wide operands: same determinant, 128-bit. Bounded by 12 * MAX_COORD**4 = 2**123.6.
    i128 aa = i128_add(i64_mul(ax, ax), i64_mul(ay, ay));
    i128 bb = i128_add(i64_mul(bx, bx), i64_mul(by, by));
    i128 cc = i128_add(i64_mul(cx, cx), i64_mul(cy, cy));

    i128 t1 = i128_sub(i128_mul_i64(cc, by), i128_mul_i64(bb, cy));
    i128 t2 = i128_sub(i128_mul_i64(cc, bx), i128_mul_i64(bb, cx));
    i128 t3 = i128_sub(i64_mul(bx, cy), i64_mul(by, cx));

    i128 det = i128_sub(i128_mul_i64(t1, ax), i128_mul_i64(t2, ay));
    // aa and t3 both fit in 64 bits of magnitude, so this product is exact
    bool s3 = i128_neg(aa) != i128_neg(t3);
    i128 p3 = u64_mul(i128_abs(aa).lo, i128_abs(t3).lo);
    det = i128_add(det, s3 ? i128_negate(p3) : p3);

    sign[t] = i128_sign(det);
"""


def incircle_gpu(ipts, quads):
    """In-circle signs for ``quads`` (m, 4) index rows, evaluated on the GPU.

    Exact for every input, with no host round trip: quads inside :data:`SAFE_ABS` take a
    plain 64-bit determinant, the rest take a 128-bit one. Matches :func:`incircle`
    element for element.
    """
    import mlx.core as mx

    ipts = np.asarray(ipts, dtype=np.int64)
    quads = np.ascontiguousarray(np.asarray(quads, dtype=np.int32))
    m = len(quads)
    if m == 0:
        return np.zeros(0, dtype=np.int8)
    if np.abs(ipts).max(initial=0) > MAX_COORD:
        raise ValueError(
            "coordinates exceed MAX_COORD; pass them through condition_points() first")

    kernel = mx.fast.metal_kernel(
        name="incircle_exact",
        input_names=["px", "py", "quads", "nquad"],
        output_names=["sign"],
        header=_I128_HEADER + f"\n#define SAFE {SAFE_ABS}\n",
        source=_PREDICATE_KERNEL_SOURCE,
    )
    (out,) = kernel(
        inputs=[mx.array(ipts[:, 0]), mx.array(ipts[:, 1]),
                mx.array(quads.reshape(-1)), mx.array([m], dtype=mx.int32)],
        grid=(m, 1, 1), threadgroup=(min(256, m), 1, 1),
        output_shapes=[(m,)], output_dtypes=[mx.int8],
    )
    return np.array(out)
