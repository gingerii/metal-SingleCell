"""gDel2D Phase 1, as a vectorised CPU reference.

This is not the fast path and never will be — it exists so that every GPU kernel written
next has an exact target to be pinned against, and so the algorithm's correctness can be
argued separately from Metal. It is deliberately written the way the kernels will be:
whole-array steps, one thread's worth of work per row, conflict resolution by scatter-min
rather than by locking. There is no per-triangle Python loop anywhere in the hot path.

The algorithm (Cao, Nanjappa, Gao & Tan) alternates two data-parallel phases until every
point is inserted:

* **Batch insertion** — every triangle that owns at least one uninserted point splits on
  exactly one of them, all triangles at once. Choosing the point nearest the triangle's
  circumcentre grows the lifted hull fastest and leaves fewer edges to repair.
* **Flipping** — every edge that fails the in-circle test flips, as many at a time as can
  be done without two flips touching the same triangle. Lawson's theorem guarantees this
  terminates at the Delaunay triangulation in 2D, which is why we need none of the paper's
  Phase 2 star splaying: that exists only because 3D flipping can get stuck.

The convex hull is handled by a single **symbolic vertex at infinity** rather than a big
enclosing triangle. A ghost triangle ``(a, b, INF)`` represents the outer face across hull
edge ``a -> b``, and both predicates degrade to orientation tests on it: a point is "inside
its circumcircle" exactly when it is outside that hull edge. Hull maintenance then needs no
special code — a reflex hull vertex is just a ghost-ghost edge that fails the in-circle
test, and the ordinary flip removes it. The alternative, three far-away real vertices, puts
artificial coordinates into the exact predicates and only approximately reproduces the hull.

Every predicate call goes through :mod:`.predicates`, so the whole thing is exact.
"""

from __future__ import annotations

import numpy as np

from .predicates import condition_points, incircle, orient2d


# --------------------------------------------------------------------------- Hilbert

def hilbert_order(ipts, bits: int = 16):
    """Indices that sort points along a Hilbert curve.

    Insertion order decides how much work the flipping phase has to undo. A Hilbert
    ordering keeps consecutive points close in space, so a batch of insertions lands in
    many different triangles instead of piling into one — which is what makes the batch
    wide. It also gives the eventual kernels coalesced access to the point array.
    """
    pts = np.asarray(ipts, dtype=np.int64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int64)

    side = 1 << bits
    lo = pts.min(axis=0)
    span = np.maximum(pts.max(axis=0) - lo, 1)
    x = ((pts[:, 0] - lo[0]) * (side - 1) // span[0]).astype(np.int64)
    y = ((pts[:, 1] - lo[1]) * (side - 1) // span[1]).astype(np.int64)

    rx = np.zeros(len(pts), dtype=np.int64)
    ry = np.zeros(len(pts), dtype=np.int64)
    d = np.zeros(len(pts), dtype=np.int64)
    s = side >> 1
    while s > 0:
        rx[:] = (x & s) > 0
        ry[:] = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        # rotate the quadrant so the curve stays continuous
        swap = ry == 0
        flip = swap & (rx == 1)
        x[flip], y[flip] = s - 1 - x[flip], s - 1 - y[flip]
        x[swap], y[swap] = y[swap].copy(), x[swap].copy()
        s >>= 1
    return np.argsort(d, kind="stable")


# ------------------------------------------------------------- predicates with INF
#
# Vertex index ``n`` is the symbolic point at infinity. These wrappers are the only place
# that knows it exists; everything below treats ghost and finite triangles alike.


def _rotate_inf_last(tv, inf):
    """Rotate each triangle so a vertex at infinity sits in the last slot."""
    out = tv.copy()
    for k in (0, 1):
        hit = out[:, k] == inf
        if hit.any():
            out[hit] = np.roll(out[hit], -(k + 1), axis=1)
    return out


def tri_contains(pts, mesh, tid, q, inf):
    """Is point ``q[i]`` inside triangle ``tid[i]``? Boundaries count as inside.

    For a ghost the test is **not** just "outside this hull edge". That half-plane test
    is the obvious thing and it is wrong: a point beyond a hull *vertex* is outside two
    hull edges, so the ghosts overlap and the plane is not partitioned. Locating such a
    point to the wrong one of the two, then splitting that ghost, inserts it into a hull
    chain it is not adjacent to and folds the boundary — the triangulation stays locally
    consistent and every triangle stays counter-clockwise, but the boundary self-
    intersects and covers part of the plane twice. It took an area computation to see it.

    So a ghost owns the wedge, not the half-plane: ``p`` belongs to ``(a, b, INF)`` when it
    is outside edge ``a -> b`` *and* not outside the preceding hull edge ``z -> a``. The
    hull edges a point can see form a contiguous chain, so exactly one of them qualifies.
    """
    tid = np.asarray(tid)
    tv = mesh.tri[tid]
    ghost = (tv == inf).any(axis=1)
    out = np.zeros(len(tv), dtype=bool)

    if (~ghost).any():
        f = np.where(~ghost)[0]
        a, b, c = tv[f, 0], tv[f, 1], tv[f, 2]
        out[f] = ((orient2d(pts, a, b, q[f]) >= 0)
                  & (orient2d(pts, b, c, q[f]) >= 0)
                  & (orient2d(pts, c, a, q[f]) >= 0))
    if ghost.any():
        g = np.where(ghost)[0]
        k = np.argmax(tv[g] == inf, axis=1)
        a = tv[g, (k + 1) % 3]
        b = tv[g, (k + 2) % 3]
        prev = mesh.nbr[tid[g], (k + 2) % 3] // 3      # ghost across edge (INF, a)
        pv = mesh.tri[prev]
        kp = np.argmax(pv == inf, axis=1)
        z = pv[np.arange(len(prev)), (kp + 1) % 3]
        out[g] = ((orient2d(pts, a, b, q[g]) > 0)
                  & (orient2d(pts, z, a, q[g]) <= 0))
    return out


def in_circumcircle(pts, tv, s, inf):
    """Sign of "vertex ``s[i]`` is inside the circumcircle of triangle ``tv[i]``".

    For a ghost ``(a, b, INF)`` the circumcircle degenerates to the open half-plane
    outside hull edge ``a -> b``, so the test becomes an orientation. The vertex at
    infinity is outside every finite circumcircle.
    """
    ghost = (tv == inf).any(axis=1)
    s_inf = s == inf
    out = np.full(len(tv), -1, dtype=np.int8)

    finite = ~ghost & ~s_inf
    if finite.any():
        f = np.where(finite)[0]
        out[f] = incircle(pts, tv[f, 0], tv[f, 1], tv[f, 2], s[f])

    gq = ghost & ~s_inf
    if gq.any():
        g = np.where(gq)[0]
        gv = _rotate_inf_last(tv[g], inf)
        out[g] = orient2d(pts, gv[:, 0], gv[:, 1], s[g])
    return out


# --------------------------------------------------------------------- triangulation


class _Mesh:
    """Triangles plus edge adjacency, as flat arrays.

    ``nbr[t, i]`` encodes the neighbour across the edge opposite vertex ``tri[t, i]`` as
    ``u * 3 + j``, where ``j`` is the reciprocal slot in ``u``. Encoding the slot makes the
    reciprocal update O(1), which matters once this is a kernel: a search for "which of
    u's three edges faces me" is three divergent branches per thread.

    Ghost triangles close the plane, so every edge always has a neighbour.
    """

    def __init__(self, tri, nbr):
        self.tri = tri
        self.nbr = nbr

    @property
    def n_tri(self):
        return len(self.tri)

    def grow(self, k):
        """Append ``k`` uninitialised triangles, returning their indices."""
        base = len(self.tri)
        self.tri = np.vstack([self.tri, np.zeros((k, 3), dtype=np.int64)])
        self.nbr = np.vstack([self.nbr, np.full((k, 3), -1, dtype=np.int64)])
        return np.arange(base, base + k)

    def link(self, code_a, code_b):
        """Make the two half-edges point at each other."""
        self.nbr[code_a // 3, code_a % 3] = code_b
        self.nbr[code_b // 3, code_b % 3] = code_a


def check_mesh(mesh, pts, inf, where=""):
    """Assert the mesh is a well-formed triangulation of the plane. Debug aid, not a test.

    Adjacency corruption is the failure mode that does not announce itself: the flip loop
    reports zero remaining bad edges because it is walking links that no longer describe
    the triangles, and only an independent pass over the vertex lists notices.
    """
    tri, nbr = mesh.tri, mesh.nbr
    m = len(tri)
    problems = []

    if (nbr < 0).any():
        problems.append(f"{int((nbr < 0).sum())} unlinked half-edges")
    t = np.repeat(np.arange(m), 3)
    i = np.tile(np.arange(3), m)
    code = nbr[t, i]
    u, j = code // 3, code % 3
    back = nbr[u, j]
    bad = back != t * 3 + i
    if bad.any():
        problems.append(f"{int(bad.sum())} non-reciprocal links")

    # the shared edge must be the same two vertices, traversed in opposite directions
    e0, e1 = tri[t, (i + 1) % 3], tri[t, (i + 2) % 3]
    f0, f1 = tri[u, (j + 1) % 3], tri[u, (j + 2) % 3]
    mism = (e0 != f1) | (e1 != f0)
    if mism.any():
        problems.append(f"{int(mism.sum())} links joining different edges")

    finite = np.where(~(tri == inf).any(axis=1))[0]
    if len(finite):
        o = orient2d(pts, tri[finite, 0], tri[finite, 1], tri[finite, 2])
        if (o <= 0).any():
            problems.append(f"{int((o <= 0).sum())} finite triangles not counter-clockwise "
                            f"({int((o == 0).sum())} degenerate)")
    if problems:
        raise AssertionError(f"mesh corrupt{' at ' + where if where else ''}: "
                             + "; ".join(problems))


def _seed(pts, inf):
    """Initial mesh: one finite triangle and the three ghosts that close the plane."""
    n = len(pts)
    a = 0
    b = next(i for i in range(1, n) if (pts[i] != pts[a]).any())
    c = next((i for i in range(b + 1, n)
              if orient2d(pts, [a], [b], [i])[0] != 0), None)
    if c is None:
        raise ValueError("all points are collinear; no triangulation exists")
    if orient2d(pts, [a], [b], [c])[0] < 0:
        a, c = c, a

    tri = np.array([[a, b, c],          # 0: the finite seed, counter-clockwise
                    [c, b, inf],        # 1: ghost across edge (b, c)
                    [a, c, inf],        # 2: ghost across edge (c, a)
                    [b, a, inf]],       # 3: ghost across edge (a, b)
                   dtype=np.int64)
    nbr = np.full((4, 3), -1, dtype=np.int64)
    m = _Mesh(tri, nbr)
    m.link(0 * 3 + 0, 1 * 3 + 2)        # seed's (b,c) edge <-> ghost 1
    m.link(0 * 3 + 1, 2 * 3 + 2)
    m.link(0 * 3 + 2, 3 * 3 + 2)
    m.link(1 * 3 + 0, 3 * 3 + 1)        # ghost ring: (b,INF)
    m.link(1 * 3 + 1, 2 * 3 + 0)        # (INF,c)
    m.link(2 * 3 + 1, 3 * 3 + 0)        # (INF,a)
    return m, (a, b, c)


def _circumcentres(pts, tv, inf):
    """Float circumcentres; only a heuristic for insertion order, so precision is free.

    Ghost and degenerate triangles get NaN, which makes every distance NaN and falls the
    selection back to lowest index.
    """
    p = pts.astype(np.float64)
    ok = ~(tv == inf).any(axis=1)
    out = np.full((len(tv), 2), np.nan)
    if not ok.any():
        return out
    v = tv[ok]
    a, b, c = p[v[:, 0]], p[v[:, 1]], p[v[:, 2]]
    d = 2.0 * (a[:, 0] * (b[:, 1] - c[:, 1])
               + b[:, 0] * (c[:, 1] - a[:, 1])
               + c[:, 0] * (a[:, 1] - b[:, 1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        aa, bb, cc = (a ** 2).sum(1), (b ** 2).sum(1), (c ** 2).sum(1)
        ux = (aa * (b[:, 1] - c[:, 1]) + bb * (c[:, 1] - a[:, 1])
              + cc * (a[:, 1] - b[:, 1])) / d
        uy = (aa * (c[:, 0] - b[:, 0]) + bb * (a[:, 0] - c[:, 0])
              + cc * (b[:, 0] - a[:, 0])) / d
    out[ok] = np.column_stack([ux, uy])
    return out


def _choose_one_per_triangle(pts, mesh, cand, loc, inf):
    """One point per triangle: the one nearest that triangle's circumcentre.

    The paper's choice, and not arbitrary — inserting the point nearest the circumcentre
    maximises how much of the lifted hull each insertion claims, so fewer edges come out
    non-Delaunay and the next batch starts from a better triangulation.

    Returns the chosen triangles, the chosen points, and the positions of those points
    within ``cand`` so the caller can carry per-candidate data along.
    """
    empty = np.zeros(0, dtype=np.int64)
    if len(cand) == 0:
        return empty, empty, empty

    cc = _circumcentres(pts, mesh.tri, inf)
    tri_of = loc[cand]
    d2 = ((pts[cand].astype(np.float64) - cc[tri_of]) ** 2).sum(axis=1)
    d2 = np.where(np.isnan(d2), np.inf, d2)

    # group by triangle, then take the nearest point in each group; the point index is
    # the final tiebreak so the choice is deterministic and does not depend on ordering
    order = np.lexsort((cand, d2, tri_of))
    grouped = tri_of[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = grouped[1:] != grouped[:-1]
    return grouped[first], cand[order][first], order[first]


def _split(pts, mesh, tri_idx, pt_idx, loc, active, inf):
    """Split each selected triangle on its selected point, all at once.

    Triangle ``t = (v0, v1, v2)`` becomes ``(v0, v1, p)``, ``(v1, v2, p)``, ``(v2, v0, p)``.
    The first child reuses ``t``'s slot so indices stay stable for everything pointing at
    it; the other two are appended.
    """
    k = len(tri_idx)
    if k == 0:
        return
    m0 = mesh.n_tri
    old = mesh.tri[tri_idx].copy()
    old_nbr = mesh.nbr[tri_idx].copy()
    new = mesh.grow(2 * k)
    c0, c1, c2 = tri_idx, new[:k], new[k:]

    v0, v1, v2 = old[:, 0], old[:, 1], old[:, 2]
    mesh.tri[c0] = np.column_stack([v0, v1, pt_idx])
    mesh.tri[c1] = np.column_stack([v1, v2, pt_idx])
    mesh.tri[c2] = np.column_stack([v2, v0, pt_idx])

    # Two adjacent triangles can be split in the same round, and then a neighbour's old
    # half-edge code names a slot that no longer holds that edge. Every outward reference
    # therefore goes through a remap from old half-edge to new, identity for triangles
    # this round left alone. Writing the reciprocal through the stale code instead is
    # silent — the mesh stays well-formed and is simply wrong.
    remap = np.arange(3 * m0, dtype=np.int64)
    remap[tri_idx * 3 + 0] = c1 * 3 + 2          # parent edge (v1,v2) -> child1
    remap[tri_idx * 3 + 1] = c2 * 3 + 2          # parent edge (v2,v0) -> child2
    remap[tri_idx * 3 + 2] = c0 * 3 + 2          # parent edge (v0,v1) -> child0
    out0, out1, out2 = (remap[old_nbr[:, 2]], remap[old_nbr[:, 0]], remap[old_nbr[:, 1]])

    # Each child keeps one of the parent's outer edges and gains two internal ones.
    # Child (v0,v1,p): opp v0 -> (v1,p) = child1's (p,v1); opp v1 -> (p,v0) = child2's;
    #                  opp p  -> (v0,v1) = the parent's edge opposite v2.
    mesh.nbr[c0] = np.column_stack([c1 * 3 + 1, c2 * 3 + 0, out0])
    mesh.nbr[c1] = np.column_stack([c2 * 3 + 1, c0 * 3 + 0, out1])
    mesh.nbr[c2] = np.column_stack([c0 * 3 + 1, c1 * 3 + 0, out2])
    for child, code in ((c0, out0), (c1, out1), (c2, out2)):
        mesh.nbr[code // 3, code % 3] = child * 3 + 2

    active[pt_idx] = False
    _rehome(pts, mesh, (c0, c1, c2), loc, active, inf)


def _walk(pts, mesh, idx, loc, inf, max_steps=10000):
    """Repair the location of points that are no longer in the triangle recorded for them.

    A visibility walk: step across whichever edge the point lies outside of until the
    triangle containing it is reached. Vectorised over all stragglers at once.

    This exists because the ghost triangles do not partition the plane — a point beyond a
    hull *vertex* is outside two hull edges at once, so it belongs to two ghosts. After a
    hull flip its recorded ghost can vanish while the other one, untouched by the flip,
    still holds it. Re-homing against the flip's own outputs therefore misses it, and the
    point silently keeps a stale triangle until something splits on it from the outside.
    """
    if len(idx) == 0:
        return
    todo = np.asarray(idx)
    for _ in range(max_steps):
        inside = tri_contains(pts, mesh, loc[todo], todo, inf)
        todo = todo[~inside]
        if len(todo) == 0:
            return
        tv = mesh.tri[loc[todo]]
        ghost = (tv == inf).any(axis=1)
        step = np.zeros(len(todo), dtype=np.int64)

        if ghost.any():
            g = np.where(ghost)[0]
            k = np.argmax(tv[g] == inf, axis=1)
            a, b = tv[g, (k + 1) % 3], tv[g, (k + 2) % 3]
            # outside this hull edge but rejected by the wedge test: the point belongs
            # further back along the visible chain, so walk the ghost ring instead of
            # dropping into the interior and having to come back out
            back = orient2d(pts, a, b, todo[g]) > 0
            step[g] = np.where(back, (k + 2) % 3, k)
        if (~ghost).any():
            f = np.where(~ghost)[0]
            o = np.stack([orient2d(pts, tv[f, (k + 1) % 3], tv[f, (k + 2) % 3], todo[f])
                          for k in range(3)], axis=1)
            step[f] = np.argmin(o, axis=1)
        loc[todo] = mesh.nbr[loc[todo], step] // 3
    raise RuntimeError(f"point location walk did not settle for {len(todo)} points")


def _rehome(pts, mesh, children, loc, active, inf):
    """Move each uninserted point into whichever of its parent's children now holds it.

    Tests containment against all candidates rather than deducing the child from the
    wedge the point falls in around the split vertex. The wedge test is cheaper — three
    orientations instead of up to nine — but it needs a special case for the wedge bounded
    by the ray to infinity, and getting that subtly wrong loses points silently rather
    than loudly. Worth revisiting when this becomes a kernel; correctness first.
    """
    group = np.full(mesh.n_tri, -1, dtype=np.int64)
    for c in children:
        group[c] = np.arange(len(c))
    moving = np.where(active & (group[loc] >= 0))[0]
    if len(moving) == 0:
        return
    row = group[loc[moving]]
    settled = np.zeros(len(moving), dtype=bool)
    for cand in children:
        hit = tri_contains(pts, mesh, cand[row], moving, inf)
        loc[moving[hit]] = cand[row][hit]
        settled |= hit
    _walk(pts, mesh, moving[~settled], loc, inf)


def _edge_slot(pts, mesh, tri_of, pt, inf):
    """For each (triangle, contained point), the slot whose edge the point lies on, or -1.

    Exactly-on-an-edge is not an exotic case: it is what a lattice produces. Three
    collinear points on a square grid put the middle one exactly on the edge joining the
    other two, and splitting that triangle 1->3 makes a zero-area child, which then
    poisons every orientation test that touches it.
    """
    out = np.full(len(tri_of), -1, dtype=np.int64)
    tv = mesh.tri[tri_of]
    fin = np.where(~(tv == inf).any(axis=1))[0]
    if len(fin) == 0:
        return out
    o = np.stack([orient2d(pts, tv[fin, (k + 1) % 3], tv[fin, (k + 2) % 3], pt[fin])
                  for k in range(3)], axis=1)
    on = (o == 0).any(axis=1)
    out[fin[on]] = np.argmax(o[on] == 0, axis=1)
    return out


def _edge_split(pts, mesh, t, slot, pt_idx, loc, active, inf):
    """Insert points that lie exactly on an edge, splitting both adjacent triangles: 2 -> 4.

    ``t = (r, a, b)`` with the point ``p`` on edge ``(a, b)``, and ``u = (s, b, a)`` across
    it, become ``(r, a, p)``, ``(r, p, b)``, ``(s, b, p)``, ``(s, p, a)``. A ghost partner
    needs no special case: ``s`` is then the vertex at infinity and two of the four
    children are ghosts, which is exactly right for a point landing on a hull edge.
    """
    if len(t) == 0:
        return
    m0 = mesh.n_tri
    code = mesh.nbr[t, slot]
    u, j = code // 3, code % 3

    # one edge split at a time per triangle, and never two that share a triangle
    cand = np.arange(len(t))
    owner = np.full(m0, len(t), dtype=np.int64)
    np.minimum.at(owner, t, cand)
    np.minimum.at(owner, u, cand)
    go = (owner[t] == cand) & (owner[u] == cand)
    t, slot, u, j, pt_idx = t[go], slot[go], u[go], j[go], pt_idx[go]
    if len(t) == 0:
        return

    r = mesh.tri[t, slot]
    a = mesh.tri[t, (slot + 1) % 3]
    b = mesh.tri[t, (slot + 2) % 3]
    s = mesh.tri[u, j]
    X = mesh.nbr[t, (slot + 2) % 3]          # t's edge (r, a)... see mapping below
    Y = mesh.nbr[t, (slot + 1) % 3]
    Z = mesh.nbr[u, (j + 1) % 3]
    W = mesh.nbr[u, (j + 2) % 3]

    k = len(t)
    new = mesh.grow(2 * k)
    T1, T2, T3, T4 = t, u, new[:k], new[k:]

    remap = np.arange(3 * m0, dtype=np.int64)
    remap[t * 3 + (slot + 1) % 3] = T2 * 3 + 1      # t's edge (b, r)
    remap[t * 3 + (slot + 2) % 3] = T1 * 3 + 2      # t's edge (r, a)
    remap[u * 3 + (j + 1) % 3] = T4 * 3 + 1         # u's edge (a, s)
    remap[u * 3 + (j + 2) % 3] = T3 * 3 + 2         # u's edge (s, b)
    X, Y, Z, W = remap[X], remap[Y], remap[Z], remap[W]

    mesh.tri[T1] = np.column_stack([r, a, pt_idx])
    mesh.tri[T2] = np.column_stack([r, pt_idx, b])
    mesh.tri[T3] = np.column_stack([s, b, pt_idx])
    mesh.tri[T4] = np.column_stack([s, pt_idx, a])
    mesh.nbr[T1] = np.column_stack([T4 * 3 + 0, T2 * 3 + 2, X])
    mesh.nbr[T2] = np.column_stack([T3 * 3 + 0, Y, T1 * 3 + 1])
    mesh.nbr[T3] = np.column_stack([T2 * 3 + 0, T4 * 3 + 2, W])
    mesh.nbr[T4] = np.column_stack([T1 * 3 + 0, Z, T3 * 3 + 1])
    for outer, target in ((X, T1 * 3 + 2), (Y, T2 * 3 + 1),
                          (W, T3 * 3 + 2), (Z, T4 * 3 + 1)):
        mesh.nbr[outer // 3, outer % 3] = target

    active[pt_idx] = False
    _rehome(pts, mesh, (T1, T2, T3, T4), loc, active, inf)


def select_flips(pts, mesh, inf):
    """Half-edges to flip this round: non-Delaunay, and free of conflicts.

    Two flips conflict when they share a triangle, so each candidate claims both of its
    triangles by scatter-min on the half-edge index and proceeds only if it still owns
    both — a lock-free independent set. The half-edge index is the claim token rather than
    a compacted candidate id specifically so that this maps onto ``atomic_fetch_min`` in
    :mod:`.gpu` with the same tie-break and no stream compaction.
    """
    m = mesh.n_tri
    t = np.repeat(np.arange(m), 3)
    slot = np.tile(np.arange(3), m)
    half = t * 3 + slot
    code = mesh.nbr[t, slot]
    u = code // 3

    keep = t < u                       # visit each edge once, from the lower triangle
    t, u, half = t[keep], u[keep], half[keep]
    if len(t) == 0:
        return np.zeros(0, dtype=np.int64)

    s = mesh.tri[u, code[keep] % 3]
    # a tie (exactly cocircular) is left alone: every choice is Delaunay, and flipping on
    # ties is how a flip loop starts cycling
    bad = in_circumcircle(pts, mesh.tri[t], s, inf) > 0
    t, u, half = t[bad], u[bad], half[bad]
    if len(t) == 0:
        return np.zeros(0, dtype=np.int64)

    owner = np.full(m, np.iinfo(np.int32).max, dtype=np.int64)
    np.minimum.at(owner, t, half)
    np.minimum.at(owner, u, half)
    return half[(owner[t] == half) & (owner[u] == half)]


def _flip_round(pts, mesh, loc, active, inf, *, backend="cpu"):
    """Flip every non-Delaunay edge that can be flipped without conflicting.

    Returns the number of flips performed. ``backend="gpu"`` runs the scan and the
    conflict resolution as Metal kernels; the selection is identical either way, so the
    resulting mesh is bit-for-bit the same.
    """
    if backend == "gpu":
        from .gpu import flip_round as gpu_flip_round
        tri_new, nbr_new, k, (ft, fu) = gpu_flip_round(pts, mesh.tri, mesh.nbr, inf)
        if k == 0:
            return 0
        mesh.tri, mesh.nbr = tri_new, nbr_new
        # exactly the CPU path's re-homing, not merely an equivalent one. Falling back to
        # a bare walk here instead diverges on tie-heavy input: a point sitting on the
        # boundary between two triangles is legitimately in either, the walk and the
        # candidate test disagree about which, and the two backends then insert in
        # different orders and return different — both valid — triangulations.
        _rehome(pts, mesh, (ft, fu), loc, active, inf)
        return k

    m = mesh.n_tri
    half = select_flips(pts, mesh, inf)
    if len(half) == 0:
        return 0
    t, slot = half // 3, half % 3
    code = mesh.nbr[t, slot]
    u, j = code // 3, code % 3

    r = mesh.tri[t, slot]
    p = mesh.tri[t, (slot + 1) % 3]
    q = mesh.tri[t, (slot + 2) % 3]
    sv = mesh.tri[u, j]

    A = mesh.nbr[t, (slot + 1) % 3]          # t's edge (q, r)
    B = mesh.nbr[t, (slot + 2) % 3]          # t's edge (r, p)
    C = mesh.nbr[u, (j + 1) % 3]             # u's edge (p, s)
    D = mesh.nbr[u, (j + 2) % 3]             # u's edge (s, q)

    # Conflict resolution keeps two flips off the same triangle, but a flip's *outer*
    # neighbour may itself be flipping, and then the edge we share with it has moved to
    # its partner. Same remap as in _split: old half-edge -> where that edge lives now.
    # (r,p) stays on t'; (q,r) moves to u'; (p,s) moves to t'; (s,q) stays on u'.
    remap = np.arange(3 * m, dtype=np.int64)
    remap[t * 3 + (slot + 1) % 3] = u * 3 + 1
    remap[t * 3 + (slot + 2) % 3] = t * 3 + 2
    remap[u * 3 + (j + 1) % 3] = t * 3 + 0
    remap[u * 3 + (j + 2) % 3] = u * 3 + 0
    A, B, C, D = remap[A], remap[B], remap[C], remap[D]

    mesh.tri[t] = np.column_stack([r, p, sv])
    mesh.tri[u] = np.column_stack([r, sv, q])
    mesh.nbr[t] = np.column_stack([C, u * 3 + 2, B])
    mesh.nbr[u] = np.column_stack([D, A, t * 3 + 1])
    for outer, target in ((C, t * 3 + 0), (B, t * 3 + 2),
                          (D, u * 3 + 0), (A, u * 3 + 1)):
        mesh.nbr[outer // 3, outer % 3] = target

    _rehome(pts, mesh, (t, u), loc, active, inf)
    return len(t)


def triangulate(points, *, max_rounds: int = 10000, return_info: bool = False,
                backend: str = "cpu"):
    """Delaunay triangulation of ``points`` (n, 2), by exact-predicate batch insertion.

    Returns an ``(m, 3)`` array of vertex indices into the *input* order. With
    ``return_info`` also returns the conditioning diagnostics and per-phase counters.
    ``backend="gpu"`` runs the flipping scan and conflict resolution as Metal kernels and
    produces an identical result; everything else still runs on the host.
    """
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")
    raw = np.asarray(points, dtype=np.float64)
    if len(raw) < 3:
        raise ValueError("a triangulation needs at least 3 points")

    ipts, info = condition_points(raw)
    order = hilbert_order(ipts)
    pts = ipts[order]
    n = len(pts)
    inf = n

    if info["n_distinct"] < n:
        raise ValueError(
            f"{n - info['n_distinct']} duplicate points after conditioning; "
            "de-duplicate before triangulating")

    mesh, seed = _seed(pts, inf)
    loc = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    active[list(seed)] = False
    _walk(pts, mesh, np.where(active)[0], loc, inf)

    rounds = flips = edge_splits = 0
    while active.any():
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(f"insertion did not converge in {max_rounds} rounds")

        # Points lying exactly on an edge cannot be inserted by a 1->3 split, so prefer
        # the ones strictly inside a triangle; the rest are handled below, by which time
        # earlier insertions have often flipped the offending edge away anyway.
        idx = np.where(active)[0]
        eslot = _edge_slot(pts, mesh, loc[idx], idx, inf)
        interior = idx[eslot < 0]

        if len(interior):
            sel_t, sel_p, _ = _choose_one_per_triangle(pts, mesh, interior, loc, inf)
            _split(pts, mesh, sel_t, sel_p, loc, active, inf)
        else:
            on_edge = idx[eslot >= 0]
            sel_t, sel_p, pos = _choose_one_per_triangle(pts, mesh, on_edge, loc, inf)
            if len(sel_t) == 0:
                raise RuntimeError("points remain but none is inside any triangle")
            _edge_split(pts, mesh, sel_t, eslot[eslot >= 0][pos], sel_p, loc, active, inf)
            edge_splits += len(sel_t)

        for _ in range(max_rounds):
            k = _flip_round(pts, mesh, loc, active, inf, backend=backend)
            flips += k
            if k == 0:
                break
        else:
            raise RuntimeError("flipping did not converge")

    finite = ~(mesh.tri == inf).any(axis=1)
    out = order[mesh.tri[finite]]
    if return_info:
        return out, {**info, "rounds": rounds, "flips": flips,
                     "edge_splits": edge_splits, "n_triangles": int(finite.sum())}
    return out
