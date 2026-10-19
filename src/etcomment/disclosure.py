"""Exact search for OpenET values that published statistics reveal jointly (scripts/14).

Each published statistic is a row of weights on the unknown OpenET values (one column per value; nonlinear
statistics enter through their gradients at the true values).  A combination of the rows that is zero outside
a set U of site-months reveals a combination (a sum, a difference, a single value) of the values in U.  The
OpenET rule of this repository: no such U with fewer than K = 6 site-months.

:func:`small_support_search` decides this exactly (up to the rank tolerance), fast enough for the joint
system of every deposited file, where :func:`etcomment.openet_compare.small_supports` (enumeration) is not:

* reduction (:func:`reduce`): columns that are parallel enter every combination in a fixed ratio; a class of
  them spanning K or more site-months must get weight 0 in a small combination, which restricts the
  combinations; repeated until nothing changes;
* matroid facts on the reduced columns: the support of every combination contains a cocircuit, and a cocircuit
  meets every basis.  With m bases on pairwise disjoint site-months, a support of at most K - 1 site-months
  meets the first basis T1 in at most K - m of them (m >= K: nothing to find);
* in the coordinates of T1 (R = Q_T1^-1 Q, R_T1 = I) a combination y = c^T R has y_T1 = c, so every set S of
  at most K - m site-months of T1 is tried as supp(c) (depth-first, with a necessary-condition prune), solving
  for c that vanishes outside S and at most K - 1 - |S| further site-months (:func:`_solve`);
* every candidate is confirmed by a rank test on the original rows (dropping its columns lowers the rank).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.linalg import null_space

MIN_SITE_MONTHS = 6  # K: combinations over fewer site-months must not be revealed
SIG_MIN = 1e-6  # columns count as a basis only if the smallest singular value of Q[:, basis] is >= SIG_MIN


def _rank(a: np.ndarray, rtol: float) -> int:
    if a.size == 0:
        return 0
    s = np.linalg.svd(a, compute_uv=False)
    return int((s > s[0] * rtol).sum()) if s[0] > 0 else 0


def parallel_classes(p: np.ndarray, rtol: float) -> list[list[int]]:
    """Indices of parallel columns of ``p`` (no zero columns); rounding may split a class, never merge two."""
    u = p / np.linalg.norm(p, axis=0)
    lead = np.argmax(np.abs(u), axis=0)
    u = u * np.sign(u[lead, np.arange(u.shape[1])])
    buckets: dict = {}
    for j in range(u.shape[1]):
        buckets.setdefault(tuple(np.round(u[:, j], 7)), []).append(j)
    out = []
    for c in buckets.values():
        rest = list(c)
        while rest:
            same = [j for j in rest if np.abs(u[:, j] - u[:, rest[0]]).max() <= max(rtol, 1e-12) * 1e3]
            out.append(same)
            rest = [j for j in rest if j not in same]
    return out


def reduce(a: np.ndarray, groups: list, k: int = MIN_SITE_MONTHS, rtol: float = 1e-9):
    """Columns still free in a combination over fewer than ``k`` groups, and an orthonormal basis of the row
    space of the reduced rows: (alive column indices, Q of shape d x len(alive), d)."""
    a = a[np.abs(a).sum(axis=1) > 0]
    if not a.size:
        return np.arange(0), np.zeros((0, 0)), 0
    a = a / np.abs(a).max(axis=1, keepdims=True)
    basis = np.eye(a.shape[0])
    alive = np.flatnonzero(np.abs(a).sum(axis=0) > 0)
    while alive.size and basis.shape[1]:
        p = basis.T @ a[:, alive]
        norm = np.linalg.norm(p, axis=0)
        ok = norm > rtol * max(1.0, float(norm.max(initial=0.0)))
        alive, p = alive[ok], p[:, ok]
        if not alive.size:
            break
        big = [c for c in parallel_classes(p, rtol) if len({groups[alive[j]] for j in c}) >= k]
        if not big:
            break
        basis = basis @ null_space(np.concatenate([p[:, c] for c in big], axis=1).T, rcond=rtol)
        drop = {j for c in big for j in c}
        alive = alive[[j for j in range(alive.size) if j not in drop]]
    if not alive.size or not basis.shape[1]:
        return alive, np.zeros((0, alive.size)), 0
    _, s, vt = np.linalg.svd(basis.T @ a[:, alive], full_matrices=False)
    d = int((s > s[0] * rtol).sum()) if s.size and s[0] > 0 else 0
    return alive, vt[:d], d


def greedy_basis(q: np.ndarray, cols: list, d: int, thr: float = SIG_MIN) -> list | None:
    """``d`` columns of ``q`` picked in order of ``cols`` (Gram-Schmidt with reorthogonalization), accepted only
    if their smallest singular value is >= ``thr``; None if there are none."""
    basis, u = [], np.zeros((q.shape[0], 0))
    for j in cols:
        v = q[:, j].copy()
        for _ in range(2):
            v -= u @ (u.T @ v)
        nv = np.linalg.norm(v)
        if nv > thr:
            basis.append(j)
            u = np.column_stack([u, v / nv])
            if len(basis) == d:
                return basis if np.linalg.svd(q[:, basis], compute_uv=False)[-1] >= thr else None
    return None


def _solve(x: np.ndarray, n: np.ndarray, byg: dict, budget: int, out: list) -> None:
    """Directions c in span(n) with c^T x zero on the columns of all but <= ``budget`` groups of ``byg``
    ({group: columns}); appends (c, or n for any c in its span, set of groups left nonzero) to ``out``."""
    if n.shape[1] == 0:
        return
    proj = {gg: n.T @ x[:, cols] for gg, cols in byg.items()}
    live = {gg: cols for gg, cols in byg.items() if np.abs(proj[gg]).max() > 1e-8 * max(1.0, np.abs(x).max())}
    if len(live) <= budget:
        out.append((n, set(live)))
        return
    order = sorted(live, key=str)
    for v in itertools.combinations(order[: budget + n.shape[1]], min(len(order), budget + n.shape[1]) - budget):
        nb = null_space(np.concatenate([proj[gg] for gg in v], axis=1).T, rcond=1e-8)  # all of v annihilated
        if nb.shape[1] == 0:
            continue
        n2 = n @ nb
        rest = {gg: live[gg] for gg in live if gg not in v}
        if n2.shape[1] == 1:
            y = n2[:, 0] @ x
            left = {gg for gg, cols in rest.items() if np.abs(y[cols]).max() > 1e-8 * max(1e-300, np.abs(y).max())}
            if len(left) <= budget:
                out.append((n2[:, 0], left))
        else:
            _solve(x, n2, rest, budget, out)


def minimal(sets) -> list[tuple]:
    """The sets of ``sets`` that contain no other one, as sorted tuples (smallest first)."""
    s = {frozenset(x) for x in sets}
    keep = [x for x in s if not any(y < x for y in s)]
    return sorted((tuple(sorted(x, key=str)) for x in keep), key=lambda c: (len(c), str(c)))


def small_support_search(a: np.ndarray, groups: list, k: int = MIN_SITE_MONTHS, rtol: float = 1e-9,
                         tries: int = 300, seed: int = 1, zero_tol: float = 1e-8, max_m: int | None = None) -> dict:
    """Every minimal set of fewer than ``k`` groups on which a combination of the rows of ``a`` lives.

    ``a`` holds one row per published functional and one column per unknown, ``groups[j]`` the site-month of
    column j.  Returns cols_alive and rank (after :func:`reduce`), m (disjoint bases found; m >= k certifies
    that nothing can be found), tried (sets of the first basis tried), candidates and confirmed (minimal sets,
    confirmed by a rank test on ``a``).  ``max_m`` caps m (a slower, wider search, as a cross-check).  Raises
    RuntimeError if no well-conditioned basis is found (nothing can be decided then).
    """
    alive, q, d = reduce(np.asarray(a, float), groups, k, rtol)
    out = {"cols_alive": int(alive.size), "rank": d, "m": 0, "tried": 0, "candidates": [], "confirmed": []}
    if d == 0:
        out["m"] = k
        return out
    g = [groups[j] for j in alive]
    rng = np.random.default_rng(seed)
    lev = (q ** 2).sum(axis=0)
    best: list = []
    for t in range(tries):  # bases on pairwise disjoint groups, packed greedily in random orders
        order = (list(np.argsort(lev + rng.uniform(0, 1.0, lev.size))) if t % 2 == 0
                 else list(rng.permutation(alive.size)))
        used, bases = set(), []
        for _ in range(k):
            b = greedy_basis(q, [j for j in order if g[j] not in used], d)
            if b is None:
                break
            bases.append(b)
            used |= {g[j] for j in b}
        if len(bases) > len(best):
            best = bases
        if len(best) >= k:
            break
    best = best[:max_m] if max_m is not None else best
    out["m"] = m = len(best)
    out["sigma_min_bases"] = [float(np.linalg.svd(q[:, b], compute_uv=False)[-1]) for b in best]
    if m >= k:
        return out
    if m == 0:
        raise RuntimeError("small_support_search: no well-conditioned basis of the reduced rows")
    t1 = best[int(np.argmin([np.linalg.cond(q[:, b]) for b in best]))]
    r = np.linalg.solve(q[:, t1], q)  # d x len(alive), r[:, t1] = I
    r[np.abs(r) < zero_tol * np.abs(r).max()] = 0.0
    gid = {x: i for i, x in enumerate(dict.fromkeys(g))}
    gcol = np.array([gid[x] for x in g])
    t1g = [gcol[j] for j in t1]
    inc = np.zeros((d, len(gid)), dtype=np.int32)  # groups each row of r touches outside its own T1 group
    for i in range(d):
        inc[i, np.unique(gcol[np.flatnonzero(r[i])])] = 1
        inc[i, t1g[i]] = 0
    size = inc.sum(axis=1)
    ov = inc @ inc.T
    np.fill_diagonal(ov, 0)
    cands: set = set()

    def check(rows: list[int]) -> None:
        out["tried"] += 1
        x = r[rows]
        sg = {g[t1[i]] for i in rows}
        byg: dict = {}
        for j in np.flatnonzero(np.abs(x).sum(axis=0) > 0):
            if g[j] not in sg:
                byg.setdefault(g[j], []).append(j)
        sols: list = []
        _solve(x, np.eye(len(rows)), byg, k - 1 - len(sg), sols)
        cands.update(frozenset(sg) | frozenset(left) for _, left in sols)

    def dfs(s: list[int], start: int, smax: int) -> None:
        if len(s) == smax:
            # necessary: a group outside S touched by only one row of S stays in the support
            budget = k - 1 - smax
            sub = inc[s]
            cover = sub.sum(axis=0)
            for a_, i in enumerate(s):
                private = int(((sub[a_] == 1) & (cover == 1)).sum())
                private -= int(sum(sub[a_, t1g[o]] for o in s if o != i and cover[t1g[o]] == 1))
                if private > budget:
                    return
            check(s)
            return
        rest, budget = smax - len(s) - 1, k - 1 - smax
        for i in range(start, d):
            s2 = s + [i]
            for a_ in s2:  # upper bound on the groups of row a_ the other rows of S can cover
                have = sum(ov[a_, o] for o in s2 if o != a_)
                cand = ov[a_, i + 1:]
                extra = np.sort(cand)[-rest:].sum() if rest > 0 and cand.size else 0
                if size[a_] - have - extra > budget + smax:
                    break
            else:
                dfs(s2, i + 1, smax)

    for smax in range(1, k - m + 1):
        dfs([], 0, smax)
    out["candidates"] = sorted((tuple(sorted(c, key=str)) for c in cands), key=lambda c: (len(c), str(c)))
    if cands:
        an = np.asarray(a, float)
        an = an[np.abs(an).sum(axis=1) > 0]
        an = an / np.abs(an).max(axis=1, keepdims=True)
        full = _rank(an, rtol)
        for c in out["candidates"]:
            keep = [j for j in range(an.shape[1]) if groups[j] not in set(c)]
            if _rank(an[:, keep], rtol) < full:
                out["confirmed"].append(c)
    out["confirmed"] = minimal(out["confirmed"])
    return out
