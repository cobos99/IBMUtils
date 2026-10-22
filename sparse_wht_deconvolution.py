"""
sparse_wht_deconvolution.py

Compute   y = ifhwt( fhwt(x) / fhwt(a) )

for x, a given as SPARSE signals over the Boolean cube {0,1}^n (n can be
huge -- e.g. n = 100, so N = 2^n) under the assumption that the true y
is itself sparse. No length-2^n array is ever materialized.

Signals are represented as dict[int, float]: {index_bitmask: value}.

------------------------------------------------------------------------
Algorithm
------------------------------------------------------------------------
fhwt(s)_k = sum_i s_i * (-1)^popcount(i & k) can be evaluated at any ONE
frequency k in O(len(s)) time. That gives cheap "oracle" query access to

    r(k) = fhwt(x)_k / fhwt(a)_k

the (dense but query-able) Hadamard-domain function whose sparse inverse
transform is exactly the y we want. Recovering a K-sparse signal from
query access to its Hadamard transform is the "sparse Walsh-Hadamard
transform" problem (Li & Ramchandran 2015; Scheibler-Haghighatshoar-
Vetterli 2014; Amrollahi et al. 2019). This is a random-subsampling +
peeling-decoder implementation of it:

  1. Build a random linear folding map M: {0,1}^n -> {0,1}^m (m << n)
     from m random "pivot bits", so it has an exact section sigma
     (sigma(u) places the bits of u at the pivot positions).
  2. Querying r at the 2^m points R(c) = XOR-combinations of M's rows,
     then running a *small* (size 2^m) inverse WHT, gives
        g(u) = sum_{i : M(i)=u} y_i
     i.e. every true nonzero of y aliased into one of 2^m bins.
  3. For a bin holding exactly one true nonzero ("singleton"), the
     missing (n-m) bits of its index i are recovered one at a time:
     re-run the same small transform on r queried at points shifted by
     e_q (unit vector for bit q); the sign of h_q(u)/g(u) is (-1)^{i_q}.
  4. A true singleton has |h_q(u)/g(u)| == 1 for *every* q (it's a pure
     sign flip); a bin with two or more colliding atoms will generically
     fail that magnitude test for some q. So the same bit-flip probes
     double as a free singleton test -- accepted candidates are then
     peeled off (subtracted from the oracle) before the next round.

Rounds repeat with fresh random M until no new atoms are found for
several rounds in a row, or an expected sparsity count is reached.

------------------------------------------------------------------------
Vectorized oracle
------------------------------------------------------------------------
The whole cost of a round is evaluating fhwt(x)/fhwt(a) at a batch of
O(n * 2^m) query points, each requiring an O(|x|) sum. Doing that one
Python-level call at a time is far too slow once |x|, |a| ~ 10,000
(the target scale, n=2^100). Support (up to 100-bit) indices are split
into 64-bit "lo" and 36-bit "hi" halves and all of a round's queries are
issued as ONE batched numpy computation (broadcast AND + popcount +
sign-weighted sum, chunked to bound memory), and the small (2^m-point)
Hadamard transform is done as a vectorized 2D butterfly across every
probe column at once.
"""

import random
import time
from math import log2, ceil

import numpy as np

MASK64 = (1 << 64) - 1


# ----------------------------------------------------------------------
# 1. Array representation + vectorized batch fhwt evaluation
# ----------------------------------------------------------------------

def eval_fhwt(sparse, k):
    """Reference (slow, pure-Python) fhwt(sparse)_k, O(len(sparse)).
    Kept for clarity/testing; the actual solver uses the vectorized
    batch_fhwt below."""
    total = 0.0
    for idx, val in sparse.items():
        total += val if (idx & k).bit_count() % 2 == 0 else -val
    return total


def xor_convolve(a, b):
    """Exact XOR-convolution of two sparse signals. O(|a|*|b|).
    Only meant for building/checking synthetic test cases -- this is
    the expensive, direct way of computing what sparse_deconvolve
    inverts."""
    out = {}
    for ia, va in a.items():
        for ib, vb in b.items():
            k = ia ^ ib
            out[k] = out.get(k, 0.0) + va * vb
    return {k: v for k, v in out.items() if abs(v) > 1e-12}

def to_arrays(sparse):
    """dict[int,float] -> (lo uint64[], hi uint64[], vals float64[])."""
    if not sparse:
        z = np.zeros(0, dtype=np.uint64)
        return z, z, np.zeros(0, dtype=np.float64)
    idxs = np.fromiter(sparse.keys(), dtype=object, count=len(sparse))
    vals = np.fromiter(sparse.values(), dtype=np.float64, count=len(sparse))
    lo = np.array([int(i) & MASK64 for i in idxs], dtype=np.uint64)
    hi = np.array([int(i) >> 64 for i in idxs], dtype=np.uint64)
    return lo, hi, vals

if hasattr(np, "bitwise_count"):
    _popcount = np.bitwise_count
else:  # numpy < 2.0 fallback: SWAR popcount, vectorized
    def _popcount(a):
        a = a.astype(np.uint64)
        m1 = np.uint64(0x5555555555555555)
        m2 = np.uint64(0x3333333333333333)
        m4 = np.uint64(0x0f0f0f0f0f0f0f0f)
        h01 = np.uint64(0x0101010101010101)
        a = a - ((a >> np.uint64(1)) & m1)
        a = (a & m2) + ((a >> np.uint64(2)) & m2)
        a = (a + (a >> np.uint64(4))) & m4
        return ((a * h01) >> np.uint64(56)).astype(np.uint64)

def batch_fhwt(lo, hi, vals, k_lo, k_hi, chunk_elems=40_000_000):
    """Vectorized fhwt(sparse) at every point in k (k_lo/k_hi same shape,
    any shape -- flattened internally). Returns float64 array shaped like
    k_lo. O(len(vals) * k.size), chunked over k to bound memory.

    Signs are kept in int8 as long as possible (only the final matmul
    promotes to float) -- staying in int8 instead of casting the whole
    popcount matrix to float64 early is roughly 2-3x faster in practice.
    """
    shape = k_lo.shape
    kl = k_lo.reshape(-1)
    kh = k_hi.reshape(-1)
    out = np.zeros(kl.shape[0], dtype=np.float64)
    if vals.size == 0 or kl.size == 0:
        return out.reshape(shape)
    chunk = max(1, chunk_elems // max(vals.size, 1))
    for s in range(0, kl.size, chunk):
        e = min(s + chunk, kl.size)
        and_lo = lo[:, None] & kl[None, s:e]
        and_hi = hi[:, None] & kh[None, s:e]
        parity = (_popcount(and_lo) + _popcount(and_hi)) & np.uint64(1)
        signs = 1 - 2 * parity.astype(np.int8)
        out[s:e] = vals @ signs
    return out.reshape(shape)

def split_hi_lo(k_int_list):
    """List[python int] -> (lo uint64[], hi uint64[])."""
    lo = np.array([int(k) & MASK64 for k in k_int_list], dtype=np.uint64)
    hi = np.array([int(k) >> 64 for k in k_int_list], dtype=np.uint64)
    return lo, hi


# ----------------------------------------------------------------------
# 2. Vectorized small (size 2^m) Hadamard transform, many columns at once
# ----------------------------------------------------------------------

def fwht_columns(mat):
    """In-place unnormalized WHT along axis 0 of a 2D array (rows =
    2^m, columns = independent probe sets transformed simultaneously).

    Fully vectorized butterfly: instead of a Python loop over the
    size/(2h) blocks of each stage, one reshape exposes the (top, bot)
    halves of every block along a length-2 axis and the whole stage is a
    single broadcast add/subtract over all blocks and all columns at once
    (numerically identical to the block-loop, just far fewer Python-level
    ops -- the same reshape trick the dense hadamard_transform uses)."""
    size, ncols = mat.shape
    h = 1
    while h < size:
        mat = mat.reshape(size // (2 * h), 2, h, ncols)
        top = mat[:, 0].copy()
        bot = mat[:, 1]
        mat[:, 0] = top + bot
        mat[:, 1] = top - bot
        mat = mat.reshape(size, ncols)
        h *= 2
    return mat

# ----------------------------------------------------------------------
# 3. Oracle over the residual r(k) = fhwt(x)_k / fhwt(a)_k
# ----------------------------------------------------------------------

class ResidualOracle:
    def __init__(self, x, a, zero_tol=1e-9):
        self.x_lo, self.x_hi, self.x_vals = to_arrays(x)
        self.a_lo, self.a_hi, self.a_vals = to_arrays(a)
        self.zero_tol = zero_tol
        self.y_hat = {}          # recovered atoms so far: {index: value}
        self._y_arrays = to_arrays({})
        self._y_dirty = False

    def note_peeled(self, idx, val):
        self.y_hat[idx] = val
        self._y_dirty = True

    def _y_arr(self):
        if self._y_dirty:
            self._y_arrays = to_arrays(self.y_hat)
            self._y_dirty = False
        return self._y_arrays

    def batch_query(self, k_lo, k_hi):
        """r(k) - (already peeled contribution), for every k in the
        (arbitrary-shape) k_lo/k_hi arrays. Returns (r, ok) where ok is
        False if fhwt(a) vanished anywhere in this batch (caller should
        discard/retry the whole round)."""
        fa = batch_fhwt(self.a_lo, self.a_hi, self.a_vals, k_lo, k_hi)
        bad = np.abs(fa) < self.zero_tol
        if bad.any():
            return None, False
        fx = batch_fhwt(self.x_lo, self.x_hi, self.x_vals, k_lo, k_hi)
        r = fx / fa
        if self.y_hat:
            ylo, yhi, yvals = self._y_arr()
            r = r - batch_fhwt(ylo, yhi, yvals, k_lo, k_hi)
        return r, True


# ----------------------------------------------------------------------
# 4. One round: random folding map + singleton detection/recovery
# ----------------------------------------------------------------------

def run_round(oracle, n, m, verify_points=3, tol=1e-6, rng=random,
              mag_tol=0.1, atom_rel_tol=0.05):
    """One random-subsampling round. Returns a list of newly recovered
    (index, value) atoms, [] if none found, or None if the round must
    be discarded (it hit a zero of fhwt(a)).

    mag_tol : how far |h_q(u)/g(u)| may stray from 1 for a bin to still
        count as a singleton. A pure singleton gives exactly 1; two or
        more colliding atoms give something else. On EXACT data any tiny
        value works, but real (finite-shot) data has a dense noise floor
        under the sparse atoms, so a true singleton's ratio is 1 +/- O(
        noise/atom): mag_tol must exceed that relative noise or every
        real atom is wrongly rejected (the old hard-wired 5e-5 did exactly
        that). 0.1 tolerates a ~10% noise-to-atom ratio; raise it for
        noisier data, lower it to reject collisions more aggressively.
    atom_rel_tol : a bin counts as carrying an atom only if |g(u)| exceeds
        atom_rel_tol * max|g| in this round. Being relative to the current
        (already-peeled) residual, it tracks only atoms standing clear of
        the noise floor; smaller atoms surface in later rounds as larger
        ones are peeled away. The old absolute 1e-6*scale admitted the
        whole noise floor as spurious candidate bins."""

    pivots = rng.sample(range(n), m)
    pivot_set = set(pivots)
    others = [b for b in range(n) if b not in pivot_set]

    # rows of M: pivot bit + random mixing over the non-pivot bits
    rows = []
    for p in pivots:
        row = 1 << p
        for b in others:
            if rng.random() < 0.5:
                row |= (1 << b)
        rows.append(row)

    # R[c] = XOR of rows[j] for j with bit j of c set (built by doubling
    # so array index <-> bit convention matches fwht_columns's)
    R = [0]
    for row in rows:
        R = R + [x ^ row for x in R]
    size = len(R)  # 2^m
    r_lo, r_hi = split_hi_lo(R)

    def reconstruct(u, nonpivot_bits):
        # Given u = M(i) and the already-recovered non-pivot bits of the
        # true i, solve for i's pivot bits: row_j & i = 0/1 (parity) and
        # row_j = (1<<p_j) | nonpivot_mask_j, so
        #   u_j = bit(i, p_j) XOR parity(nonpivot_mask_j & nonpivot_bits)
        # => bit(i, p_j) = u_j XOR parity(row_j & nonpivot_bits)
        idx = nonpivot_bits
        for j, p in enumerate(pivots):
            correction = (rows[j] & nonpivot_bits).bit_count() % 2
            pivot_bit = ((u >> j) & 1) ^ correction
            if pivot_bit:
                idx |= (1 << p)
        return idx

    # -- build ALL of this round's probe columns at once --
    # column 0: base fold (R unshifted)
    # columns 1..len(others): R shifted by e_q, one per non-pivot bit
    # remaining columns: R shifted by a random combined mask (extra check)
    shift_vals = [0] + [1 << q for q in others]
    masks = []
    for _ in range(verify_points):
        mask = 0
        for b in others:
            if rng.random() < 0.5:
                mask |= (1 << b)
        masks.append(mask)
    shift_vals += masks
    s_lo, s_hi = split_hi_lo(shift_vals)  # shape (ncols,)

    K_lo = r_lo[:, None] ^ s_lo[None, :]  # (size, ncols)
    K_hi = r_hi[:, None] ^ s_hi[None, :]

    r_batch, ok = oracle.batch_query(K_lo, K_hi)
    if not ok:
        return None

    G = fwht_columns(r_batch.copy())  # (size, ncols), col0 = base fold
    g_raw = G[:, 0]
    g = g_raw / size

    scale = np.abs(g).max() if g.size else 0.0
    if scale == 0.0:
        return []
    active_bins = np.nonzero(np.abs(g) > atom_rel_tol * scale)[0]
    if active_bins.size == 0:
        return []

    denom = g_raw[active_bins]                     # (A,)
    probe_cols = G[active_bins, 1:]                  # (A, ncols-1)
    ratio = probe_cols / denom[:, None]
    bad_mag = np.abs(np.abs(ratio) - 1.0) > mag_tol   # (A, ncols-1)
    is_singleton = ~bad_mag.any(axis=1)               # (A,)
    sign_neg = ratio < 0                              # (A, ncols-1)

    n_bit_cols = len(others)
    found = []
    for row_idx, u in enumerate(active_bins):
        if not is_singleton[row_idx]:
            continue
        nonpivot_bits = 0
        for col, q in enumerate(others):
            if sign_neg[row_idx, col]:
                nonpivot_bits |= (1 << q)
        idx = reconstruct(int(u), nonpivot_bits)
        found.append((idx, float(g[u])))
    return found


# ----------------------------------------------------------------------
# 5. Top-level driver
# ----------------------------------------------------------------------

def sparse_deconvolve(x, a, n, m=None, expected_k=None, max_rounds=200,
                      stall_rounds=8, tol=1e-6, seed=None, verbose=False,
                      verify_points=3, mag_tol=0.1, atom_rel_tol=0.05):
    """Compute sparse y with fhwt(y) == fhwt(x)/fhwt(a), i.e.
    y = ifhwt(fhwt(x)/fhwt(a)), assuming y is sparse.

    x, a       : dict[int, float] sparse signals over {0,1}^n
    n          : number of bits (e.g. 100)
    m          : subsampling dimension (the transform uses 2**m bins);
                 auto-sized from expected_k if None. The per-round cost is
                 O((|x|+|a|) * n * 2**m), i.e. it DOUBLES with every unit
                 of m, so this is by far the dominant runtime knob.
    expected_k : rough upper bound on nnz(y), i.e. the number of SIGNIFICANT
                 outcomes of the sparse result -- typically tens to a few
                 hundred. It sizes m (~log2(expected_k)+4) and is the early-
                 stopping target. NOTE it is NOT the number of shots and NOT
                 the size of the input support |x|: passing the shot count
                 (e.g. 2**18) drives m into the 20s and 2**m into the
                 millions, which is the usual cause of a "it runs forever".
    mag_tol, atom_rel_tol : noise-robustness knobs of the singleton
                 detector (see run_round). The defaults tolerate the
                 finite-shot noise floor of real measured distributions;
                 for perfectly exact (synthetic) signals they can be set
                 much tighter, and for very noisy data mag_tol can be
                 raised toward ~0.3.
    """
    rng = random.Random(seed)
    if m is None:
        base = expected_k or 1000
        # Cap the auto value: 2**14 = 16384 bins already over-samples any
        # realistically sparse (few-hundred-atom) recovery, and the cost is
        # O(2**m), so uncapping turns an over-large expected_k (e.g. someone
        # passing the shot count) into a multi-million-bin, minutes-per-round
        # blow-up. Pass m explicitly to go above this deliberately.
        m = max(4, min(n - 1, 14, ceil(log2(max(base, 4))) + 4))

    oracle = ResidualOracle(x, a)
    stalled = 0
    for rnd in range(max_rounds):
        t0 = time.time() if verbose else None
        result = run_round(oracle, n, m, tol=tol, rng=rng,
                           verify_points=verify_points, mag_tol=mag_tol,
                           atom_rel_tol=atom_rel_tol)
        if result is None:
            continue  # bad round (hit a zero of fhwt(a)); redraw M
        if verbose:
            print(f"round {rnd}: +{len(result)} atoms "
                  f"(total {len(oracle.y_hat) + len(result)}) "
                  f"[{time.time() - t0:.2f}s]")
        new_atoms = 0
        for idx, val in result:
            if idx not in oracle.y_hat:
                oracle.note_peeled(idx, val)
                new_atoms += 1
        stalled = 0 if new_atoms else stalled + 1
        if expected_k is not None and len(oracle.y_hat) >= expected_k:
            break
        if stalled >= stall_rounds:
            break

    return {i: v for i, v in oracle.y_hat.items() if abs(v) > 1e-9}

# ----------------------------------------------------------------------
# 6. Self-test: build x = a (xor-conv) y_true so the sparsity guarantee
#    holds exactly, then check sparse_deconvolve recovers y_true.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    n = 100
    Ka, Ky = 20, 20

    def rand_sparse(k, n, rng):
        idxs = set()
        while len(idxs) < k:
            idxs.add(rng.getrandbits(n))
        return {i: round(rng.uniform(-3, 3), 4) for i in idxs}

    rng = random.Random(1)
    a = rand_sparse(Ka, n, rng)
    y_true = rand_sparse(Ky, n, rng)
    x = xor_convolve(a, y_true)  # guarantees exact sparse quotient exists

    print(f"n={n}  |a|={len(a)}  |y_true|={len(y_true)}  |x|={len(x)}")

    t0 = time.time()
    y_hat = sparse_deconvolve(x, a, n, m=8, expected_k=len(y_true) + 5,
                               seed=42, verbose=True)
    t1 = time.time()

    matched = sum(1 for i, v in y_true.items()
                  if i in y_hat and abs(y_hat[i] - v) < 1e-4)
    spurious = [i for i in y_hat if i not in y_true]

    print(f"recovered {len(y_hat)} atoms in {t1 - t0:.2f}s")
    print(f"matched exactly: {matched}/{len(y_true)}")
    print(f"spurious atoms:  {len(spurious)}")