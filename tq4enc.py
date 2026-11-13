#!/usr/bin/env python3
"""TQ4 quantizer for tqLLM (GGUF tensor type 300), written from the engine's TQ4 decoder.

Format, per block of 128 consecutive weights of a row:
  w_rot = Hadamard128(signs_S1 * w) / sqrt(128)
  w_rot[i] ~= scale(FP16) * LEVEL[code[i]],  code = (negative << 3) | rank,  LEVELS = 6,18,31,44,58,75,96,127
  one 66-byte record: the FP16 scale, then 64 bytes of codes (low nibble = even column).

Usage:
  tq4enc.py --recipe TQ_L.gguf --source-hq model-Q8_0.gguf --source-other model-Q4_K_M.gguf --out model-TQ4.gguf
    --recipe        an existing TQ file of the same architecture: its per-tensor type layout is reused
                    (300 = TQ4, 30 = BF16 in the rotated basis; everything else is copied)
    --source-hq     Q8_0 (or F32) GGUF of the model to quantize: the weights of the TQ4 and BF16 tensors
    --source-other  GGUF of the same model whose other tensors and metadata are copied unchanged

Three rounding modes, each adding to the previous one:
  (default)            per-block scale search, nearest level
  --imatrix FILE       weighted refinement with a llama.cpp importance matrix (diagonal metric)
  --calib FILE         per-block 128 x 128 input statistics collected by the engine itself (full metric)
  --calib FILE --ldlq  sequential rounding with error feedback inside each block (LDLQ / GPTQ), under that metric
"""
import argparse, os, struct, sys, time
import numpy as np

try:
    if os.environ.get('TQ4ENC_CPU'):  # force the CPU path (independent checks)
        raise ImportError
    import cupy as xp
    GPU = True
except Exception:  # no GPU library: the same computation with numpy
    xp = np
    GPU = False

S1 = (0xE46A0359, 0x42F85949, 0xA4C63BBB, 0x49F250A9)
LEVELS = np.array([6, 18, 31, 44, 58, 75, 96, 127], dtype=np.float32)
BOUNDS = (LEVELS[1:] + LEVELS[:-1]) / 2
T_F32, T_Q8_0, T_BF16, T_TQ4 = 0, 8, 30, 300
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FP16_MIN = 5.9604645e-08  # smallest positive FP16 value: the scale of an all-zero block


# ---------------------------------------------------------------- minimal GGUF reader
def _rs(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


def _skip_value(f, t):
    if t == 8:
        _rs(f)
    elif t == 9:
        et, n = struct.unpack('<IQ', f.read(12))
        if et == 8:
            for _ in range(n):
                _rs(f)
        else:
            f.seek(SZ[et] * n, 1)
    else:
        f.seek(SZ[t], 1)


class Gguf:
    def __init__(self, path):
        self.path = path
        f = open(path, 'rb')
        assert f.read(4) == b'GGUF', path
        self.version, nt, nk = struct.unpack('<IQQ', f.read(20))
        self.align = 32
        for _ in range(nk):
            k = _rs(f)
            t = struct.unpack('<I', f.read(4))[0]
            if k == 'general.alignment':
                self.align = struct.unpack('<I', f.read(4))[0]
            else:
                _skip_value(f, t)
        self.kv_end = f.tell()
        self.n_kv = nk
        self.tensors = {}
        order = []
        for _ in range(nt):
            name = _rs(f)
            nd = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack('<%dQ' % nd, f.read(8 * nd))
            ty, off = struct.unpack('<IQ', f.read(12))
            order.append([name, ty, dims, off])
        self.data_start = (f.tell() + self.align - 1) // self.align * self.align
        total = os.path.getsize(path) - self.data_start
        by_off = sorted(order, key=lambda x: x[3])
        size = {}
        for i, t in enumerate(by_off):
            size[t[0]] = (by_off[i + 1][3] if i + 1 < len(by_off) else total) - t[3]
        for name, ty, dims, off in order:
            self.tensors[name] = dict(type=ty, dims=dims, off=off, size=size[name])
        self.order = [t[0] for t in order]
        self.f = f

    def raw(self, name):
        t = self.tensors[name]
        self.f.seek(self.data_start + t['off'])
        return self.f.read(t['size'])

    def header_bytes(self):
        self.f.seek(0)
        return self.f.read(self.kv_end)


def dequant_q8_0(raw, n):
    """Q8_0: blocks of 32 weights = one FP16 scale + 32 signed bytes."""
    rec = np.frombuffer(raw, dtype=np.dtype([('d', '<f2'), ('q', 'i1', 32)]), count=n // 32)
    return (rec['q'].astype(np.float32) * rec['d'].astype(np.float32)[:, None]).reshape(-1)


def load_f32(g, name):
    t = g.tensors[name]
    n = int(np.prod(t['dims']))
    raw = g.raw(name)
    if t['type'] == T_Q8_0:
        return dequant_q8_0(raw, n)
    if t['type'] == T_F32:
        return np.frombuffer(raw, dtype='<f4', count=n).copy()
    raise SystemExit('unsupported source type for %s: %d' % (name, t['type']))


def to_bf16(w):
    u = w.astype('<f4').view('<u4').astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16  # round to nearest, ties to even
    return u.astype('<u2').tobytes()


# ---------------------------------------------------------------- TQ4 transform
def hadamard128():
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    return h


def sign_vector():
    s = np.ones(128, dtype=np.float32)
    for i in range(128):
        if (S1[i // 32] >> (i % 32)) & 1:
            s[i] = -1.0
    return s


LDLQ_FACTORS = (0.90, 0.95, 1.0, 1.05, 1.10)


def ldlq(y, s0, M, factors=LDLQ_FACTORS):
    """Sequential rounding with error feedback (LDLQ / GPTQ restricted to one block). y [B, 128], s0 [B] starting
    scales, M [G, 128, 128] positive definite (B a multiple of G; a block's group is its rank in the row).

    For a fixed scale we look for the levels q that minimise e M e^T with e = y / s - q. With M = C C^T (Cholesky, C
    lower triangular), e M e^T = sum_j (sum_{i >= j} e_i C_ij)^2: fixing the columns from the last to the first, term j
    then depends only on q_j and on the errors already made, which q_j compensates.
    Several scales are tried; each block keeps the cheapest. Returns (rank, negative, scale, cost)."""
    lv = xp.asarray(LEVELS)
    bd = xp.asarray(BOUNDS)
    B = y.shape[0]
    G = M.shape[0]
    assert B % G == 0, (B, G)
    Mh = xp.asnumpy(M) if GPU else np.asarray(M)
    C = xp.asarray(np.linalg.cholesky(Mh.astype(np.float64)).astype(np.float32))   # [G, 128, 128]
    Md = xp.asarray(Mh)
    cdiag = xp.ascontiguousarray(xp.diagonal(C, axis1=1, axis2=2))                 # [G, 128]
    R = B // G

    def quad(e):  # e M e^T per block
        r = xp.matmul(xp.ascontiguousarray(e.reshape(R, G, 128).transpose(1, 0, 2)), Md)
        return (xp.ascontiguousarray(r.transpose(1, 0, 2)).reshape(B, 128) * e).sum(axis=1)

    best_cost = xp.full(B, np.inf, dtype=xp.float32)
    best_q = xp.zeros_like(y)
    best_s = s0.copy()
    for f in factors:
        s = (s0 * np.float32(f)).astype(xp.float16).astype(xp.float32)
        s = xp.where(s > 0, s, xp.float32(FP16_MIN))
        x = (y / s[:, None]).reshape(R, G, 128)
        q = xp.zeros_like(x)
        acc = xp.zeros_like(x)                                   # acc[:, :, j] = sum of the e_k C_kj already fixed (k > j)
        for i in range(127, -1, -1):
            v = x[:, :, i] + acc[:, :, i] / cdiag[None, :, i]    # target corrected for the error already made
            a = xp.abs(v)
            qi = xp.where(v < 0, -1.0, 1.0).astype(xp.float32) * lv[xp.searchsorted(bd, a.ravel()).reshape(a.shape)]
            q[:, :, i] = qi
            e = x[:, :, i] - qi
            if i:
                acc[:, :, :i] += e[:, :, None] * C[None, :, i, :i]
        qf = q.reshape(B, 128)
        cost = quad(y - s[:, None] * qf)
        better = cost < best_cost
        best_cost = xp.where(better, cost, best_cost)
        best_q = xp.where(better[:, None], qf, best_q)
        best_s = xp.where(better, s, best_s)
    neg = best_q < 0
    rank = xp.searchsorted(bd, xp.abs(best_q).ravel()).reshape(best_q.shape)
    return rank, neg, best_s, best_cost


def encode_tq4(w, cand, refits, M=None, passes=0, use_ldlq=False):
    """w: float32 [nblocks, 128]. Returns (FP16 scales [nblocks], uint8 codes [nblocks, 128], (error, norm), stats).
    M (optional): importance metric [G, 128, 128]; nblocks must then be a multiple of G.
    use_ldlq: sequential rounding with error feedback before the refinement (kept block by block only where it wins)."""
    H = xp.asarray(hadamard128())
    sg = xp.asarray(sign_vector())
    lv = xp.asarray(LEVELS)
    bd = xp.asarray(BOUNDS)
    w = xp.asarray(w)
    y = ((w * sg) @ H) * np.float32(1.0 / np.sqrt(128.0))  # H is symmetric: w_rot = R w
    a = xp.abs(y)
    amax = a.max(axis=1)
    amax = xp.where(amax > 0, amax, 1.0).astype(xp.float32)
    best_err = xp.full(a.shape[0], np.inf, dtype=xp.float32)
    best_s = amax / 127.0

    def assign(s):
        rank = xp.searchsorted(bd, (a / s[:, None]).ravel()).reshape(a.shape)
        return rank, lv[rank]

    def refit(lq):
        return (a * lq).sum(axis=1) / xp.maximum((lq * lq).sum(axis=1), 1e-30)

    for fct in cand:
        s = (amax / 127.0 * np.float32(fct)).astype(xp.float32)
        for _ in range(refits):
            _, lq = assign(s)
            s = xp.maximum(refit(lq), 1e-30).astype(xp.float32)
        s = s.astype(xp.float16).astype(xp.float32)  # the scale is stored as FP16
        s = xp.where(s > 0, s, xp.float32(FP16_MIN))
        _, lq = assign(s)
        err = ((a - s[:, None] * lq) ** 2).sum(axis=1)
        better = err < best_err
        best_err = xp.where(better, err, best_err)
        best_s = xp.where(better, s, best_s)
    rank, _ = assign(best_s)
    neg = y < 0
    stats = None
    if M is not None and (passes > 0 or use_ldlq):
        c_start = None
        if use_ldlq:
            Mg = xp.asarray(M)
            Bn, Gn = y.shape[0], M.shape[0]

            def quad(e):
                r = xp.matmul(xp.ascontiguousarray(e.reshape(Bn // Gn, Gn, 128).transpose(1, 0, 2)), Mg)
                return (xp.ascontiguousarray(r.transpose(1, 0, 2)).reshape(Bn, 128) * e).sum(axis=1)

            q0 = xp.where(neg, -lv[rank], lv[rank])
            c0 = quad(y - best_s[:, None] * q0)                  # cost of plain rounding: the reference to beat
            c_start = float(c0.sum())
            r2, n2, s2, c2 = ldlq(y, best_s, M)
            ok = c2 < c0
            rank = xp.where(ok[:, None], r2, rank)
            neg = xp.where(ok[:, None], n2, neg)
            best_s = xp.where(ok, s2, best_s)
        rank, neg, best_s, stats = refine_weighted(y, rank, neg, best_s, M, max(passes, 1))
        if c_start is not None:
            stats = (c_start, stats[1], stats[2])
    codes = (rank.astype(xp.uint8) | (neg.astype(xp.uint8) << 3))
    q = xp.where(neg, -lv[rank], lv[rank])
    num, den = float(((y - best_s[:, None] * q) ** 2).sum()), float((y * y).sum())
    if GPU:
        return xp.asnumpy(best_s.astype(xp.float16)), xp.asnumpy(codes), (num, den), stats
    return best_s.astype(np.float16), codes, (num, den), stats


def importance_metric(d):
    """d: importance of each input column (mean squared activation), length a multiple of 128.
    A row's output error is sum_j d_j * e_j^2 in the original basis; in the rotated basis it becomes eps M eps^T with
    M = H D H / 128 per group of 128 columns (the S1 signs cancel). Returns M [G, 128, 128]."""
    d = np.asarray(d, dtype=np.float64).reshape(-1, 128)
    d = d / max(d.mean(), 1e-30)
    d = np.maximum(d, 1e-4)  # floor: keeps M positive definite, no column fully ignored
    H = hadamard128().astype(np.float64)
    return np.einsum('ij,gj,jk->gik', H, d, H, optimize=True).astype(np.float32) / 128.0


def refine_weighted(y, rank, neg, s, M, passes):
    """Importance-weighted refinement. y [B, 128] (B a multiple of the group count G), M [G, 128, 128].
    Each pass refits the scale in the weighted sense, then moves codes by one level where that lowers the weighted
    cost; a block keeps its moves only if its real cost goes down. The cost can therefore only decrease."""
    lv = xp.asarray(LEVELS)
    G = M.shape[0]
    M = xp.asarray(M)
    diag = xp.asarray(np.ascontiguousarray(np.diagonal(xp.asnumpy(M) if GPU else M, axis1=1, axis2=2)))  # [G, 128]
    B = y.shape[0]
    assert B % G == 0, (B, G)

    def mm(v):  # v [B, 128] -> v M, a block's group being its rank in the row
        # one large product per column group ([rows, 128] x [128, 128]) instead of millions of small ones
        r = xp.matmul(xp.ascontiguousarray(v.reshape(B // G, G, 128).transpose(1, 0, 2)), M)
        return xp.ascontiguousarray(r.transpose(1, 0, 2)).reshape(B, 128)

    def cost_of(e):
        return (mm(e) * e).sum(axis=1)

    def fp16(v):
        v = v.astype(xp.float16).astype(xp.float32)
        return xp.where(v > 0, v, xp.float32(FP16_MIN))

    dg = xp.tile(diag, (B // G, 1))
    q = xp.where(neg, -lv[rank], lv[rank])
    c0 = cost_of(y - s[:, None] * q)
    cost = c0
    rng = xp.random.RandomState(1234)
    for p in range(passes):
        # 1. the weighted-optimal scale for the current codes
        qm = mm(q)
        s2 = fp16(xp.where((qm * q).sum(axis=1) > 0, (qm * y).sum(axis=1) / xp.maximum((qm * q).sum(axis=1), 1e-30), s))
        c2 = cost_of(y - s2[:, None] * q)
        ok = c2 < cost
        s = xp.where(ok, s2, s)
        cost = xp.where(ok, c2, cost)
        # 2. one-level moves (up, or down with a sign change when crossing zero)
        e = y - s[:, None] * q
        g = mm(e)
        sgn = xp.where(neg, -1.0, 1.0).astype(xp.float32)
        up_ok = rank < 7
        q_up = sgn * lv[xp.minimum(rank + 1, 7)]
        q_dn = xp.where(rank > 0, sgn * lv[xp.maximum(rank - 1, 0)], -sgn * lv[0])
        best_dq = xp.zeros_like(q)
        best_gain = xp.zeros_like(q)
        for qn, valid in ((q_up, up_ok), (q_dn, None)):
            dq = (qn - q) * s[:, None]
            dc = -2.0 * dq * g + dq * dq * dg
            if valid is not None:
                dc = xp.where(valid, dc, 0.0)
            take = dc < best_gain
            best_gain = xp.where(take, dc, best_gain)
            best_dq = xp.where(take, qn - q, best_dq)
        move = (best_gain < 0) & (rng.random_sample(q.shape) < (0.5 if p else 0.35))
        q_new = xp.where(move, q + best_dq, q)
        c_new = cost_of(y - s[:, None] * q_new)
        acc = c_new < cost
        q = xp.where(acc[:, None], q_new, q)
        cost = xp.where(acc, c_new, cost)
    neg2 = q < 0
    rank2 = xp.searchsorted(xp.asarray(BOUNDS), xp.abs(q).ravel()).reshape(q.shape)
    den = cost_of(y)
    return rank2, neg2, s, (float(c0.sum()), float(cost.sum()), float(den.sum()))


def pack_tq4(scales, codes):
    n = scales.shape[0]
    rec = np.zeros(n, dtype=np.dtype([('s', '<f2'), ('c', 'u1', 64)]))
    rec['s'] = scales
    rec['c'] = codes[:, 0::2] | (codes[:, 1::2] << 4)  # low nibble = even column
    return rec.tobytes()


def decode_tq4(scales, codes):
    """Reference decoder on the CPU (exact inverse of encode_tq4's packing): returns the approximated weights."""
    lv = LEVELS[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    y = lv * scales.astype(np.float32)[:, None]
    return ((y @ hadamard128()) * np.float32(1.0 / np.sqrt(128.0))) * sign_vector()


# ---------------------------------------------------------------- program
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--recipe', required=True, help='existing TQ GGUF of the same architecture: its tensor type layout is reused')
    ap.add_argument('--source-hq', required=True, help='Q8_0 or F32 GGUF of the model to quantize')
    ap.add_argument('--source-other', required=True, help='GGUF of the same model: other tensors and metadata are copied from it')
    ap.add_argument('--out', required=True, help='output GGUF')
    ap.add_argument('--candidates', type=int, default=24, help='scales tried per block')
    ap.add_argument('--refits', type=int, default=2, help='scale refits per candidate')
    ap.add_argument('--blocks-per-batch', type=int, default=1 << 19)
    ap.add_argument('--limit', type=int, default=0, help='trial run: quantize only the first N TQ4 tensors')
    ap.add_argument('--imatrix', default='', help='llama.cpp importance matrix (GGUF): weighted rounding, diagonal metric')
    ap.add_argument('--passes', type=int, default=4, help='weighted refinement passes (with --imatrix or --calib)')
    ap.add_argument('--alpha', type=float, default=1.0, help='exponent applied to the imatrix importance (0 = no effect, 1 = full)')
    ap.add_argument('--calib', default='', help='per-block statistics collected by the engine (TQLLM_CALIB_OUT): the full metric '
                    'of each block; takes precedence over --imatrix for the tensors it covers')
    ap.add_argument('--ridge', type=float, default=1e-3, help='regularisation of the --calib metric (fraction of the mean diagonal); '
                    'larger = trust the calibration less. 0.05 is the value validated with --ldlq (see README)')
    ap.add_argument('--ldlq', action='store_true', help='sequential rounding with error feedback inside each block, before the refinement')
    a = ap.parse_args()

    rec, hq, oth = Gguf(a.recipe), Gguf(a.source_hq), Gguf(a.source_other)
    imx = Gguf(a.imatrix) if a.imatrix else None
    cal = None
    if a.calib:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from calib_reader import Calib
        cal = Calib(a.calib)
        print('calibration: %d measured inputs in %s' % (len(cal.keys), a.calib), flush=True)
    n_cal = 0
    wsum, n_weighted = [0.0, 0.0, 0.0], 0
    cand = np.linspace(0.40, 1.15, a.candidates)
    plan = []
    for name in oth.order:
        r = rec.tensors.get(name)
        if r is None:
            raise SystemExit('tensor missing from the recipe: ' + name)
        if tuple(r['dims']) != tuple(oth.tensors[name]['dims']):
            raise SystemExit('different dimensions for ' + name)
        plan.append((name, r['type'] if r['type'] in (T_TQ4, T_BF16) else None))
    print('compute backend:', 'GPU (cupy)' if GPU else 'CPU (numpy)', flush=True)
    print('tensors: %d TQ4, %d BF16, %d copied' % (sum(p[1] == T_TQ4 for p in plan), sum(p[1] == T_BF16 for p in plan),
                                                  sum(p[1] is None for p in plan)), flush=True)

    tmp = a.out + '.data'
    infos, off, done, t0 = [], 0, 0, time.time()
    err_sum = err_n = 0.0
    with open(tmp, 'wb') as out:
        for name, kind in plan:
            src = oth.tensors[name]
            dims = src['dims']
            if kind == T_TQ4 and (a.limit == 0 or done < a.limit):
                cols = int(dims[0])
                if cols % 128:
                    raise SystemExit('%s: row length %d is not a multiple of 128' % (name, cols))
                w = load_f32(hq, name).reshape(-1, 128)
                parts, num, den = [], 0.0, 0.0
                G = cols // 128
                M = None
                if cal is not None:
                    M = cal.metric(name, cols, a.ridge)
                    if M is not None:
                        n_cal += 1
                        n_weighted += 1
                if M is None and imx is not None and (name + '.in_sum2') in imx.tensors:
                    d = np.frombuffer(imx.raw(name + '.in_sum2'), dtype='<f4', count=cols).astype(np.float64)
                    cnt = float(np.frombuffer(imx.raw(name + '.counts'), dtype='<f4', count=1)[0]) if (name + '.counts') in imx.tensors else 1.0
                    if np.isfinite(d).all() and d.sum() > 0:
                        M = importance_metric((d / max(cnt, 1.0)) ** a.alpha)
                        n_weighted += 1
                batch = max(G, (a.blocks_per_batch // G) * G)  # a multiple of the column group count
                for i in range(0, w.shape[0], batch):
                    s, c, rel, st = encode_tq4(w[i:i + batch], cand, a.refits, M, a.passes, a.ldlq)
                    parts.append(pack_tq4(s, c))
                    num += rel[0]
                    den += rel[1]
                    if st:
                        wsum[0] += st[0]
                        wsum[1] += st[1]
                        wsum[2] += st[2]
                    if done == 0 and i == 0:  # check: decoding gives the weights back
                        back = decode_tq4(s[:4096], c[:4096])
                        ref = w[:4096]
                        print('  round-trip check %s: relative error %.5f' % (name, float(((back - ref) ** 2).sum() / (ref ** 2).sum())), flush=True)
                data, ty = b''.join(parts), T_TQ4
                rel_acc = num / max(den, 1e-30)
                err_sum += num
                err_n += den
                done += 1
                if done % 20 == 0 or done == 1:
                    print('  %3d TQ4 tensors | %s | relative error %.5f | %.0f s' % (done, name, rel_acc, time.time() - t0), flush=True)
            elif kind == T_BF16:
                # In a TQ4-body file these matrices receive the same rotated input as the TQ4 tensors: the official
                # files store them in the rotated basis (checked: cosine 1.0000), without rounding.
                w = load_f32(hq, name)
                if int(dims[0]) % 128:
                    raise SystemExit('%s: row length %d is not a multiple of 128' % (name, int(dims[0])))
                w = ((w.reshape(-1, 128) * sign_vector()) @ hadamard128()) * np.float32(1.0 / np.sqrt(128.0))
                data, ty = to_bf16(w.reshape(-1)), T_BF16
            else:
                data, ty = oth.raw(name), src['type']
            infos.append((name, dims, ty, off))
            out.write(data)
            pad = (-len(data)) % oth.align
            out.write(b'\0' * pad)
            off += len(data) + pad

    head = oth.header_bytes()
    with open(a.out, 'wb') as f:
        f.write(head)
        for name, dims, ty, o in infos:
            nb = name.encode('utf-8')
            f.write(struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(dims)) + struct.pack('<%dQ' % len(dims), *dims) + struct.pack('<IQ', ty, o))
        f.write(b'\0' * ((-f.tell()) % oth.align))
        with open(tmp, 'rb') as d:
            while True:
                b = d.read(1 << 26)
                if not b:
                    break
                f.write(b)
    os.remove(tmp)
    print('wrote %s: %.2f GiB | mean TQ4 relative error %.5f | %.0f s' % (a.out, os.path.getsize(a.out) / 2 ** 30, err_sum / max(err_n, 1), time.time() - t0))
    if cal is not None:
        print('calibration: engine-measured metric for %d of %d weighted tensors' % (n_cal, n_weighted))
    if n_weighted:
        print('weighting: %d tensors | relative weighted error before %.5f -> after %.5f (%.1f %% lower)' % (
            n_weighted, wsum[0] / max(wsum[2], 1e-30), wsum[1] / max(wsum[2], 1e-30), 100.0 * (1.0 - wsum[1] / max(wsum[0], 1e-30))))


if __name__ == '__main__':
    main()
