#!/usr/bin/env python3
"""CPU tests for tq4enc (no GPU, no model file): python tests/test_tq4enc.py

  roundtrip          pack + decode gives back the rotated-basis values the encoder chose
  plain_error        plain rounding of Gaussian weights stays under the expected relative error
  refine_monotonic   the weighted refinement never raises the weighted cost
  ldlq_not_worse     with a non-diagonal metric, LDLQ + refinement is never worse than refinement alone
  metric_diagonal    importance_metric of a flat importance is the identity
  calib_roundtrip    a calibration file written in the engine's layout is read back, normalised and regularised
"""
import os, struct, sys, tempfile
os.environ['TQ4ENC_CPU'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import tq4enc as t
from calib_reader import Calib

CAND = np.linspace(0.40, 1.15, 24)


def spd_metric(groups, seed):
    """A random symmetric positive definite metric per group, mean diagonal 1 (what Calib.metric returns)."""
    r = np.random.RandomState(seed)
    a = r.standard_normal((groups, 128, 160)).astype(np.float64)
    m = a @ a.transpose(0, 2, 1) / 160
    m = m / (np.einsum('gii->g', m).sum() / (groups * 128))
    return (m + 0.05 * np.eye(128)[None]).astype(np.float32)


def weighted_cost(w, s, c, M):
    H, sg = t.hadamard128(), t.sign_vector()
    y = ((w * sg) @ H) * np.float32(1.0 / np.sqrt(128.0))
    q = t.LEVELS[c & 7] * np.where(c & 8, -1.0, 1.0).astype(np.float32)
    e = (y - s.astype(np.float32)[:, None] * q).reshape(-1, M.shape[0], 128)
    return float(np.einsum('rgi,gij,rgj->', e, M.astype(np.float64), e))


def roundtrip():
    w = np.random.RandomState(1).standard_normal((512, 128)).astype(np.float32)
    s, c, (num, den), _ = t.encode_tq4(w, CAND, 2)
    back = t.decode_tq4(s, c)
    rel = float(((back - w) ** 2).sum() / (w ** 2).sum())
    assert abs(rel - num / den) < 1e-4, (rel, num / den)
    packed = t.pack_tq4(s, c)
    assert len(packed) == 512 * 66
    rec = np.frombuffer(packed, dtype=np.dtype([('s', '<f2'), ('c', 'u1', 64)]))
    assert (rec['c'] & 15 == c[:, 0::2]).all() and (rec['c'] >> 4 == c[:, 1::2]).all()


def plain_error():
    w = np.random.RandomState(2).standard_normal((2048, 128)).astype(np.float32)
    _, _, (num, den), _ = t.encode_tq4(w, CAND, 2)
    assert num / den < 0.02, num / den   # 16 levels on Gaussian data: about 1 %


def refine_monotonic():
    w = np.random.RandomState(3).standard_normal((4 * 64, 128)).astype(np.float32)
    M = spd_metric(4, 30)
    _, _, _, st = t.encode_tq4(w, CAND, 2, M, passes=4)
    assert st[1] <= st[0] * (1 + 1e-6), st


def ldlq_not_worse():
    w = np.random.RandomState(4).standard_normal((4 * 64, 128)).astype(np.float32)
    M = spd_metric(4, 40)
    s1, c1, _, _ = t.encode_tq4(w, CAND, 2, M, passes=4)
    s2, c2, _, st = t.encode_tq4(w, CAND, 2, M, passes=4, use_ldlq=True)
    a, b = weighted_cost(w, s1, c1, M), weighted_cost(w, s2, c2, M)
    assert b <= a * 1.0001, (a, b)
    assert st[1] <= st[0], st       # reported: cost after <= cost of plain rounding


def metric_diagonal():
    M = t.importance_metric(np.ones(256))
    assert M.shape == (2, 128, 128) and np.allclose(M, np.eye(128)[None], atol=1e-5)


def calib_roundtrip():
    r = np.random.RandomState(5)
    x = r.standard_normal((1000, 256)).astype(np.float32)
    sums = np.stack([x[:, i:i + 128].T @ x[:, i:i + 128] for i in (0, 128)]).astype('<f4')
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'c.bin')
        with open(p, 'wb') as f:
            f.write(b'TQCALIB1' + struct.pack('<4I', 64, 4, 128, 0))
            f.write(struct.pack('<4IQ', 7, 2, 256, 0, 1000) + sums.tobytes())
        c = Calib(p)
        assert c.key_of('blk.7.ffn_gate.weight') == (7, 2) and c.key_of('blk.7.ffn_down.weight') is None
        M = c.metric('blk.7.ffn_up.weight', 256, ridge=0.05)
        assert M.shape == (2, 128, 128) and np.allclose(M, M.transpose(0, 2, 1), atol=1e-6)
        assert abs(np.einsum('gii->', M) / 256 - 1.05) < 1e-3     # mean diagonal 1, plus the ridge
        assert np.linalg.eigvalsh(M.astype(np.float64)).min() > 0.04


CASES = [roundtrip, plain_error, refine_monotonic, ldlq_not_worse, metric_diagonal, calib_roundtrip]
failed = 0
for case in CASES:
    try:
        case()
        print('PASS', case.__name__)
    except Exception as e:  # noqa: BLE001 - report every case
        failed += 1
        print('FAIL %s: %s: %s' % (case.__name__, type(e).__name__, e))
sys.exit(1 if failed else 0)
