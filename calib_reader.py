#!/usr/bin/env python3
"""Reader and checker for the calibration file written by the engine (TQLLM_CALIB_OUT, see patches/).

Layout: "TQCALIB1", u32 layers, u32 kinds, u32 block, u32 reserved; then for each measured input:
  u32 layer, u32 kind, u32 dim, u32 reserved, u64 tokens, dim/block matrices of block x block float32 (sums of x x^T
  in the rotated basis).
Kinds: 0 the layer input (attn_q/k/v; attn_qkv, attn_gate, ssm_alpha, ssm_beta)   1 input of attn_output / ssm_out
       2 input of ffn_gate / ffn_up                                              3 input of ffn_down

Used as a module by tq4enc.py (class Calib); run directly, it prints a check of the file:
  calib_reader.py file.bin [imatrix.gguf]
"""
import re, struct, sys
import numpy as np

KIND = {'attn_q': 0, 'attn_k': 0, 'attn_v': 0, 'attn_qkv': 0, 'attn_gate': 0, 'ssm_alpha': 0, 'ssm_beta': 0,
        'attn_output': 1, 'ssm_out': 1, 'ffn_gate': 2, 'ffn_up': 2, 'ffn_down': 3}
NAME = re.compile(r'^blk\.(\d+)\.([a-z_]+)\.weight$')


class Calib:
    def __init__(self, path):
        self.path = path
        self.keys = {}   # (layer, kind) -> (dim, tokens, data offset)
        with open(path, 'rb') as f:
            if f.read(8) != b'TQCALIB1':
                raise SystemExit('not a calibration file: ' + path)
            self.layers, self.kinds, self.block, _ = struct.unpack('<4I', f.read(16))
            if self.block != 128:
                raise SystemExit('unexpected block size: %d' % self.block)
            while True:
                h = f.read(24)
                if not h:
                    break
                if len(h) != 24:
                    raise SystemExit('truncated calibration file (header)')
                layer, kind, dim, _, tokens = struct.unpack('<4IQ', h)
                n = dim * self.block * 4
                self.keys[(layer, kind)] = (dim, tokens, f.tell())
                f.seek(n, 1)
            if f.tell() != f.seek(0, 2):
                raise SystemExit('truncated calibration file (data)')

    def key_of(self, tensor_name):
        m = NAME.match(tensor_name)
        if not m or m.group(2) not in KIND:
            return None
        k = (int(m.group(1)), KIND[m.group(2)])
        return k if k in self.keys else None

    def blocks(self, key):
        """Per-token mean of x x^T for each block: float64 [dim/128, 128, 128], symmetrised."""
        dim, tokens, pos = self.keys[key]
        with open(self.path, 'rb') as f:
            f.seek(pos)
            a = np.frombuffer(f.read(dim * self.block * 4), dtype='<f4').astype(np.float64)
        a = a.reshape(dim // self.block, self.block, self.block) / max(tokens, 1)
        return (a + a.transpose(0, 2, 1)) * 0.5

    def metric(self, tensor_name, cols, ridge=1e-3):
        """Metric [G, 128, 128] float32 for encode_tq4, or None when this tensor was not measured.
        Normalised (mean diagonal 1) then regularised: M + ridge * I, which keeps it positive definite."""
        k = self.key_of(tensor_name)
        if k is None:
            return None
        if self.keys[k][0] != cols:
            raise SystemExit('%s: %d columns, calibration measured on %d' % (tensor_name, cols, self.keys[k][0]))
        M = self.blocks(k)
        if not np.isfinite(M).all():
            raise SystemExit('%s: non-finite values in the calibration' % tensor_name)
        d = np.einsum('gii->g', M).sum() / (M.shape[0] * 128)
        if d <= 0:
            return None
        M = M / d
        M = M + ridge * np.eye(128)[None]
        return M.astype(np.float32)


def main():
    c = Calib(sys.argv[1])
    print('%s: %d layers, %d measured inputs' % (sys.argv[1], c.layers, len(c.keys)))
    by = {}
    for (l, k), (dim, tok, _) in sorted(c.keys.items()):
        by.setdefault((k, dim), []).append(tok)
    for (k, dim), toks in sorted(by.items()):
        print('  kind %d, width %5d: %3d inputs, %d to %d tokens' % (k, dim, len(toks), min(toks), max(toks)))
    # Structure: share of the energy off the diagonal in the rotated basis, and extreme eigenvalues of a few blocks
    for key in [(0, 0), (0, 2), (0, 3), (c.layers // 2, 0), (c.layers // 2, 3), (c.layers - 1, 2)]:
        if key not in c.keys:
            continue
        M = c.blocks(key)
        diag = np.einsum('gii->gi', M)
        off = (M ** 2).sum(axis=(1, 2)) - (diag ** 2).sum(axis=1)
        ev = np.linalg.eigvalsh(M[: min(8, M.shape[0])])
        print('  layer %2d kind %d: off-diagonal = %.1f %% of the norm; min/max eigenvalue (8 blocks) %.2e / %.2e; negative %d'
              % (key[0], key[1], 100 * off.sum() / (M ** 2).sum(), ev.min(), ev.max(), int((ev < 0).sum())))
    if len(sys.argv) > 2:
        # Comparison with the diagonal importance matrix (llama.cpp): the diagonal of R^T M R should resemble it
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import tq4enc as t
        imx = t.Gguf(sys.argv[2])
        H = t.hadamard128().astype(np.float64); s = t.sign_vector().astype(np.float64)
        R = (H * s[None, :]) / np.sqrt(128.0)        # x' = R x per block
        for name in ['blk.0.ffn_gate.weight', 'blk.10.ffn_down.weight', 'blk.3.attn_q.weight', 'blk.20.attn_qkv.weight', 'blk.40.ffn_up.weight']:
            k = c.key_of(name)
            if k is None or (name + '.in_sum2') not in imx.tensors:
                continue
            dim = c.keys[k][0]
            d = np.frombuffer(imx.raw(name + '.in_sum2'), dtype='<f4', count=dim).astype(np.float64).reshape(-1, 128)
            M = c.blocks(k)
            back = np.einsum('ji,gjk,kl->gil', R, M, R)     # R^T M R = mean of x x^T in the original basis
            dg = np.einsum('gii->gi', back)
            a, b = dg.ravel() / dg.mean(), d.ravel() / d.mean()
            offp = 100 * (1 - (dg ** 2).sum() / (back ** 2).sum())
            print('  %-26s correlation with the diagonal matrix %.3f; in the original basis, off-diagonal = %.1f %% of the norm'
                  % (name, float(np.corrcoef(np.log(a + 1e-12), np.log(b + 1e-12))[0, 1]), offp))


if __name__ == '__main__':
    main()
