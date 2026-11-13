# tq4enc

An open quantizer for the **TQ4** weight format of [tqLLM](https://github.com/turbo-tan/tqllm) (GGUF tensor type 300:
128-weight Hadamard rotation, one FP16 scale per block, 16 levels), with three rounding modes of increasing fidelity.
It turns a `Q8_0` GGUF of a model into a TQ4 file the engine runs as is.

It was written to quantize a fine-tune that has no official TQ file (an uncensored Qwen3.8-27B), on one RTX 4090.

> Status: works end to end on Qwen3.8-27B. Tested on that one architecture only. Not affiliated with the tqLLM
> authors; the format is theirs, this encoder was written from the engine's decoder.

## What it does

| Mode | Option | Rounding error is measured with |
|---|---|---|
| plain | (default) | nothing: per-block scale search, nearest level |
| weighted | `--imatrix FILE` | a llama.cpp importance matrix (one weight per input column) |
| calibrated | `--calib FILE` | the full 128 x 128 input statistics of every block, **collected by the engine itself** |
| calibrated + error feedback | `--calib FILE --ldlq` | the same statistics, with sequential rounding (LDLQ / GPTQ inside each block) |

The calibrated modes need a small patch to the engine ([patches/](patches/)): with `TQLLM_CALIB_OUT=FILE`, tqserve
accumulates `x xᵀ` for each matrix input over the prompts it reads, per 128-wide block, in the rotated basis the weights
are stored in. That is the exact Hessian of a block's rounding error, measured in the engine that will run the file,
with its own arithmetic and KV format. `collect_calibration.py` feeds it a corpus.

## Results

Qwen3.8-27B (uncensored fine-tune), RTX 4090, tqLLM 1.5.0. KL divergence against the logits of the `Q8_0` GGUF
(llama.cpp), 512-token passages, BF16 KV cache, measured with the engine's `tqkld`. "code" is C++ source, "text" is
the WikiText-2 test split; the calibration corpus contains neither.

| File | Mean KLD, code | Mean KLD, text | Size |
|---|---|---|---|
| llama.cpp `Q4_K_M` (reference point) | 0.041 | 0.043 | 15.66 GiB |
| TQ4, plain | 0.038 | 0.037 | 13.63 GiB |
| TQ4, `--imatrix` | 0.0325 | 0.0320 | 13.63 GiB |
| TQ4, `--calib` | 0.0315 | 0.0325 | 13.63 GiB |
| TQ4, `--calib --ldlq --ridge 0.001` | 0.0212 | 0.0230 | 13.63 GiB |
| **TQ4, `--calib --ldlq --ridge 0.05`** | **0.0218** | **0.0235** | 13.63 GiB |

Error feedback is what moves the number: one third less divergence than weighted rounding, at the same size and the
same decoding speed (24 prompts, 28.8k tokens, temperature 0: 176 tok/s and 38.4 % accepted drafts, against 172 tok/s
and 37.9 % for the `--imatrix` file).

### Fidelity is not the whole story: check behaviour too

The lowest-KLD file (`--ridge 0.001`) is **not** the one to use. On a 23-task coding / logic / agent bench at
temperature 0 it fell into exact reasoning loops (the same line written until `max_tokens`) on 2 tasks; the
`--ridge 0.05` file did not, for a KLD difference inside the measurement noise (± 0.0008).

| File | Bench, temperature 0 | Exact reasoning loops |
|---|---|---|
| `--imatrix` | 22-23 / 23 | 0 |
| `--calib --ldlq --ridge 0.001` | 21 / 23 | 2 |
| `--calib --ldlq --ridge 0.05` | 22 / 23 | 0 |

`--ridge` is how much the rounding trusts the calibration statistics: a small value fits them closely, including what
is specific to the calibration corpus. KLD on a few hundred passages does not see the difference; long generations
do. If you tune this, measure both.

At temperature 0.6 the picture is different and has nothing to do with the quantizer: with or without calibration the
model sometimes thinks without concluding (13 empty answers out of 30 for both files on two hard tasks, with no
reasoning budget). A reasoning budget fixes that; the quantizer does not cause it.

Full context (148,762 tokens of real code in the prompt, temperature 0): the `--imatrix` and the `--ridge 0.05` files
both recall 6 / 6 planted and real facts, port a function from the context correctly, and write 3-4k words without
repetition.

## Use

Requirements: Python 3.10+, numpy; cupy for a GPU run (about 10 minutes for a 27B model on an RTX 4090; the CPU path
gives the same bytes and is much slower).

```sh
# 1. plain or weighted: no engine change needed
python tq4enc.py --recipe Qwen3.8-27B-TQ_L.gguf --source-hq model-Q8_0.gguf --source-other model-Q4_K_M.gguf \
                 --imatrix model.imatrix.gguf --out model-TQ4.gguf

# 2. calibrated: build tqserve with patches/, then collect statistics with a first TQ4 file
TQLLM_CALIB_OUT=calib.bin tqserve --model model-TQ4.gguf --context 16384 --port 8086 &
python collect_calibration.py --port 8086 --tokens 340000 \
       --code 'src/**/*.cpp' 'lib/**/*.py' --docs 'docs/**/*.md' --prose wiki.train.txt --self 24

# 3. quantize again with the statistics
python tq4enc.py --recipe Qwen3.8-27B-TQ_L.gguf --source-hq model-Q8_0.gguf --source-other model-Q4_K_M.gguf \
                 --calib calib.bin --ldlq --ridge 0.05 --out model-TQ4-calibrated.gguf
```

- `--recipe` is an existing TQ file of the same architecture (for Qwen3.8-27B, the official `TQ_L`): only its layout
  is read, which tensors are TQ4, which stay BF16 in the rotated basis, which are copied. No weights come from it.
- `--source-hq` holds the weights to quantize (`Q8_0` or `F32`); `--source-other` provides the metadata and the tensors
  that are copied unchanged (embeddings, head, vision, MTP).
- The statistics are collected with a first quantization of the same model (step 1), then used for the second.

`python tests/test_tq4enc.py` runs the CPU tests (no model needed). `python calib_reader.py calib.bin` prints a check
of a calibration file.

## How it works

**Format.** A row is cut into blocks of 128 weights. Each block is sign-flipped, Hadamard-rotated and scaled by
1/√128, then stored as one FP16 scale and 128 four-bit codes (sign + one of 8 magnitudes). The rotation spreads
outliers, so one scale per block is enough.

**Plain.** 24 candidate scales per block, each refit twice by least squares, nearest level; the scale with the lowest
squared error wins.

**Weighted.** With an importance `d_j` per input column, a row's output error is `Σ d_j e_j²` in the original basis,
that is `ε M εᵀ` with `M = H D H / 128` in the rotated one. Passes alternate a weighted scale refit and one-level
code moves; a block keeps a pass only if its real cost went down.

**Calibrated.** `M` is no longer built from a diagonal: it is the measured second moment of the block's actual input
in the rotated basis, normalised to a mean diagonal of 1, plus `ridge · I`.

**Error feedback.** With `M = C Cᵀ` (Cholesky), `ε M εᵀ = Σ_j (Σ_{i≥j} ε_i C_ij)²`. Fixing the columns from last to
first, each rounding sees the error already made on the others and compensates it. Five scales around the plain one
are tried; a block keeps the result only where it beats plain rounding under `M`, then goes through the weighted
refinement. The cost under `M` can therefore only decrease.

## Limits

- One architecture tested (Qwen3.8-27B, dense). MoE files and the `-QSA` files of tqLLM 1.6.0 (in-file attention
  indexer) are not produced.
- A recipe file of the same architecture is required.
- The statistics behind the results above were collected with the engine patch on tqLLM 1.5.0. The file in `patches/` is the same change on 1.6.0, built and checked there (it writes the same 256 inputs). On Windows, 1.6.0 itself needs [tqllm#41](https://github.com/turbo-tan/tqllm/pull/41) to build.
- KLD numbers are against `Q8_0`, not BF16: they are comparable with each other, not with numbers measured against
  BF16 elsewhere.
- Error feedback acts inside a block of 128, not across a whole row.

## Credits

- [tqLLM](https://github.com/turbo-tan/tqllm) (Apache-2.0): the engine and the TQ formats. The encoder mirrors its
  TQ4 decoder; the calibration patch is a change to its `qwen35` engine.
- [azllm](https://github.com/xhinker/azllm) (Apache-2.0): the fork in which the first version of this tool was
  developed and measured.
- TurboQuant and the rotation-based quantizers credited by tqLLM for the format itself.
- GPTQ (Frantar et al., 2022) and LDLQ / QuIP (Chee et al., 2023) for sequential rounding with error feedback.
- llama.cpp for the importance matrix format and the `Q8_0` / `Q4_K_M` references.

Written with the assistance of Claude (Anthropic). The measurements were run and checked on the author's RTX 4090.

## License

Apache-2.0, see [LICENSE](LICENSE).
