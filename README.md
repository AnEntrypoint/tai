# tai

Fast desktop inference for the PLE TinyLM, in Rust.

tai runs the 28.9M-parameter Per-Layer-Embedding language model from this
repository on a desktop CPU. The model file is memory-mapped, the int4
group-quantized matvecs run on AVX2+FMA kernels (with a scalar fallback), and
the 3.1M-MAC output head is split across every core with rayon. The same
model generates at 9.5 tok/s on an ESP32-S3; see RESULTS.md for the
microcontroller story and `tai bench` below for desktop numbers.

## Quick start

```bash
cargo build --release

# prove the runtime against a PyTorch golden (committed fixture, no training)
./target/release/tai verify --model fixtures/model-small.bin --golden fixtures/golden-small.txt

# generate; raw token ids work without any tokenizer file
./target/release/tai generate --model fixtures/model-small.bin --prompt-ids 1,2,3 --tokens 32 --seed 1

# with a trained deploy model and its tokenizer
./target/release/tai generate \
  --model firmware/model/model.bin \
  --tokenizer data/bpe32768.json \
  --prompt "Once upon a time" \
  --tokens 200 --temperature 0.8 --top-k 40

# throughput across thread counts, with a per-stage profile
./target/release/tai bench --model firmware/model/model.bin --threads 1,2,4,8,16
```

Shared flags: `--threads N` (0 = all cores), `--scalar` (force the scalar
kernels), `--vocab-cap N` (score only the first N head rows), `--seed`.

Sampling follows `src/model.py`: temperature scaling, top-k masking (default
40), softmax, multinomial. `--temperature 0` is greedy.

## Model format

tai reads the same `PLE1` model.bin the ESP32 firmware uses: a flat,
mmap-friendly blob of group-wise int4 tensors (fp16 scales, ragged packing)
plus fp32 norm vectors, fully described by its header. Produce one with the
Python pipeline:

```bash
uv sync
uv run python data/prepare.py --vocab 32768   # TinyStories slice + BPE
uv run python src/train.py --arm ple ...      # train; see experiments/
uv run python src/export.py <run-tag>         # firmware/model/model.bin + golden.txt
```

For a numerics fixture without training, `src/make_ckpt.py` writes a
random-init checkpoint in the same format (deploy config by default, `--small`
for the CI fixture).

## Performance

Measured on a 16-core desktop (`tai bench --model firmware/model/model.bin
--threads 1,2,4,8,16`, 200-token greedy decode of the 28.9M-parameter deploy
model):

| threads | tok/s | ms/token |
|---:|---:|---:|
| 1 | 2409 | 0.4 |
| 2 | 2245 | 0.4 |
| 4 | 2689 | 0.4 |
| 8 | 2667 | 0.4 |
| 16 | 2385 | 0.4 |

Baselines on the same machine and model: the C host runtime (`bench.c`, -O3)
285 tok/s, tai's scalar fallback 240 tok/s, tai with the fp32 AVX2 head
(`--fp32-head`) 932 tok/s single-threaded. The default head is int8-on-int8
(`maddubs`), staged once at load; it is ~4x the fp32 AVX2 head. Numbers are
from a Ryzen 7 6800H (8C/16T); `--threads 0` uses physical core count, and
more than 8 threads buys nothing on this part. For reference, the same model
runs at 9.5 tok/s on an ESP32-S3.

The int8 head is the same activation-quantization trick the ESP32 runtime
ships: activations are quantized to int8 once per token and each head row is
an exact integer dot. Its quality cost, measured with
`firmware/host_verify/ppl.c`'s methodology over 4096 val predictions
(`tai ppl --model firmware/model/model.bin --val data/val_v32768.bin`):

| runtime | val CE | ppl |
|---|---:|---:|
| tai fp32-head | 2.9318 | 18.76 |
| tai int8-head | 2.9316 | 18.76 |
| C fp32 (llm.h) | 2.9318 | 18.76 |
| C int8 (llm.h) | 2.9318 | 18.76 |

## Verification

`tai verify` forwards the golden prompt with the exact fp32 head and compares
every last-position logit against the PyTorch reference exported beside the
model, the same check `firmware/host_verify/verify.c` runs for the C runtime.
On the deploy artifact all three (PyTorch, C, Rust) agree to fp print
precision (max abs diff = 0.00000). `tai ppl` measures val perplexity, and is
how the int8 generation head is validated (above).

## Layout

```
tai/                 the Rust desktop runtime
src/                 training, quantization, export (Python)
data/prepare.py      dataset and tokenizer generation
firmware/            ESP32-S3 firmware and host verifiers (the original target)
experiments/         ablation and deploy scripts
fixtures/            a tiny committed model so CI and fresh clones can verify
RESULTS.md           the PLE-on-ESP32 research writeup
```

## Why it is fast

- mmap: the 14.9MB model is demand-paged straight from disk, never parsed
  into heap structures
- int8 head: the tied output head (32768 x 96, scanned in full every token)
  is unpacked to bytes once at load, activations are quantized to int8 once
  per token, and each row is an exact integer dot (`maddubs`/`madd`), row-split
  across cores with the argmax fused into the same pass
- AVX2+FMA for every other matvec: int4 nibbles unpack to f32 lanes, 32
  columns per iteration, with `target-cpu=native` for the scalar stages
- the dense core is only 559K parameters by design (Per-Layer Embeddings put
  25M parameters in a sparsely-read table), so per-token compute is ~4M MACs
- one scratch allocation up front, precomputed RoPE tables, contiguous
  KV cache

## Credit

The TinyStories dataset (Ronen Eldan and Yuanzhi Li, Microsoft Research,
arXiv:2305.07759) and Google's Per-Layer Embeddings design from the Gemma
models. This repository began as an ESP32-S3 project applying that idea to a
microcontroller memory hierarchy (see RESULTS.md and LICENSE); tai is its
desktop Rust runtime. MIT license.
