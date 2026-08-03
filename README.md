# tai

Fast desktop inference for the PLE TinyLM, in Rust.

tai runs the 28.9M-parameter Per-Layer-Embedding language model from this
repository on a desktop CPU. The model file is memory-mapped, the int4
group-quantized matvecs run on AVX2+FMA kernels (with a scalar fallback), and
the 3.1M-MAC output head is split across every core with rayon. The same
model generates at 9.5 tok/s on an ESP32-S3; see RESULTS.md in `traintai/`
for the microcontroller story and `tai bench` below for desktop numbers.

## Quick start

```bash
cargo build --release

# prove the runtime against a PyTorch golden (committed fixture, no training)
./target/release/tai verify --model fixtures/model-small.bin --golden fixtures/golden-small.txt

# generate; raw token ids work without any tokenizer file
./target/release/tai generate --model fixtures/model-small.bin --prompt-ids 1,2,3 --tokens 32 --seed 1

# with a trained deploy model and its tokenizer (from the traintai submodule)
./target/release/tai generate \
  --model firmware/model/model.bin \
  --tokenizer traintai/data/bpe32768.json \
  --prompt "Once upon a time" \
  --tokens 200 --temperature 0.8 --top-k 40

# throughput across thread counts, with a per-stage profile
./target/release/tai bench --model firmware/model/model.bin --threads 1,2,4,8,16
```

Shared flags: `--threads N` (0 = all cores), `--scalar` (force the scalar
kernels), `--vocab-cap N` (score only the first N head rows), `--seed`.

Sampling follows `src/model.py`: temperature scaling, top-k masking (default
40), softmax, multinomial. `--temperature 0` is greedy.

## NPC dialog model

The runtime's companion model is a single-purpose SillyTavern-format NPC
dialog model (same 28.9M PLE architecture), trained in the sibling repo
[AnEntrypoint/traintai](https://github.com/AnEntrypoint/traintai), which is
included here as the `traintai/` submodule. The ST convention -- card
fields plus name-prefixed turns:

```
Description: <who they are>
Personality: <traits>
Scenario: <where this happens>
<START>
<Name>: <first message>
Player: <question>
<Name>: <answer>
```

The name is the turn prefix, so name binding is structural, not memorized.
Outputs are dialog-only (no *action beats*) with at most one
machine-readable action line per turn (`[DEAL: item gold]`,
`[GOTO: place]`), emitted only when the object or place is really
available in context; a game engine can strip or execute that line, and an
invalid one is just a bad decision, not a crash.

Ship checkpoint: `traintai`'s `runs/ple-st-r16-grpo.pt` lineage -- honest
forge pass 74% (720-rollout adversarial sweep), template echo 0%,
action-beats 0%, persona swap ~0%. The measured arc (3% -> 74%, every
lever root-caused) is documented in `traintai/AGENTS.md`.

In the runtime (prompts end `Name:` with no trailing space -- the measured
deployment convention):

```bash
./target/release/tai generate   --model firmware/model/model.bin   --tokenizer traintai/data/bpe32768.json   --prompt "Description: Dorn, a grumpy dwarven blacksmith of Karhold.
Scenario: Dorn's forge, anvils ringing, coal smoke in the air.
<START>
Dorn: If you've come for steel, say what it needs to do.
Player: What do you have for sale?
Dorn:"   --tokens 60 --temperature 0.5 --stop-string "Player:"
```

GPU notes: training ran on an RTX 3060 Laptop (torch cu128, ~6.4k tok/s vs
1.6k tok/s CPU). For inference, a CUDA-graph decode engine
(`traintai/src/cuda_graph_infer.py`, fp32-logit-exact) measures 1316 tok/s
vs the CPU runtime's 4289-5766 -- at 28.9M params single-stream decode is
kernel-serialization-bound on GPU and bandwidth-trivial on CPU, so the CPU
wins decode while the GPU wins prefill (29,798 tok/s) and training.

Many streams (a game full of NPCs) flip that verdict. The batched engine
(`traintai/src/gpu_batch_infer.py`) captures one CUDA graph serving B
streams per replay -- per-stream positions and causal masks, per-stream
exactness verified against the plain model -- and the same weight reads
amortize across all of them:

| streams | CPU aggregate | GPU batch aggregate |
|---:|---:|---:|
| 1 | 4289 | 1447 |
| 4 | 2469 | 4938 |
| 8 | 4324 | 9283 |
| 16 | 4923 | 17385 |
| 32 | ~5000 (saturated) | 25547 |
| 64 | ~5000 (saturated) | 34595 |

(CPU baseline: N concurrent `tai generate --threads 1` processes, wall-timed;
the CPU saturates near 5k tok/s aggregate however configured.) The crossover
is around 2-4 streams; at 16+ streams the GPU is 3.5-7x. Rule of thumb:
single NPC -> CPU, a scene full of NPCs -> GPU batch.

`traintai/src/npc_demo_batch.py` runs the canonical demo: 10 different NPC
personas with 10 different player questions generated as one batched pass
(`DEMO_SEED=11 UV_NO_SYNC=1 uv run python npc_demo_batch.py` from
`traintai/src/`).

## Model format

tai reads the same `PLE1` model.bin the ESP32 firmware uses: a flat,
mmap-friendly blob of group-wise int4 tensors (fp16 scales, ragged packing)
plus fp32 norm vectors, fully described by its header. Produce one with the
Python pipeline in the `traintai/` submodule:

```bash
cd traintai && uv sync
UV_NO_SYNC=1 uv run python data/prepare.py --vocab 32768   # TinyStories slice + BPE
UV_NO_SYNC=1 uv run python src/train.py --arm ple ...      # train; see experiments/
UV_NO_SYNC=1 uv run python src/export.py <run-tag>         # firmware/model/model.bin + golden.txt
```

For a numerics fixture without training, `traintai/src/make_ckpt.py` writes
a random-init checkpoint in the same format (deploy config by default,
`--small` for the CI fixture).

## Performance

Measured on a 16-core desktop (`tai bench --model firmware/model/model.bin
--threads 1,2,4,8,16`, 200-token greedy decode of the 28.9M-parameter deploy
model):

| threads | tok/s | ms/token |
|---:|---:|---:|
| 1 | 4289 | 0.23 |
| 2 | 5146 | 0.19 |
| 4 | 5766 | 0.17 |
| 8 | 5265 | 0.19 |
| 16 | 2665 | 0.38 |

Baselines on the same machine and model: the C host runtime (`bench.c`, -O3)
247 tok/s, tai's scalar fallback 217 tok/s, tai with fp32 AVX2 matvecs
(`--fp32-head`) 1973 tok/s at 8 threads. Every matvec runs as an exact
integer int8 dot by default: weights are unpacked to bytes once at load,
each matvec input is quantized to int8 once, and rows are computed four at a
time against shared activation registers (`maddubs`/`madd`). Numbers are
from a Ryzen 7 6800H (8C/16T); `--threads 0` uses physical core count, and
more than 8 threads buys nothing on this part. For reference, the same model
runs at 9.5 tok/s on an ESP32-S3.

Attention is a fused single pass over the KV cache for all heads with AVX2
score dots and value accumulation. At long context (pos ~480, where
attention dominates) the runtime holds 2917 tok/s; the per-head scalar
traversal this replaced managed 1436.

Reducing head bytes further was measured and rejected: a packed-int4 head
with in-kernel SIMD unpack (`--i4-head`, bit-identical logits) reads half
the weight bytes but loses ~18% single-threaded to the unpack ALU work and
washes out within noise at 4-8 threads on this chip. Staged int8 stays the
default; the flag remains for bandwidth-poorer hosts.

The int8-everywhere math is the same activation-quantization trick the ESP32
runtime ships: activations are quantized to int8 once per matvec and every
row is an exact integer dot. Its quality cost, measured with
`firmware/host_verify/ppl.c`'s methodology over 4096 val predictions
(`tai ppl --model firmware/model/model.bin --val traintai/data/val_v32768.bin`):

| runtime | val CE | ppl |
|---|---:|---:|
| tai fp32 everywhere | 2.9318 | 18.76 |
| tai int8 everywhere | 2.9320 | 18.76 |
| C fp32 (llm.h) | 2.9318 | 18.76 |
| C int8 (llm.h) | 2.9318 | 18.76 |

## Verification

`tai verify` forwards the golden prompt with exact fp32 matvecs and compares
every last-position logit against the PyTorch reference exported beside the
model, the same check `firmware/host_verify/verify.c` runs for the C runtime.
On the deploy artifact all three (PyTorch, C, Rust) agree to fp print
precision (max abs diff = 0.00000). `tai ppl` measures val perplexity, and is
how the int8 generation path is validated (above).

## Layout

```
tai/                 the Rust desktop runtime
traintai/            training repo (submodule): data engines, sim, loop, evals
firmware/            ESP32-S3 firmware and host verifiers (the original target)
fixtures/            a tiny committed model so CI and fresh clones can verify
media/               demo capture
```

RESULTS.md (the PLE-on-ESP32 research writeup) now lives in `traintai/`.

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
microcontroller memory hierarchy (see RESULTS.md in `traintai/` and LICENSE);
tai is its desktop Rust runtime. MIT license.
