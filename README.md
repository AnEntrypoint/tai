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

## NPC dialog model

Beyond the TinyStories base, this repo trains and ships a single-purpose
SillyTavern-format NPC dialog model. The canonical format is the ST
convention -- card fields plus name-prefixed turns:

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
`src/st_data.py` (cards to conversations), `src/st_world.py` (a world DB of
items, prices, places, and quests to grounded dialogues), `src/st_prepare.py`
(interleaved anti-overfit mixture with ~20% general text), plus real
roleplay sets (chimbiwide, apache-2.0; NousResearch/CharacterCodex;
dprashar quest pool; amaydle) feed the pipeline. The model covers the full
ST surface: first messages, example dialogues, lorebook blocks, author's
notes, group scenes, narration-only mode, and user personas.

## The measured training arc (~200M tokens, 14 rounds, RTX 3060)

Each claim below has a number behind it, measured by `src/npc_forge.py`
(generate-grade-inject cycles against 150 cards x 8 samples = 2400 rollouts):

| stage | what it taught | evidence |
|---|---|---|
| SFT on real RP | dialog form, turn structure | val ppl 1458 -> 3.06 in 3 rounds |
| synthetic persona anchors | answering the question asked | intent rate 11% -> 100% in one round |
| name-echo curricula | NOT enough for identity | persona swap stayed 22-78% across all SFT |
| template-heavy SFT | is a regression vector | card_continuation 1% -> 20% after one template-heavy top-up |
| **GRPO (critic-free, DeepSeekMath-style)** | **identity + stopping** | **persona_swap -> 0.1-2%, no_stop 94% -> 2%, pass rate 3% -> 55%** |
| world-DB grounded SFT | grounded onsets | answers open with real items at real prices (object_ungrounded 0.5%) |
| **decontaminated data engine + anti-template GRPO** | **the actual ceiling removed** | **template_echo 81% -> 0%, honest pass 3% -> 30% -> 46% in two rounds** |

The forge is the co-evolution loop: generate, grade (persona_swap, card
continuation, template echo, intent, stop, grounding, repetition), inject
repairs, retrain. An LLM judges the dashboards between rounds; the rule
grader itself was caught over-firing once (83% "drift" that was really 1%
persona swap) and under-firing once (blind to template echo, below).

## The scale assertion, revised by measurement (round 9)

The round-8 version of this section claimed content depth was
capacity-bound at the 559K-param core. Round 9 falsified that: the plateau
was not capacity, it was three missing levers, each found by adversarial
diagnostics rather than by training harder:

1. **The training data contained the ceiling.** The synthetic generators
   spliced raw second-person card scenario text into ~16 fixed response
   templates; one skeleton ("I deal in what this place provides...") alone
   occupied ~1,300 response slots in the bins. The model's "word salad"
   was verbatim reproduction of the data's own template seams. A
   combinatorial rewrite (opener x grounding x closer banks, world-DB
   goods grounding, zero raw scenario splicing) plus a decontamination
   filter removed all of it.
2. **The grader and reward were blind to the failure.** The old forge
   grader had no template-echo check, so the "55% pass" ship number was
   mostly the reward being farmed by the memorized template. Measured
   with the template-aware grader, the round-8 ship model passes 3%,
   with 81% template echo. The decontaminated-data retrain plus two GRPO
   rounds (template-echo penalty, run-global dedup, adaptive 15-85%
   pass-zone prompt curriculum, n-gram repetition penalty) reached 46%
   with 0% echo and 0.4% repetition on the same honest grader.
3. **The RL objective destabilized silently.** Doubling the generation
   window doubled the policy-gradient scale (logprobs were summed over
   the response); round 3 collapsed (no_stop 95%) with loss spikes to
   +50. Per-token mean logprob plus group-std advantage normalization
   brought loss back to +/-0.1 and stabilized further rounds.

**Identity is still reward-solvable** (persona_swap 0-1% throughout),
**form is still data-solvable**, and the depth question is now open
again: each round of the honest loop moves it, which is the signature of
a lever problem, not a capacity wall. The 559K core's true ceiling, if
it exists, has not yet been measured -- every previous "plateau" was an
instrumentation or data artifact.

Levers measured and rejected at this scale, for the record: Muon
(Newton-Schulz orthogonalized momentum on the 2D core matrices) loses to
AdamW at matched budget -- val 3.44 vs 2.90 at 800 steps / 13.1M tokens
on identical bins (`--optimizer muon` remains in train.py for larger
horizons where Muon's advantage is documented to appear). Decoding
(temperature/top-k grid) moves nothing structural: the template onset
was identical from greedy to t=0.9, which is what first proved the
problem was in the data, not the sampler.

Ship checkpoint: `runs/ple-st-r9-grpo2.pt` (decontaminated SFT + GRPO
round 2): honest forge pass 46%, template_echo 0%, repetition 0.4%,
persona_swap 0%, across 720-rollout sweeps on the template-aware grader.
GRPO rounds past two oscillate (40-46%) -- saturation, not collapse;
the normalized objective keeps them stable.

In the runtime:

```bash
./target/release/tai generate   --model firmware/model/model.bin   --tokenizer data/bpe32768.json   --prompt "Description: Dorn, a grumpy dwarven blacksmith of Karhold.
Scenario: Dorn's forge, anvils ringing, coal smoke in the air.
<START>
Dorn: *does not look up* If you've come for steel, say what it needs to do.
Player: What do you have for sale?
Dorn:"   --tokens 60 --temperature 0.5 --stop-string "Player:"
```

`src/npc_demo_batch.py` runs 10 such conversations as one batched GPU pass.

GPU notes:GPU notes: training ran on an RTX 3060 Laptop (torch cu128, ~6.4k tok/s vs
1.6k tok/s CPU). For inference, a CUDA-graph decode engine
(`src/cuda_graph_infer.py`, fp32-logit-exact) measures 1316 tok/s vs the CPU
runtime's 4289-5766 -- at 28.9M params single-stream decode is
kernel-serialization-bound on GPU and bandwidth-trivial on CPU, so the CPU
wins decode while the GPU wins prefill (29,798 tok/s) and training.

Many streams (a game full of NPCs) flip that verdict. The batched engine
(`src/gpu_batch_infer.py`) captures one CUDA graph serving B streams per
replay -- per-stream positions and causal masks, per-stream exactness
verified against the plain model -- and the same weight reads amortize
across all of them:

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

`src/npc_demo_batch.py` runs the canonical demo: 10 different NPC personas
with 10 different player questions generated as one batched pass
(`DEMO_SEED=11 uv run python src/npc_demo_batch.py`).

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
(`tai ppl --model firmware/model/model.bin --val data/val_v32768.bin`):

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
