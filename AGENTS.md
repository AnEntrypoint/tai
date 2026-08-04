# tai — agent guide (inference repo)

Desktop Rust runtime for the 28.9M-param PLE TinyLM and its SillyTavern NPC
dialog model. Training lives in the `traintai/` submodule
([AnEntrypoint/traintai](https://github.com/AnEntrypoint/traintai)) — its
`AGENTS.md` is the canonical discovered-lever log; read it before touching
anything training-related.

## Commands

```bash
cargo build --release
./target/release/tai verify --model fixtures/model-small.bin --golden fixtures/golden-small.txt
./target/release/tai generate --model firmware/model/model.bin \
  --tokenizer traintai/data/bpe32768.json --prompt "..." --tokens 80 \
  --temperature 0.5 --stop-string "Player:"
./target/release/tai bench --model firmware/model/model.bin --threads 1,2,4,8,16
```

- CI is Rust-only (build + `tai verify` against the committed fixture); the
  submodule is not needed for CI.
- `firmware/model/model.bin` is a gitignored build artifact, produced by
  `traintai/src/export.py` from the current ship checkpoint.
- NPC prompts end `Name:` with NO trailing space — the measured deployment
  convention (46% vs 32% on the same checkpoint).
- The NPC model emits dialog-only text plus at most one bracket action
  line (`[DEAL: item gold]`, `[GOTO: place]`). Engines should strip or
  execute that line and tolerate invalid ones as bad decisions, not errors.

## Runtime invariants (do not regress)

- int8-everywhere matvecs: activations quantized once per matvec, exact
  integer dots, 4-row shared-activation kernel; `--fp32-head` only as a flag.
- Fused single-pass attention over the KV cache; argmax fused into the head
  pass; one scratch allocation; `--threads 0` = physical cores.
- `tai verify` must agree with the PyTorch golden to fp print precision.
