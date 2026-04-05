# Llama 2 Internals - Artifacts

From-scratch implementation and visualization artifacts for:

- RoPE (Rotary Position Embedding)
- GQA vs MHA KV-cache scaling in Llama 2

References:

- RoFormer: https://arxiv.org/abs/2104.09864
- Llama 2: https://arxiv.org/abs/2307.09288
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- GQA paper: https://arxiv.org/abs/2305.13245

Accompanies the blog post: [LLaMA 2: How Three Borrowed Techniques Fit a 70B Model on Two GPUs
](https://yashpatel.xyz/blog/llama-2-how-three-borrowed-techniques-fit-a-70b-model-on-two-gpus)

---

## Files

| File | Description |
|------|-------------|
| `rope_from_scratch.py` | Pure NumPy RoPE implementation. Numerically verifies the relative-position property and compares RoPE decay curves against additive sinusoidal encodings. |
| `gqa_kvcache_benchmark.py` | Empirical GPU benchmark that allocates real KV-cache tensors for GPT-2 XL, Llama-2 7B, Llama-2 70B (GQA), and an MQA variant. Measures memory and computes max batch under a fixed VRAM budget. |
| `rope_diagram.py` | Manim scene that renders a two-panel RoPE diagram (geometric rotation view + algebraic identity) for blog or slide use. |

## Included Artifacts

This repo currently includes generated figures:

- `gqa_kvcache_memory.png`
- `gqa_max_batch.png`
- `rope_decay_curves.png`
- `rope_rotation_heatmap.png`

## Hardware

- `gqa_kvcache_benchmark.py`: CUDA GPU required (script exits if CUDA is unavailable)
- `rope_from_scratch.py`: CPU-only, NumPy + Matplotlib
- `rope_diagram.py`: CPU/GPU optional, depends on Manim rendering stack

Benchmarks in code comments were run on RTX 3090 (24GB VRAM), but the scripts are portable.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy==1.24.3 matplotlib==3.7.1 manim

# Torch (CUDA 12.1 wheel for GPU benchmark)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Note: The scripts import a shared plotting theme module:

```python
from deep_sea_theme import apply_theme, PALETTE, EXTENDED, add_source_label
```

Make sure `deep_sea_theme.py` is available at `../theme` relative to this repo.

## Run

```bash
python rope_from_scratch.py
python gqa_kvcache_benchmark.py

# Render the RoPE diagram (exports the final frame)
manim -ql --save_last_frame rope_diagram.py RopeDiagram
```

## Outputs

### `rope_from_scratch.py`

- `rope_decay_curves.png` - RoPE vs sinusoidal attention-magnitude decay over relative distance
- `rope_rotation_heatmap.png` - per-dimension rotation angle map across positions

### `gqa_kvcache_benchmark.py`

- `gqa_kvcache_memory.png` - KV-cache GB vs sequence length for MHA/GQA/MQA configs
- `gqa_max_batch.png` - max batch size under a fixed VRAM KV budget

### `rope_diagram.py`

- Manim output under `media/images/rope_diagram/...png`

---

## Key Findings

1. RoPE relative-position property is verified numerically:

```text
dot(R(m)*q, R(n)*k) depends on (m - n), not absolute m or n
```

2. GQA compresses KV cache by the query-to-KV head ratio:

```text
KV bytes = 2 * batch * layers * seq_len * num_kv_heads * head_dim * dtype_bytes
compression vs MHA = num_q_heads / num_kv_heads
```

For Llama-2 70B (64 query heads, 8 KV heads), GQA gives an 8x KV-cache reduction versus hypothetical MHA at the same size.

3. Under fixed KV memory budgets, GQA translates directly into larger feasible batch sizes at long context lengths.
