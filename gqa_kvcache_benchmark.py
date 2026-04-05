"""
GQA vs MHA -- KV Cache Memory Benchmark
========================================
Papers:
  Llama 2: Open Foundation and Fine-Tuned Chat Models (arXiv: 2307.09288)
  GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
       (arXiv: 2305.13245)

What this measures:
  Allocates real PyTorch KV-cache tensors on GPU for four architecture configs
  (GPT-2 XL MHA, Llama-2 7B MHA, Llama-2 70B GQA, MQA variant) and measures
  memory empirically with torch.cuda.memory_allocated(). Sweeps over sequence
  lengths 128-4096 and computes the maximum batch size achievable under a fixed
  20 GB VRAM budget. Produces two Deep Sea theme charts.

  No model weights are loaded -- the KV cache depends only on:
      num_layers x num_kv_heads x head_dim x seq_len x batch x 2 x dtype_bytes
  We allocate shaped tensors that exactly match each architecture's KV cache layout.

Hardware: RTX 3090, 24 GB VRAM (GPU required -- exits if no CUDA device)
Time to run: ~25 seconds

Dependencies (pinned):
    torch==2.1.0        pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    matplotlib==3.7.1   pip install matplotlib==3.7.1
    numpy==1.24.3       pip install numpy==1.24.3

Run: python gqa_kvcache_benchmark.py
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import torch

# Deep Sea chart theme -- consistent styling across the LLM Engineering Stack series
sys.path.insert(0, str(Path(__file__).parent.parent / "theme"))
from deep_sea_theme import apply_theme, PALETTE, EXTENDED, add_source_label
apply_theme()

print("=" * 60)
print("GQA vs MHA KV Cache Benchmark  |  Llama 2 / GQA paper")
print("=" * 60)

if not torch.cuda.is_available():
    print("\nERROR: No CUDA device found. This benchmark requires a GPU.")
    print("  The script measures torch.cuda.memory_allocated() on real tensors.")
    sys.exit(1)

DEVICE = torch.device("cuda:0")
_vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
print(f"\nGPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {_vram_total_gb:.1f} GB total")

# Conservative budget -- leaves ~4 GB headroom for activations and model weights
VRAM_BUDGET_GB = 20.0


# =============================================================================
# Section 1: Architecture Configs
# =============================================================================

@dataclass
class AttentionConfig:
    """
    Specifies the KV-cache-relevant dimensions for one model architecture.

    Only num_kv_heads appears in the memory formula -- not num_q_heads. That
    asymmetry is the entire point of GQA: query heads are cheap at inference
    (computed once, discarded), while KV heads must be stored for every token
    in every layer for the full sequence lifetime.
    """
    name: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int    # == num_q_heads for MHA; 1 for MQA; < num_q_heads for GQA
    head_dim: int
    dtype: torch.dtype = torch.float16

    @property
    def kv_head_ratio(self) -> float:
        """Compression factor: how many Q heads share each KV head."""
        return self.num_q_heads / self.num_kv_heads

    @property
    def dtype_bytes(self) -> int:
        return {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}[self.dtype]


# Architecture specs drawn from published model cards and Table 1 of arXiv:2307.09288.
# GPT-2 XL: n_embd=1600, n_head=25, n_layer=48 (OpenAI model card)
# Llama-2 7B: 32 layers, 32 heads, head_dim=128 (Table 1, arXiv:2307.09288)
# Llama-2 70B: 80 layers, 64 Q-heads, 8 KV-heads via GQA (Table 1, arXiv:2307.09288)
CONFIGS: List[AttentionConfig] = [
    AttentionConfig("GPT-2 XL (MHA)",    num_layers=48, num_q_heads=25, num_kv_heads=25, head_dim=64),
    AttentionConfig("Llama-2 7B (MHA)",  num_layers=32, num_q_heads=32, num_kv_heads=32, head_dim=128),
    AttentionConfig("Llama-2 70B (GQA)", num_layers=80, num_q_heads=64, num_kv_heads=8,  head_dim=128),
    AttentionConfig("MQA variant",       num_layers=80, num_q_heads=64, num_kv_heads=1,  head_dim=128),
]

SEQ_LENGTHS: List[int] = [128, 256, 512, 1024, 2048, 4096]


# =============================================================================
# Section 2: KV Cache Memory Formula
# =============================================================================

def kvcache_bytes(batch: int, seq_len: int, config: AttentionConfig) -> int:
    """
    Compute theoretical KV cache size in bytes.

    Formula:
        bytes = 2 (K+V) x batch x num_layers x seq_len x num_kv_heads x head_dim x dtype_bytes

    This is a pure function -- no GPU allocation. Used both to verify empirical
    measurements and inside max_batch_for_vram().
    """
    return (
        2                    # K tensor + V tensor
        * batch
        * config.num_layers
        * seq_len
        * config.num_kv_heads
        * config.head_dim
        * config.dtype_bytes
    )


# =============================================================================
# Section 3: Real GPU Allocation
# =============================================================================

def allocate_kvcache(batch: int, seq_len: int, config: AttentionConfig) -> int:
    """
    Allocate real KV cache tensors on GPU and measure actual bytes consumed.

    Each layer's cache is two tensors of shape (batch, num_kv_heads, seq_len, head_dim).
    This matches the layout used by HuggingFace Transformers' LLaMA implementation
    (see modeling_llama.py: past_key_value stores K, V as separate (b, n_kv, s, d) tensors).

    Returns the delta in torch.cuda.memory_allocated() -- exact bytes, no rounding.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    mem_before = torch.cuda.memory_allocated(DEVICE)

    tensors = []
    kv_shape = (batch, config.num_kv_heads, seq_len, config.head_dim)
    for _ in range(config.num_layers):
        k = torch.empty(kv_shape, dtype=config.dtype, device=DEVICE)
        v = torch.empty(kv_shape, dtype=config.dtype, device=DEVICE)
        tensors.append((k, v))

    measured = torch.cuda.memory_allocated(DEVICE) - mem_before

    del tensors
    torch.cuda.empty_cache()
    return measured


# =============================================================================
# Section 4: Max Batch Under VRAM Budget
# =============================================================================

def max_batch_for_vram(
    vram_budget_gb: float,
    seq_len: int,
    config: AttentionConfig,
) -> int:
    """
    Return the largest batch size whose KV cache fits within vram_budget_gb.

    Uses the analytical formula from kvcache_bytes() -- no GPU allocation needed.
    Floor-division gives the exact integer answer: any larger batch would exceed budget.

    Args:
        vram_budget_gb : available VRAM in gigabytes (e.g. 20.0)
        seq_len        : sequence length in tokens
        config         : AttentionConfig instance

    Returns:
        max_batch : largest integer batch that fits (minimum 1)

    Hints:
        - Convert vram_budget_gb to bytes first (1 GB = 1024**3 bytes)
        - kvcache_bytes(batch=1, seq_len, config) gives bytes for one sequence
        - Floor-divide budget by cost-per-sequence
        - Clamp to at least 1 with max(1, ...)
    """
    budget_bytes = int(vram_budget_gb * 1024 ** 3)
    bytes_per_seq = kvcache_bytes(batch=1, seq_len=seq_len, config=config)
    return max(1, budget_bytes // bytes_per_seq)


# =============================================================================
# Section 5: Sweep Runners
# =============================================================================

def run_memory_sweep(
    configs: List[AttentionConfig],
    seq_lengths: List[int],
    batch_size: int = 1,
) -> Dict[str, List[float]]:
    """
    Allocate real KV cache tensors for each (config, seq_len) pair.

    Prints measured vs theoretical GB and their ratio. A ratio slightly above
    1.000 for small tensors reveals CUDA allocator block-alignment -- an effect
    not mentioned in the GQA paper but visible in any real allocation benchmark.
    """
    print(f"\n{'─'*60}")
    print(f"KV cache memory sweep  (batch={batch_size})")
    print(f"{'─'*60}")

    results: Dict[str, List[float]] = {cfg.name: [] for cfg in configs}

    for cfg in configs:
        print(f"\n  [{cfg.name}]  kv_heads={cfg.num_kv_heads}  ratio={cfg.kv_head_ratio:.1f}x")
        for seq_len in seq_lengths:
            measured = allocate_kvcache(batch_size, seq_len, cfg)
            theoretical = kvcache_bytes(batch_size, seq_len, cfg)
            measured_gb = measured / 1024 ** 3
            ratio = measured / theoretical
            results[cfg.name].append(measured_gb)
            print(
                f"    seq={seq_len:4d}  "
                f"measured={measured_gb:.4f} GB  "
                f"formula={theoretical / 1024**3:.4f} GB  "
                f"ratio={ratio:.3f}"
            )

    return results


def run_batch_sweep(
    configs: List[AttentionConfig],
    seq_lengths: List[int],
    vram_budget_gb: float = VRAM_BUDGET_GB,
) -> Dict[str, List[int]]:
    """
    Compute max batch size under vram_budget_gb for each (config, seq_len).

    Uses the analytical formula (no GPU allocation) -- the memory sweep already
    validated that formula against empirical measurements.
    """
    return {
        cfg.name: [max_batch_for_vram(vram_budget_gb, s, cfg) for s in seq_lengths]
        for cfg in configs
    }


# =============================================================================
# Section 6: Charts
# =============================================================================

# Color + style per config -- GQA/MQA get distinct linestyles for greyscale legibility
_STYLE: Dict[str, dict] = {
    "GPT-2 XL (MHA)":    {"color": PALETTE["steel"],      "ls": "-",  "lw": 2.0, "marker": "o"},
    "Llama-2 7B (MHA)":  {"color": PALETTE["slate"],      "ls": "--", "lw": 2.0, "marker": "s"},
    "Llama-2 70B (GQA)": {"color": EXTENDED["light_blue"], "ls": "-.", "lw": 2.2, "marker": "D"},
    "MQA variant":       {"color": EXTENDED["success"],    "ls": ":",  "lw": 2.2, "marker": "^"},
}


def plot_kvcache_memory(
    sweep_results: Dict[str, List[float]],
    seq_lengths: List[int],
    out_dir: Path,
) -> None:
    """Line chart: KV cache GB vs sequence length for all four configs."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, gb_vals in sweep_results.items():
        s = _STYLE[name]
        ax.plot(seq_lengths, gb_vals, label=name,
                color=s["color"], linestyle=s["ls"], linewidth=s["lw"],
                marker=s["marker"], markersize=6,
                markeredgecolor="#FFFFFF", markeredgewidth=1.2)
        # Annotate final point only (seq=4096) to avoid clutter
        ax.annotate(
            f"{gb_vals[-1]:.3f} GB",
            xy=(seq_lengths[-1], gb_vals[-1]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8, color=s["color"], va="center",
        )

    ax.axhline(VRAM_BUDGET_GB, color=EXTENDED["warning"], linewidth=1.4, linestyle="--", alpha=0.8,
               label=f"{VRAM_BUDGET_GB:.0f} GB budget")
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("KV cache (GB, batch=1)")
    ax.set_title("KV Cache Memory vs Sequence Length\nGQA reduces memory by 8x at 70B scale")
    ax.set_xticks(seq_lengths)
    ax.legend(loc="upper left", fontsize=9)
    add_source_label(ax)

    save_path = out_dir / "gqa_kvcache_memory.png"
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {save_path.relative_to(save_path.parent.parent)}")


def plot_max_batch(
    batch_results: Dict[str, List[int]],
    seq_lengths: List[int],
    out_dir: Path,
) -> None:
    """Line chart: max batch size under VRAM budget vs sequence length (log Y scale)."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, batch_vals in batch_results.items():
        s = _STYLE[name]
        ax.plot(seq_lengths, batch_vals, label=name,
                color=s["color"], linestyle=s["ls"], linewidth=s["lw"],
                marker=s["marker"], markersize=6,
                markeredgecolor="#FFFFFF", markeredgewidth=1.2)
        ax.annotate(
            f"{batch_vals[-1]}",
            xy=(seq_lengths[-1], batch_vals[-1]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8, color=s["color"], va="center",
        )

    ax.axhline(1, color=PALETTE["slate"], linewidth=1.0, linestyle="--", alpha=0.6,
               label="batch = 1 (minimum)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel(f"Max batch size (under {VRAM_BUDGET_GB:.0f} GB KV budget)")
    ax.set_title(f"Max Batch Size Under {VRAM_BUDGET_GB:.0f} GB KV Cache Budget\n"
                 "GQA enables 8x larger batches at fixed sequence length")
    ax.set_xticks(seq_lengths)
    ax.legend(loc="upper right", fontsize=9)
    add_source_label(ax)

    save_path = out_dir / "gqa_max_batch.png"
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {save_path.relative_to(save_path.parent.parent)}")


# =============================================================================
# Section 7: Main
# =============================================================================

def main() -> None:
    t0 = time.perf_counter()
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Sanity check: verify the analytical path executes before running sweeps.
    try:
        _ = max_batch_for_vram(VRAM_BUDGET_GB, 128, CONFIGS[0])
    except NotImplementedError as e:
        print(e)
        sys.exit(1)

    # ── Config table ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Architecture configs")
    print(f"{'─'*60}")
    header = f"  {'Config':<24} {'layers':>6}  {'Q heads':>7}  {'KV heads':>8}  {'head_dim':>8}  {'ratio':>7}  dtype"
    print(header)
    for cfg in CONFIGS:
        print(
            f"  {cfg.name:<24} {cfg.num_layers:>6}  {cfg.num_q_heads:>7}  "
            f"{cfg.num_kv_heads:>8}  {cfg.head_dim:>8}  {cfg.kv_head_ratio:>6.1f}x  "
            f"{str(cfg.dtype).split('.')[-1]}"
        )

    # ── Memory sweep ─────────────────────────────────────────────────────────
    sweep_results = run_memory_sweep(CONFIGS, SEQ_LENGTHS, batch_size=1)

    # ── Batch sweep ──────────────────────────────────────────────────────────
    batch_results = run_batch_sweep(CONFIGS, SEQ_LENGTHS, VRAM_BUDGET_GB)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Max batch size under {VRAM_BUDGET_GB:.0f} GB KV cache budget")
    print(f"{'─'*60}")
    seq_header = "  " + f"{'Config':<24}" + "".join(f"  seq={s:<5}" for s in SEQ_LENGTHS)
    print(seq_header)
    for cfg in CONFIGS:
        vals = batch_results[cfg.name]
        row = "  " + f"{cfg.name:<24}" + "".join(f"  {v:<8}" for v in vals)
        print(row)

    # ── GQA compression headline ─────────────────────────────────────────────
    llama7b  = batch_results["Llama-2 7B (MHA)"][-1]   # seq=4096
    llama70b = batch_results["Llama-2 70B (GQA)"][-1]
    mqa      = batch_results["MQA variant"][-1]
    print(f"\n[GQA compression summary at seq=4096 under {VRAM_BUDGET_GB:.0f} GB budget]")
    print(f"  Llama-2 7B  (MHA, ratio=1.0x) : max_batch = {llama7b:>5}")
    print(f"  Llama-2 70B (GQA, ratio=8.0x) : max_batch = {llama70b:>5}  "
          f"(+{(llama70b/llama7b - 1)*100:.0f}% vs MHA 7B)")
    print(f"  MQA variant (ratio=64.0x)      : max_batch = {mqa:>5}  "
          f"(+{(mqa/llama7b - 1)*100:.0f}% vs MHA 7B)")

    llama70b_kv_gb = sweep_results["Llama-2 70B (GQA)"][-1]
    hypothetical_mha_gb = llama70b_kv_gb * 8   # what 70B would cost with MHA
    print(f"\n  70B KV cache at seq=4096 batch=1:")
    print(f"    Hypothetical MHA (64 KV heads) : {hypothetical_mha_gb:.3f} GB")
    print(f"    Actual GQA      (8  KV heads)  : {llama70b_kv_gb:.3f} GB  ({hypothetical_mha_gb/llama70b_kv_gb:.1f}x reduction)")

    # ── Charts ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Generating plots  ->  {out_dir}")
    print(f"{'─'*60}")
    plot_kvcache_memory(sweep_results, SEQ_LENGTHS, out_dir)
    plot_max_batch(batch_results, SEQ_LENGTHS, out_dir)

    elapsed = time.perf_counter() - t0
    print(f"\n[Done]  Total elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
