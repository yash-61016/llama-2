"""
RoPE (Rotary Position Embedding) -- From-Scratch Implementation
================================================================
Papers:
  RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv: 2104.09864)
  Llama 2: Open Foundation and Fine-Tuned Chat Models (arXiv: 2307.09288)

What this implements:
  Rotary Position Embedding in pure NumPy. Proves numerically that the dot product
  of a rotated query at position m and a rotated key at position n depends only on
  (m - n), not on m or n individually. Plots the resulting attention decay curve
  against a sinusoidal additive baseline.

Hardware: CPU-only (no GPU required -- pure NumPy)
Time to run: ~45 seconds (decay curve: 200 pairs x 128 distances)

Dependencies (pinned):
    numpy==1.24.3       pip install numpy==1.24.3
    matplotlib==3.7.1   pip install matplotlib==3.7.1

Run: python rope_from_scratch.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Deep Sea chart theme -- consistent styling across the LLM Engineering Stack series
sys.path.insert(0, str(Path(__file__).parent.parent / "theme"))
from deep_sea_theme import apply_theme, PALETTE, EXTENDED, add_source_label
apply_theme()

print("=" * 60)
print("RoPE From Scratch  |  LLaMA 2 / RoFormer")
print("=" * 60)


# =============================================================================
# Section 1: Configuration and Data
# =============================================================================

# LLaMA 2 head dimensions from Table 2, arXiv:2307.09288
HEAD_DIM   = 128       # 4096 hidden dim / 32 heads = 128 per head
THETA_BASE = 10000.0   # base frequency for the geometric schedule, same as Vaswani 2017

# A concrete sentence whose tokens span varied semantic distances.
# Positions here correspond to actual token indices in a real forward pass.
SENTENCE = "the rotary embedding encodes relative position directly into the dot product of queries and keys"
TOKENS   = SENTENCE.split()
SEQ_LEN  = len(TOKENS)

print(f"\nSentence : '{SENTENCE}'")
print(f"Tokens   : {TOKENS}")
print(f"Seq len  : {SEQ_LEN}  |  Head dim: {HEAD_DIM}  |  Theta base: {THETA_BASE}")

# Simulate post-RMSNorm Q and K activations.
# LLaMA 2 applies RMSNorm before Q/K/V projections (Section 2.1, arXiv:2307.09288),
# so the vectors entering the rotation step have approximately unit norm.
rng   = np.random.default_rng(42)
Q_raw = rng.standard_normal((SEQ_LEN, HEAD_DIM)).astype(np.float64)
K_raw = rng.standard_normal((SEQ_LEN, HEAD_DIM)).astype(np.float64)
Q_raw /= np.linalg.norm(Q_raw, axis=-1, keepdims=True)
K_raw /= np.linalg.norm(K_raw, axis=-1, keepdims=True)

print(f"\nQ_raw  shape: {Q_raw.shape},  ||Q_raw[0]||  = {np.linalg.norm(Q_raw[0]):.6f}")
print(f"K_raw  shape: {K_raw.shape},  ||K_raw[0]||  = {np.linalg.norm(K_raw[0]):.6f}")


# =============================================================================
# Section 2: RoPE Frequency Schedule
# =============================================================================

def compute_rope_freqs(head_dim: int, seq_len: int, theta_base: float = 10000.0) -> np.ndarray:
    """
    Compute per-position rotation angles for each dimension pair.

    Returns shape (seq_len, head_dim // 2):
        freqs[m, k] = m * theta_base^(-2k / head_dim)

    The frequency schedule is geometric: pair 0 rotates at 1 rad/token,
    pair head_dim//2 - 1 rotates at theta_base^(-1) ~ 1e-4 rad/token.
    This 10,000x spread is what lets a single model distinguish tokens at
    distance 1 (via high-frequency pairs that saturate quickly) and tokens at
    distance 100k+ (via low-frequency pairs that have barely rotated by then).

    Sinusoidal PE uses the same frequency schedule but ADDS the encoding to the
    input. The rotation structure here is the key choice that makes the dot product
    depend only on relative distance -- sinusoidal addition does not have this property.
    """
    exponents = np.arange(0, head_dim, 2, dtype=np.float64) / head_dim  # (head_dim//2,)
    inv_freqs = 1.0 / (theta_base ** exponents)                          # (head_dim//2,)
    positions = np.arange(seq_len, dtype=np.float64)                    # (seq_len,)
    freqs     = np.outer(positions, inv_freqs)                           # (seq_len, head_dim//2)
    return freqs


freqs = compute_rope_freqs(HEAD_DIM, SEQ_LEN, THETA_BASE)
print(f"\n[compute_rope_freqs]")
print(f"  Output shape    : {freqs.shape}")
print(f"  freqs[0]        : all zeros  (position 0 applies identity rotation)")
print(f"  freqs[1,  :4]   : {np.round(freqs[1, :4], 6)}  (fast-rotating pairs)")
print(f"  freqs[1, -4:]   : {np.round(freqs[1, -4:], 10)}  (slow-rotating pairs)")
print(f"  Fastest/slowest : {freqs[1, 0] / freqs[1, -1]:.0f}x ratio at position 1")


# =============================================================================
# Section 3: Apply RoPE
# =============================================================================

def apply_rope(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Apply rotary position embedding to a batch of query or key vectors.

    Args:
        x     : (seq_len, head_dim)       -- raw Q or K after linear projection
        freqs : (seq_len, head_dim // 2)  -- rotation angles from compute_rope_freqs()

    Returns:
        x_rot : (seq_len, head_dim)       -- rotationally encoded vectors

    Each consecutive pair (x[m, 2k], x[m, 2k+1]) is treated as a 2D point and
    rotated by angle freqs[m, k]. In complex notation, if z = x[2k] + i*x[2k+1],
    the rotated value is z_rot = z * exp(i * freqs[m, k]).

    Expanding the complex multiplication into real operations:
        x_rot[m, 2k]   = x[m, 2k] * cos(freqs[m, k]) - x[m, 2k+1] * sin(freqs[m, k])
        x_rot[m, 2k+1] = x[m, 2k] * sin(freqs[m, k]) + x[m, 2k+1] * cos(freqs[m, k])

    Rotation is isometric: ||x_rot[m]|| == ||x[m]|| for all m. This preserves the
    scale of attention logits -- additive position encodings inflate the vector norm
    by the PE magnitude, which shifts the softmax distribution in an uncontrolled way.
    """
    x_even = x[:, 0::2]   # (seq_len, head_dim//2)
    x_odd  = x[:, 1::2]   # (seq_len, head_dim//2)
    cos_f  = np.cos(freqs) 
    sin_f  = np.sin(freqs)
    x_rot_even = x_even * cos_f - x_odd * sin_f
    x_rot_odd  = x_even * sin_f + x_odd * cos_f
    x_rot = np.stack([x_rot_even, x_rot_odd], axis=-1).reshape(x.shape[0], -1)
    return x_rot



# =============================================================================
# Section 4: Sinusoidal Baseline
# =============================================================================

def sinusoidal_encoding(seq_len: int, head_dim: int, theta_base: float = 10000.0) -> np.ndarray:
    """
    Additive sinusoidal positional encoding from Vaswani et al. 2017 (arXiv:1706.03762).

    Returns shape (seq_len, head_dim):
        PE[m, 2k]   = sin(m / theta_base^(2k / head_dim))
        PE[m, 2k+1] = cos(m / theta_base^(2k / head_dim))

    Uses the same frequency schedule as RoPE for a fair comparison.
    The structural difference is how positions combine with content:

        (q + PE[m]) . (k + PE[n])
            = q.k  +  q.PE[n]  +  PE[m].k  +  PE[m].PE[n]

    The cross terms q.PE[n] and PE[m].k depend on absolute positions m and n
    separately. No algebraic simplification reduces these to a function of
    (m - n) alone -- that is why sinusoidal PE does not generalize cleanly to
    sequences longer than those seen during training.
    """
    exponents   = np.arange(0, head_dim, 2, dtype=np.float64) / head_dim
    inv_freqs   = 1.0 / (theta_base ** exponents)
    positions   = np.arange(seq_len, dtype=np.float64)
    angles      = np.outer(positions, inv_freqs)
    pe          = np.zeros((seq_len, head_dim), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


# =============================================================================
# Section 5: Numerical Proof of Relative Distance Property
# =============================================================================

def verify_relative_distance(q_all: np.ndarray, k_all: np.ndarray,
                              freqs_seq: np.ndarray) -> None:
    """
    Numerically verify that RoPE encodes relative position, not absolute position.

    For any positions m, n, and shift delta:
        dot(q_rope[m], k_rope[n]) = dot(q_rope[m+delta], k_rope[n+delta])

    The proof in the RoFormer paper (Section 3.4, arXiv:2104.09864) is algebraic.
    This function confirms it numerically for 20 random (m, n, delta) triples and
    verifies our real-valued implementation against the canonical complex reference.
    """
    MAX_POS    = 512
    freqs_long = compute_rope_freqs(HEAD_DIM, MAX_POS, THETA_BASE)
    q1         = q_all[:1]   # single query vector, shape (1, HEAD_DIM)
    k1         = k_all[:1]   # single key vector

    def apply_rope_complex_ref(x: np.ndarray, freqs_r: np.ndarray) -> np.ndarray:
        """
        Reference implementation using NumPy complex numbers.
        z_rot = z * exp(i*theta) is mathematically identical to the real-valued
        rotation matrix; serves as ground truth to verify apply_rope() against.
        """
        z     = x[:, 0::2] + 1j * x[:, 1::2]             # (seq_len, head_dim//2)
        z_rot = z * np.exp(1j * freqs_r)                   # rotate: z * e^{i*theta}
        return np.stack([z_rot.real, z_rot.imag], axis=-1).reshape(x.shape[0], -1)

    print("\n[Numerical proof: relative distance property]")
    rng_test = np.random.default_rng(7)
    max_err  = 0.0
    for _ in range(20):
        m     = int(rng_test.integers(0, 200))
        n     = int(rng_test.integers(0, 200))
        delta = int(rng_test.integers(1, 100))
        q_m   = apply_rope(q1, freqs_long[m:m+1])
        k_n   = apply_rope(k1, freqs_long[n:n+1])
        q_md  = apply_rope(q1, freqs_long[m+delta:m+delta+1])
        k_nd  = apply_rope(k1, freqs_long[n+delta:n+delta+1])
        max_err = max(max_err, abs(float(q_m @ k_n.T) - float(q_md @ k_nd.T)))

    print(f"  Max |dot(q[m],k[n]) - dot(q[m+d],k[n+d])| over 20 triples: {max_err:.2e}")
    print(f"  Expected: < 1e-12  (floating-point rounding, not approximation)")

    # Verify our real-valued rotation matches the complex reference exactly
    q_ref  = apply_rope_complex_ref(q_all, freqs_seq)
    q_ours = apply_rope(q_all, freqs_seq)
    k_ref  = apply_rope_complex_ref(k_all, freqs_seq)
    k_ours = apply_rope(k_all, freqs_seq)
    print(f"\n[Verification vs. complex reference]")
    print(f"  Max |q_complex - q_ours| : {np.max(np.abs(q_ref - q_ours)):.2e}")
    print(f"  Max |k_complex - k_ours| : {np.max(np.abs(k_ref - k_ours)):.2e}")
    print(f"  Expected: < 1e-14  (exact same computation, different notation)")


# =============================================================================
# Section 6: Attention Decay Curves
# =============================================================================

def compute_decay_curves(n_pairs: int = 200, max_dist: int = 128) -> tuple:
    """
    Measure E[|q_rope[0] . k_rope[d]|] as a function of distance d,
    averaged over n_pairs random unit-norm (q, k) pairs.

    Averaging over many random content vectors isolates the purely positional
    effect. At d=0, the rotation is identical on both vectors, so the dot
    product matches the unrotated case. At large d, the per-dimension rotation
    phases become progressively misaligned, causing the expected dot product
    to decay toward zero.

    The sinusoidal baseline adds PE to q and k and computes cosine similarity.
    The mean-zero cross terms (q.PE[n], PE[m].k) average out over n_pairs,
    leaving a signal dominated by the PE[0].PE[d] term -- which is deterministic
    and does NOT decay monotonically, showing the lack of a clean locality bias.
    """
    rng_decay  = np.random.default_rng(123)
    freqs_long = compute_rope_freqs(HEAD_DIM, max_dist, THETA_BASE)
    pe_long    = sinusoidal_encoding(max_dist, HEAD_DIM, THETA_BASE)

    Q_batch = rng_decay.standard_normal((n_pairs, HEAD_DIM))
    K_batch = rng_decay.standard_normal((n_pairs, HEAD_DIM))
    Q_batch /= np.linalg.norm(Q_batch, axis=-1, keepdims=True)
    K_batch /= np.linalg.norm(K_batch, axis=-1, keepdims=True)

    # freqs[0] = all zeros, so Q_rope_0 = Q_batch (identity rotation at position 0)
    freqs_pos0 = np.tile(freqs_long[0:1], (n_pairs, 1))
    Q_rope_0   = apply_rope(Q_batch, freqs_pos0)

    # pe_long[0] = [0, 1, 0, 1, ...] (sin(0)=0, cos(0)=1 interleaved)
    Q_sin_0  = Q_batch + pe_long[0]
    Q_sin_0 /= np.linalg.norm(Q_sin_0, axis=-1, keepdims=True)

    rope_means = np.zeros(max_dist)
    sin_means  = np.zeros(max_dist)

    for d in range(max_dist):
        freqs_d  = np.tile(freqs_long[d:d+1], (n_pairs, 1))
        K_rope_d = apply_rope(K_batch, freqs_d)
        rope_means[d] = np.mean(np.abs(np.sum(Q_rope_0 * K_rope_d, axis=-1)))

        K_sin_d  = K_batch + pe_long[d]
        K_sin_d /= np.linalg.norm(K_sin_d, axis=-1, keepdims=True)
        sin_means[d] = np.mean(np.abs(np.sum(Q_sin_0 * K_sin_d, axis=-1)))

    return rope_means, sin_means


# =============================================================================
# Section 7: Visualizations
# =============================================================================

def plot_decay_curves(rope_means: np.ndarray, sin_means: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    distances = np.arange(len(rope_means))

    labels   = ["RoPE", "Sinusoidal (additive)"]
    colors   = [PALETTE["steel"], PALETTE["slate"]]
    styles   = ["-", "--"]
    markers  = ["o", "s"]

    for ax, log_scale in zip(axes, [False, True]):
        for label, color, ls, marker, data in zip(labels, colors, styles, markers,
                                                  [rope_means, sin_means]):
            x = distances[1:] if log_scale else distances
            y = data[1:]      if log_scale else data
            if log_scale:
                ax.semilogy(
                    x,
                    y,
                    color=color,
                    lw=2.5,
                    label=label,
                    linestyle=ls,
                    marker=marker,
                    markersize=5,
                    markeredgecolor="#FFFFFF",
                    markeredgewidth=1.2,
                    markevery=12,
                )
            else:
                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=2.5,
                    label=label,
                    linestyle=ls,
                    marker=marker,
                    markersize=5,
                    markeredgecolor="#FFFFFF",
                    markeredgewidth=1.2,
                    markevery=12,
                )

        ax.set_xlabel("Relative distance (tokens)")
        ax.set_ylabel("E[|q . k|]" + (" (log scale, unitless)" if log_scale else " (unitless)"))
        ax.set_title("Tail Behavior (log scale)" if log_scale
                     else "Attention Magnitude vs. Relative Distance")
        ax.legend()
        add_source_label(ax)

    plt.tight_layout()
    save_path = out_dir / "rope_decay_curves.png"
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_rotation_heatmap(out_dir: Path) -> None:
    """
    Visualize freqs[position, k] for the first 64 positions.
    Rows correspond to dimension pairs; columns to token positions.
    High-index rows (slow pairs) barely move across the plot.
    Low-index rows (fast pairs) complete multiple full rotations.
    This spread is what lets the model disambiguate tokens at any distance.
    """
    freqs_vis = compute_rope_freqs(HEAD_DIM, 64, THETA_BASE)   # (64, 64)
    fig, ax   = plt.subplots(figsize=(11, 5))
    im = ax.imshow(
        (freqs_vis % (2 * np.pi)).T,   # wrap to [0, 2pi] to show periodicity clearly
        aspect="auto",
        cmap="deep_sea_light",
        origin="lower",
    )
    plt.colorbar(im, ax=ax, label="Rotation angle (rad, mod 2pi)")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Dimension pair index k  (0 = fastest, 63 = slowest)")
    ax.set_title("RoPE rotation angles: freqs[position, k] mod 2pi")
    add_source_label(ax)
    plt.tight_layout()
    save_path = out_dir / "rope_rotation_heatmap.png"
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # Step 1: Apply RoPE to the full sequence
    print("\n[Applying RoPE]")
    Q_rope = apply_rope(Q_raw, freqs)
    K_rope = apply_rope(K_raw, freqs)
    print(f"  Q_rope shape  : {Q_rope.shape}")
    print(f"  K_rope shape  : {K_rope.shape}")
    print(f"  ||Q_raw[0]||  : {np.linalg.norm(Q_raw[0]):.6f}")
    print(f"  ||Q_rope[0]|| : {np.linalg.norm(Q_rope[0]):.6f}  (norm preserved by rotation)")

    # Step 2: Inspect the rotation on concrete token vectors
    print(f"\n[Token '{TOKENS[0]}' at position 0  -- first 8 dims]")
    print(f"  Q_raw[0,  :8] : {np.round(Q_raw[0,  :8], 4)}")
    print(f"  Q_rope[0, :8] : {np.round(Q_rope[0, :8], 4)}  (identity at pos 0 -- freqs[0] = 0)")
    print(f"\n[Token '{TOKENS[1]}' at position 1  -- first 8 dims]")
    print(f"  Q_raw[1,  :8] : {np.round(Q_raw[1,  :8], 4)}")
    print(f"  Q_rope[1, :8] : {np.round(Q_rope[1, :8], 4)}  (rotated by freqs[1])")

    # Step 3: Show how RoPE changes which tokens the query attends to
    print(f"\n[Attention scores: query='{TOKENS[0]}' (pos 0) vs all keys]")
    scores_raw  = Q_raw[0]  @ K_raw.T
    scores_rope = Q_rope[0] @ K_rope.T
    top_raw  = np.argsort(scores_raw)[-3:][::-1].tolist()
    top_rope = np.argsort(scores_rope)[-3:][::-1].tolist()
    print(f"  Without RoPE  top 3: {[(i, TOKENS[i]) for i in top_raw]}")
    print(f"  With RoPE     top 3: {[(i, TOKENS[i]) for i in top_rope]}")

    # Step 4: Numerical proof of relative distance property
    verify_relative_distance(Q_raw, K_raw, freqs)

    # Step 5: Compute decay curves (timing: 200 pairs x 128 distances)
    print(f"\n[Computing decay curves  (200 pairs x 128 distances)...]")
    start = time.time()
    rope_decay, sin_decay = compute_decay_curves(n_pairs=200, max_dist=128)
    print(f"  Elapsed: {time.time() - start:.2f}s")
    print(f"  RoPE   E[|dot|]  d=0  : {rope_decay[0]:.4f}")
    print(f"  RoPE   E[|dot|]  d=64 : {rope_decay[64]:.4f}  "
          f"({rope_decay[64] / rope_decay[0] * 100:.1f}% of d=0)")
    print(f"  RoPE   E[|dot|] d=127 : {rope_decay[127]:.4f}  "
          f"({rope_decay[127] / rope_decay[0] * 100:.1f}% of d=0)")
    print(f"  Sinusoidal      d=64  : {sin_decay[64]:.4f}  "
          f"({sin_decay[64] / sin_decay[0] * 100:.1f}% of d=0)")
    print(f"  Sinusoidal      d=127 : {sin_decay[127]:.4f}  "
          f"({sin_decay[127] / sin_decay[0] * 100:.1f}% of d=0)")

    # Step 6: Generate plots
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    print(f"\n[Generating plots -> {out_dir}]")
    plot_decay_curves(rope_decay, sin_decay, out_dir)
    plot_rotation_heatmap(out_dir)

    print(f"\n[Summary]")
    print(f"  RoPE decay ratio   (d=127 / d=0) : {rope_decay[-1] / rope_decay[0]:.4f}")
    print(f"  Sinusoidal decay ratio             : {sin_decay[-1] / sin_decay[0]:.4f}")
    rope_ratio = rope_decay[0] / rope_decay[-1]
    sin_ratio  = sin_decay[0]  / sin_decay[-1]
    print(f"  RoPE shows {rope_ratio / sin_ratio:.1f}x stronger locality signal at distance 127")
