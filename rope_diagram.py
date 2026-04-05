"""
RoPE Rotation Diagram
=====================
Visualises the two-panel diagram described in llama-blog.md:

  Left  — query vector q at position m and key vector k at position n,
           both decomposed into a single 2D dimension pair, rotated by
           m·θ and n·θ respectively.
  Right — the dot-product identity showing that only the *relative*
           angle (m-n)·θ enters the computation.

Render (no LaTeX required — uses manimpango Text renderer):
    manim -ql --save_last_frame rope_diagram.py RopeDiagram
    # output: media/images/rope_diagram/RopeDiagram_ManimCE_*.png

Colors: Deep Sea palette (#415A77 steel, #778DA9 slate, #E8927C highlight,
        #6B9E7D success, #0D1B2A navy, #E0E1DD bone)
"""

import sys
from pathlib import Path

# ── Deep Sea palette — sourced from shared theme module ───────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "theme"))
from deep_sea_theme import PALETTE, EXTENDED

from manim import *

NAVY      = PALETTE["navy"]
DARK_BLUE = PALETTE["dark_blue"]
STEEL     = PALETTE["steel"]
SLATE     = PALETTE["slate"]
BONE      = PALETTE["bone"]
TEAL      = EXTENDED["teal"]
HIGHLIGHT = EXTENDED["highlight"]   # warm accent — query vector / m·θ
SUCCESS   = EXTENDED["success"]     # green accent — key vector / n·θ
WARNING   = EXTENDED["warning"]     # amber — relative angle arc


def ds_text(s, size=18, color=BONE, **kwargs):
    """Convenience wrapper: Text with consistent font."""
    return Text(s, font_size=size, color=color, **kwargs)


class RopeDiagram(Scene):
    def construct(self):
        self.camera.background_color = NAVY

        # ── Title ─────────────────────────────────────────────────────────────
        title = ds_text("Rotary Position Embedding (RoPE)", size=28, color=BONE)
        title.to_edge(UP, buff=0.3)

        subtitle = ds_text(
            "Relative position enters the dot product — absolute positions vanish",
            size=15, color=SLATE,
        ).next_to(title, DOWN, buff=0.12)
        self.add(title, subtitle)

        # ── Centre divider ────────────────────────────────────────────────────
        divider = Line(UP * 2.7, DOWN * 2.8, color=STEEL, stroke_width=1)
        self.add(divider)

        # ═════════════════════════════════════════════════════════════════════
        # LEFT PANEL — geometric view
        # ═════════════════════════════════════════════════════════════════════
        left_center = LEFT * 3.3

        panel_left_label = ds_text("Geometric View", size=18, color=SLATE)
        panel_left_label.move_to(left_center + UP * 2.45)
        self.add(panel_left_label)

        # Number plane (unit grid)
        plane = NumberPlane(
            x_range=[-1.6, 1.6, 1],
            y_range=[-1.6, 1.6, 1],
            x_length=3.6,
            y_length=3.6,
            background_line_style={"stroke_color": DARK_BLUE, "stroke_width": 1},
            axis_config={"stroke_color": STEEL, "stroke_width": 1.5},
        ).move_to(left_center + DOWN * 0.15)
        self.add(plane)

        origin = plane.c2p(0, 0)

        # Base (unrotated) vector — ghost reference
        base_angle = 35 * DEGREES
        base_tip   = plane.c2p(np.cos(base_angle), np.sin(base_angle))
        ghost_arrow = Arrow(
            origin, base_tip, buff=0,
            stroke_width=1.5, color=STEEL,
            tip_length=0.15,
            max_stroke_width_to_length_ratio=999,
        )
        ghost_label = ds_text("xₖ", size=18, color=SLATE).next_to(
            ghost_arrow.get_end(), UR, buff=0.08
        )
        self.add(ghost_arrow, ghost_label)

        # Query vector — rotated by m·θ  (m=2, θ=25°)
        m, theta = 2, 25 * DEGREES
        q_angle = base_angle + m * theta
        q_tip   = plane.c2p(np.cos(q_angle), np.sin(q_angle))

        q_arrow = Arrow(
            origin, q_tip, buff=0,
            stroke_width=3, color=HIGHLIGHT,
            tip_length=0.18,
            max_stroke_width_to_length_ratio=999,
        )
        q_label = ds_text("R_mθ · qₖ", size=17, color=HIGHLIGHT).next_to(
            q_arrow.get_end(), RIGHT, buff=0.1
        )

        # Key vector — rotated by n·θ  (n=5, θ=25°)
        n = 5
        k_angle = base_angle + n * theta
        k_tip   = plane.c2p(np.cos(k_angle), np.sin(k_angle))

        k_arrow = Arrow(
            origin, k_tip, buff=0,
            stroke_width=3, color=SUCCESS,
            tip_length=0.18,
            max_stroke_width_to_length_ratio=999,
        )
        k_label = ds_text("R_nθ · kₖ", size=17, color=SUCCESS).next_to(
            k_arrow.get_end(), LEFT, buff=0.1
        )

        # Arc: m·θ from base to query
        arc_m = Arc(
            radius=0.52,
            start_angle=base_angle,
            angle=m * theta,
            color=HIGHLIGHT, stroke_width=2,
        ).move_arc_center_to(origin)
        arc_m_mid = arc_m.point_from_proportion(0.5)
        arc_m_label = ds_text("mθₖ", size=15, color=HIGHLIGHT).move_to(
            arc_m_mid + normalize(arc_m_mid - origin) * 0.32
        )

        # Arc: n·θ from base to key
        arc_n = Arc(
            radius=0.36,
            start_angle=base_angle,
            angle=n * theta,
            color=SUCCESS, stroke_width=2,
        ).move_arc_center_to(origin)
        arc_n_mid = arc_n.point_from_proportion(0.5)
        arc_n_label = ds_text("nθₖ", size=15, color=SUCCESS).move_to(
            arc_n_mid + normalize(arc_n_mid - origin) * 0.32
        )

        # Relative arc between q and k  (the key insight)
        arc_rel = Arc(
            radius=0.70,
            start_angle=q_angle,
            angle=(n - m) * theta,
            color=WARNING, stroke_width=2.5,
        ).move_arc_center_to(origin)
        arc_rel_mid = arc_rel.point_from_proportion(0.5)
        arc_rel_label = ds_text("(m−n)θₖ", size=14, color=WARNING).move_to(
            arc_rel_mid + normalize(arc_rel_mid - origin) * 0.36
        )

        self.add(
            q_arrow, k_arrow,
            arc_m, arc_n, arc_rel,
            arc_m_label, arc_n_label, arc_rel_label,
            q_label, k_label,
        )

        # Position legend
        pos_m = ds_text(f"pos m={m}", size=13, color=HIGHLIGHT)
        pos_n = ds_text(f"pos n={n}", size=13, color=SUCCESS)
        pos_m.move_to(left_center + DOWN * 2.38 + LEFT * 0.6)
        pos_n.next_to(pos_m, RIGHT, buff=0.5)
        self.add(pos_m, pos_n)

        # ═════════════════════════════════════════════════════════════════════
        # RIGHT PANEL — algebraic identity
        # ═════════════════════════════════════════════════════════════════════
        right_center = RIGHT * 3.3

        panel_right_label = ds_text("Algebraic Identity", size=18, color=SLATE)
        panel_right_label.move_to(right_center + UP * 2.45)
        self.add(panel_right_label)

        # ── Step 1: rotation matrix ───────────────────────────────────────────
        step1_head = ds_text("1. Rotate each dimension pair:", size=15, color=BONE)
        step1_head.move_to(right_center + UP * 1.85)

        # Draw rotation matrix as Text (2×2 visual)
        mat_lines = VGroup(
            ds_text("⎡ cos(mθₖ)  −sin(mθₖ) ⎤", size=14, color=TEAL),
            ds_text("⎣ sin(mθₖ)   cos(mθₖ) ⎦", size=14, color=TEAL),
        ).arrange(DOWN, buff=0.06).move_to(right_center + UP * 1.2)

        vec_q = ds_text("[ q₂ₖ, q₂ₖ₊₁ ]ᵀ", size=14, color=HIGHLIGHT).next_to(mat_lines, RIGHT, buff=0.15)

        self.add(step1_head, mat_lines, vec_q)

        # ── Step 2: dot product expansion ────────────────────────────────────
        step2_head = ds_text("2. Dot product of rotated q and k:", size=15, color=BONE)
        step2_head.move_to(right_center + UP * 0.3)

        lhs = ds_text("⟨ R_mθ·q, R_nθ·k ⟩", size=18, color=BONE)
        eq1  = ds_text("=", size=18, color=SLATE)
        rhs1 = ds_text("qᵀ · R_mθᵀ · R_nθ · k", size=18, color=BONE)

        # colour-code q/k parts
        step2_eq = VGroup(lhs, eq1, rhs1).arrange(RIGHT, buff=0.2)
        step2_eq.move_to(right_center + DOWN * 0.25)

        # Use coloured sub-labels below
        lhs_note_q = ds_text("query (pos m)", size=12, color=HIGHLIGHT).next_to(lhs, DOWN, buff=0.05)
        lhs_note_k = ds_text("key (pos n)", size=12, color=SUCCESS).next_to(lhs_note_q, RIGHT, buff=0.4)

        self.add(step2_head, step2_eq, lhs_note_q, lhs_note_k)

        # ── Step 3: telescoping ───────────────────────────────────────────────
        step3_head = ds_text("3. Rotation matrices telescope:", size=15, color=BONE)
        step3_head.move_to(right_center + DOWN * 1.1)

        step3_lhs = ds_text("R_mθᵀ · R_nθ", size=20, color=TEAL)
        step3_eq  = ds_text("=", size=20, color=SLATE)
        step3_rhs = ds_text("R_(n−m)θ", size=20, color=WARNING)
        step3_row = VGroup(step3_lhs, step3_eq, step3_rhs).arrange(RIGHT, buff=0.25)
        step3_row.move_to(right_center + DOWN * 1.65)
        self.add(step3_head, step3_row)

        # ── Key insight box ───────────────────────────────────────────────────
        insight_bg = RoundedRectangle(
            width=5.8, height=0.82,
            corner_radius=0.1,
            color=STEEL,
            fill_color=DARK_BLUE,
            fill_opacity=0.92,
            stroke_width=1.5,
        ).move_to(right_center + DOWN * 2.52)

        insight_lhs  = ds_text("∴  ⟨ R_mθ·q, R_nθ·k ⟩", size=18, color=BONE)
        insight_eq   = ds_text("=", size=18, color=SLATE)
        insight_rhs  = ds_text("f((m−n)·θₖ)", size=18, color=WARNING)
        insight_row  = VGroup(insight_lhs, insight_eq, insight_rhs).arrange(RIGHT, buff=0.18)
        insight_row.move_to(insight_bg)
        self.add(insight_bg, insight_row)

        # ── Footer ───────────────────────────────────────────────────────────
        footer = ds_text(
            "Only relative distance (m−n) enters — absolute positions m, n vanish",
            size=13, color=SLATE,
        ).to_edge(DOWN, buff=0.18)
        self.add(footer)
