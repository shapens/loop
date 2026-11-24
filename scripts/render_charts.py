#!/usr/bin/env python3
"""Generate clean archival-editorial SVG charts from saved loop run JSONs."""

from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _e(s: str) -> str:
    return html.escape(s, quote=True)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "docs" / "assets"

# Warm archival paper editorial palette (legible on both GitHub light & dark themes)
PAPER = "#F5F2EB"
PAPER_DEEP = "#E7E1D5"
RULE = "#DDD6C9"
RULE_STRONG = "#B8B0A2"
INK = "#141413"
MUTED = "#6F6A60"

C_FOREST = "#1C5D43"
C_TERRA = "#C6462B"
C_SLATE = "#2E5A88"
C_OCHRE = "#B3741E"
C_PLUM = "#7A4B7F"
C_STONE = "#8A8377"


def _svg_header(w: int, h: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="{_e(title)}">\n'
        f"<style>"
        f"text {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}"
        f".title {{ font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif; "
        f"font-size: 14.5px; font-weight: 600; fill: {INK}; letter-spacing: -0.01em; }}"
        f".sub {{ font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif; "
        f"font-size: 11px; fill: {MUTED}; }}"
        f".ax {{ font-size: 10px; fill: {MUTED}; }}"
        f".leg {{ font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif; "
        f"font-size: 10.5px; fill: {INK}; }}"
        f".lbl {{ font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif; "
        f"font-size: 11.5px; font-weight: 500; fill: {INK}; }}"
        f".grid {{ stroke: {RULE}; stroke-width: 1; stroke-dasharray: 2 4; }}"
        f".axis {{ stroke: {RULE_STRONG}; stroke-width: 1.2; }}"
        f"</style>\n"
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="6" '
        f'fill="{PAPER}" stroke="{RULE}"/>\n'
    )


def _fmt_pct(v: float) -> str:
    pct = v * 100.0
    if abs(pct - round(pct)) < 0.05:
        return f"{round(pct)}%"
    return f"{pct:.1f}%"


def _line_plot(
    x0: float,
    y0: float,
    pw: float,
    ph: float,
    *,
    title: str,
    subtitle: str,
    x_max: float,
    x_min: float = 0.0,
    y_max: float,
    y_ticks: list[float],
    x_ticks: list[int],
    y_fmt: str,
    series: list[dict[str, Any]],
    band: dict[str, Any] | None = None,
    legend_cols: int | None = None,
) -> str:
    parts: list[str] = []
    parts.append(f'<text x="{x0 + 18}" y="{y0 + 26}" class="title">{_e(title)}</text>')
    parts.append(f'<text x="{x0 + 18}" y="{y0 + 43}" class="sub">{_e(subtitle)}</text>')

    pl, pr, pt, pb = x0 + 50, x0 + pw - 20, y0 + 62, y0 + ph - 66
    iw, ih = pr - pl, pb - pt
    x_span = max(x_max - x_min, 1e-9)

    def sx(v: float) -> float:
        return pl + ((v - x_min) / x_span) * iw

    def sy(v: float) -> float:
        return pb - (min(max(v, 0.0), y_max) / max(y_max, 1e-9)) * ih

    for idx, v in enumerate(y_ticks):
        yy = sy(v)
        cls = "axis" if idx == 0 else "grid"
        parts.append(f'<line x1="{pl:.1f}" x2="{pr:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" class="{cls}"/>')
        if y_fmt == "pct":
            lbl = f"{round(v * 100)}%"
        elif y_fmt == "int":
            lbl = f"{round(v)}"
        else:
            lbl = f"{v:.1f}"
        parts.append(f'<text x="{pl - 9:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" class="ax">{lbl}</text>')

    for idx, xt in enumerate(x_ticks):
        xx = sx(xt)
        txt = f"Step {xt}" if idx == 0 else f"{xt}"
        parts.append(f'<text x="{xx:.1f}" y="{pb + 17:.1f}" text-anchor="middle" class="ax">{txt}</text>')

    if band:
        xs, lo, hi, col = band["xs"], band["lo"], band["hi"], band["color"]
        top = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, hi, strict=True))
        bot = " ".join(
            f"{sx(x):.1f},{sy(y):.1f}"
            for x, y in reversed(list(zip(xs, lo, strict=True)))
        )
        parts.append(f'<polygon points="{top} {bot}" fill="{col}" fill-opacity="0.12"/>')

    for s in series:
        xs, ys, col = s["xs"], s["ys"], s["color"]
        if not xs:
            continue
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
            for i, (x, y) in enumerate(zip(xs, ys, strict=True))
        )
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        op = f' stroke-opacity="{s.get("opacity", 1.0)}"'
        parts.append(
            f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{s.get("width", 2.3)}"'
            f'{dash}{op} stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if s.get("dots", True):
            r = s.get("r", 3.2)
            for x, y in zip(xs, ys, strict=True):
                parts.append(
                    f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r}" fill="{col}" stroke="{PAPER}" stroke-width="1"/>'
                )

    # Clean legend below x-axis with generous breathing room
    lx = x0 + 18
    cols = legend_cols or (3 if len(series) > 4 else 2)
    col_w = (pw - 28) / cols
    total_rows = (len(series) + cols - 1) // cols
    base_y = pb + 38 if total_rows == 2 else pb + 46
    for idx, s in enumerate(series):
        col = s["color"]
        name = s["label"]
        cx = lx + (idx % cols) * col_w
        row_idx = idx // cols
        cy = base_y + row_idx * 16
        if s.get("dash"):
            parts.append(
                f'<line x1="{cx:.1f}" x2="{cx + 12:.1f}" y1="{cy - 3.5:.1f}" y2="{cy - 3.5:.1f}" '
                f'stroke="{col}" stroke-width="2" stroke-dasharray="3 2"/>'
            )
        else:
            parts.append(
                f'<line x1="{cx:.1f}" x2="{cx + 12:.1f}" y1="{cy - 3.5:.1f}" y2="{cy - 3.5:.1f}" '
                f'stroke="{col}" stroke-width="2.4" stroke-linecap="round"/>'
            )
        parts.append(f'<text x="{cx + 17:.1f}" y="{cy:.1f}" class="leg">{_e(name)}</text>')

    return "\n".join(parts) + "\n"


def render_goodhart() -> None:
    hard = json.loads((RESULTS / "grpo_qwen3b_hardened.json").read_text())
    naive = json.loads((RESULTS / "grpo_qwen3b_naive.json").read_text())

    steps = [s["step"] for s in hard["steps"]]
    e_steps = [e["step"] for e in hard["evals"]]
    h_p1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in hard["evals"]]
    h_lo = [e["report"]["pass_at_k"]["1"]["ci_low"] for e in hard["evals"]]
    h_hi = [e["report"]["pass_at_k"]["1"]["ci_high"] for e in hard["evals"]]
    n_p1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in naive["evals"]]

    def _fam(run: dict[str, Any], name: str) -> list[float]:
        out: list[float] = []
        for e in run["evals"]:
            m = {f["family"]: f["pass_at_1"] for f in e["report"]["by_family"]}
            out.append(float(m.get(name, 0.0)))
        return out

    w, h = 880, 344
    svg = _svg_header(w, h, "Hardened vs Naive Grading on Qwen2.5-Coder-3B")
    svg += f'<line x1="{w / 2:.1f}" x2="{w / 2:.1f}" y1="22" y2="{h - 22}" stroke="{RULE}" stroke-width="1"/>\n'
    svg += _line_plot(
        8,
        8,
        424,
        328,
        title="Train reward vs. held-out pass@1",
        subtitle="Qwen2.5-Coder-3B, 20 GRPO steps, 30 held-out tasks x 4",
        x_max=20,
        y_max=1.0,
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        x_ticks=[0, 5, 10, 15, 20],
        y_fmt="pct",
        band={"xs": e_steps, "lo": h_lo, "hi": h_hi, "color": C_FOREST},
        series=[
            {"label": "Hardened pass@1 (38.3%)", "xs": e_steps, "ys": h_p1, "color": C_FOREST, "width": 2.6},
            {"label": "Naive pass@1 (19.2%)", "xs": e_steps, "ys": n_p1, "color": C_STONE, "width": 2.3},
            {"label": "Naive train (100% peak)", "xs": steps, "ys": [s["mean_reward"] for s in naive["steps"]], "color": C_TERRA, "width": 1.6, "dash": "4 3", "dots": False, "opacity": 0.9},
            {"label": "Hardened train reward", "xs": steps, "ys": [s["mean_reward"] for s in hard["steps"]], "color": C_FOREST, "width": 1.4, "dash": "3 3", "dots": False, "opacity": 0.65},
        ],
    )
    svg += _line_plot(
        448,
        8,
        424,
        328,
        title="Held-out pass@1 by task family",
        subtitle="Hardened fixes multi-file ledger invariants; naive falls to 0%",
        x_max=20,
        y_max=1.0,
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        x_ticks=[0, 5, 10, 15, 20],
        y_fmt="pct",
        series=[
            {"label": "Hardened rate (92.5%)", "xs": e_steps, "ys": _fam(hard, "rate_limiter_window"), "color": C_SLATE, "width": 2.5},
            {"label": "Hardened ledger (22.5%)", "xs": e_steps, "ys": _fam(hard, "ledger_invariants"), "color": C_FOREST, "width": 2.5},
            {"label": "Naive rate (57.5%)", "xs": e_steps, "ys": _fam(naive, "rate_limiter_window"), "color": C_OCHRE, "width": 1.9, "dash": "4 3", "r": 2.6},
            {"label": "Naive ledger (0.0%)", "xs": e_steps, "ys": _fam(naive, "ledger_invariants"), "color": C_TERRA, "width": 1.9, "dash": "4 3", "r": 2.6},
        ],
    )
    svg += "</svg>\n"
    (ASSETS / "ablation_goodhart.svg").write_text(svg, encoding="utf-8")


def render_async_mis() -> None:
    p_van = RESULTS / "grpo_3b_async_vanilla.json"
    p_mis = RESULTS / "grpo_3b_async_mis.json"
    p_van8 = RESULTS / "grpo_3b_lag8_vanilla.json"
    p_mis8 = RESULTS / "grpo_3b_lag8_mis.json"
    if not (p_van.exists() and p_mis.exists()):
        return
    van = json.loads(p_van.read_text())
    mis = json.loads(p_mis.read_text())
    steps = [s["step"] for s in mis["steps"]]
    e_steps = [e["step"] for e in mis["evals"]]
    max_s = max(steps)
    m_p1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in mis["evals"]]
    v_p1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in van["evals"]]
    m_max = [s["tim"]["max_abs_logprob_diff"] for s in mis["steps"]]
    v_max = [s["tim"]["max_abs_logprob_diff"] for s in van["steps"]]

    left_series: list[dict[str, Any]] = [
        {"label": f"lag=3 MIS ({m_p1[-1]*100:.1f}%)", "xs": e_steps, "ys": m_p1, "color": C_FOREST, "width": 2.6},
        {"label": f"lag=3 Vanilla ({v_p1[-1]*100:.1f}%)", "xs": e_steps, "ys": v_p1, "color": C_SLATE, "width": 2.2, "dash": "4 3"},
    ]
    right_series: list[dict[str, Any]] = [
        {"label": "lag=3 MIS", "xs": steps, "ys": m_max, "color": C_FOREST, "width": 2.1},
        {"label": "lag=3 Vanilla", "xs": steps, "ys": v_max, "color": C_SLATE, "width": 1.9, "dash": "4 3"},
    ]
    all_max_vals = list(m_max) + list(v_max)
    if p_van8.exists() and p_mis8.exists():
        van8 = json.loads(p_van8.read_text())
        mis8 = json.loads(p_mis8.read_text())
        m8_p1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in mis8["evals"]]
        v8_p1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in van8["evals"]]
        m8_max = [s["tim"]["max_abs_logprob_diff"] for s in mis8["steps"]]
        v8_max = [s["tim"]["max_abs_logprob_diff"] for s in van8["steps"]]
        e8_steps = [e["step"] for e in mis8["evals"]]
        s8_steps = [s["step"] for s in mis8["steps"]]
        left_series.extend(
            [
                {"label": f"lag=8 MIS ({m8_p1[-1]*100:.1f}%)", "xs": e8_steps, "ys": m8_p1, "color": C_OCHRE, "width": 2.3},
                {"label": f"lag=8 Vanilla ({v8_p1[-1]*100:.1f}%)", "xs": e8_steps, "ys": v8_p1, "color": C_TERRA, "width": 2.1, "dash": "4 3"},
            ]
        )
        right_series.extend(
            [
                {"label": "lag=8 MIS", "xs": s8_steps, "ys": m8_max, "color": C_OCHRE, "width": 2.1},
                {"label": "lag=8 Vanilla", "xs": s8_steps, "ys": v8_max, "color": C_TERRA, "width": 2.0, "dash": "4 3"},
            ]
        )
        all_max_vals.extend(m8_max + v8_max)

    peak_drift = max(all_max_vals)
    y_top = float(int(peak_drift) + 1)
    y_int_ticks = [float(i) for i in range(int(y_top) + 1)]

    w, h = 880, 344
    svg = _svg_header(w, h, "Async Off-Policy Staleness (lag=3 vs lag=8): Vanilla GRPO vs MIS")
    svg += f'<line x1="{w / 2:.1f}" x2="{w / 2:.1f}" y1="22" y2="{h - 22}" stroke="{RULE}" stroke-width="1"/>\n'
    svg += _line_plot(
        8,
        8,
        424,
        328,
        title="Async staleness (lag=3 vs. lag=8): pass@1",
        subtitle="Qwen2.5-Coder-3B, 5 families; lag=8 MIS 3x faster at step 5 (18% vs 6%)",
        x_max=max_s,
        y_max=0.5,
        y_ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        x_ticks=[0, 5, 10, 15, max_s],
        y_fmt="pct",
        series=left_series,
    )
    svg += _line_plot(
        448,
        8,
        424,
        328,
        title="Max off-policy logprob drift",
        subtitle=f"lag=8, kl=0 reaches max |delta log p| = {peak_drift:.2f} (106x ratio)",
        x_min=1,
        x_max=max_s,
        y_max=y_top,
        y_ticks=y_int_ticks,
        x_ticks=[1, 5, 10, 15, max_s],
        y_fmt="int",
        series=right_series,
    )
    svg += "</svg>\n"
    (ASSETS / "ablation_async_mis.svg").write_text(svg, encoding="utf-8")


def render_sparse_curriculum() -> None:
    p_bin = RESULTS / "grpo_7b_5fam_hardened.json"
    p_sub = RESULTS / "grpo_7b_5fam_subtest.json"
    p_cur = RESULTS / "grpo_7b_hard_curriculum_g16.json"
    if not (p_bin.exists() and p_sub.exists() and p_cur.exists()):
        return
    r_bin = json.loads(p_bin.read_text())
    r_sub = json.loads(p_sub.read_text())
    r_cur = json.loads(p_cur.read_text())

    def _fam(run: dict[str, Any], name: str) -> tuple[list[int], list[float]]:
        xs = [e["step"] for e in run["evals"]]
        ys = []
        for e in run["evals"]:
            m = {f["family"]: f["pass_at_1"] for f in e["report"]["by_family"]}
            ys.append(float(m.get(name, 0.0)))
        return xs, ys

    bx, bk = _fam(r_bin, "keyset_paging")
    sx, sk = _fam(r_sub, "keyset_paging")
    cx, ck = _fam(r_cur, "keyset_paging")

    bp1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in r_bin["evals"]]
    sp1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in r_sub["evals"]]
    cp1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in r_cur["evals"]]
    cp4 = [e["report"]["pass_at_k"]["4"]["mean"] for e in r_cur["evals"]]

    w, h = 880, 344
    svg = _svg_header(w, h, "Binary vs Sub-Test Rewards on Qwen2.5-Coder-7B")
    svg += f'<line x1="{w / 2:.1f}" x2="{w / 2:.1f}" y1="22" y2="{h - 22}" stroke="{RULE}" stroke-width="1"/>\n'
    svg += _line_plot(
        8,
        8,
        424,
        328,
        title="keyset_paging held-out pass@1 (Qwen-7B)",
        subtitle="Sub-test shaping stays at 0%; binary G=16 curriculum reaches 70%",
        x_max=20,
        y_max=1.0,
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        x_ticks=[0, 5, 10, 15, 20],
        y_fmt="pct",
        series=[
            {"label": f"Binary G=16 curriculum ({_fmt_pct(ck[-1])})", "xs": cx, "ys": ck, "color": C_FOREST, "width": 2.8},
            {"label": f"Binary G=8 all-5 ({_fmt_pct(bk[-1])})", "xs": bx, "ys": bk, "color": C_SLATE, "width": 2.2},
            {"label": f"Sub-test G=8 ({_fmt_pct(sk[-1])})", "xs": sx, "ys": sk, "color": C_TERRA, "width": 2.2, "dash": "4 3"},
        ],
    )
    svg += _line_plot(
        448,
        8,
        424,
        328,
        title="All-5 held-out accuracy (50 tasks x 4)",
        subtitle="Sub-test shaping trails binary by 9.5 pp; G=16 reaches 68.0% pass@4",
        x_max=20,
        y_max=1.0,
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        x_ticks=[0, 5, 10, 15, 20],
        y_fmt="pct",
        series=[
            {"label": f"G=16 pass@4 ({cp4[-1]*100:.1f}%)", "xs": cx, "ys": cp4, "color": C_FOREST, "width": 2.2, "dash": "3 3"},
            {"label": f"Binary G=8 pass@1 ({bp1[-1]*100:.1f}%)", "xs": bx, "ys": bp1, "color": C_SLATE, "width": 2.6},
            {"label": f"G=16 pass@1 ({cp1[-1]*100:.1f}%)", "xs": cx, "ys": cp1, "color": C_FOREST, "width": 2.6},
            {"label": f"Sub-test G=8 pass@1 ({sp1[-1]*100:.1f}%)", "xs": sx, "ys": sp1, "color": C_TERRA, "width": 2.2, "dash": "4 3"},
        ],
    )
    svg += "</svg>\n"
    (ASSETS / "ablation_sparse_curriculum.svg").write_text(svg, encoding="utf-8")


def render_7b_headline() -> None:
    p = RESULTS / "grpo_7b_5fam_hardened.json"
    if not p.exists():
        return
    run = json.loads(p.read_text())
    e_steps = [e["step"] for e in run["evals"]]
    max_s = max(e_steps)
    p1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in run["evals"]]
    lo = [e["report"]["pass_at_k"]["1"]["ci_low"] for e in run["evals"]]
    hi = [e["report"]["pass_at_k"]["1"]["ci_high"] for e in run["evals"]]
    p4 = [e["report"]["pass_at_k"]["4"]["mean"] for e in run["evals"]]

    fams = ["rate_limiter_window", "async_keyed_queue", "spec_invoice_calc", "ledger_invariants", "keyset_paging"]
    colors = [C_SLATE, C_FOREST, C_STONE, C_OCHRE, C_PLUM]
    fam_series = []
    for fam, col in zip(fams, colors, strict=True):
        ys = []
        for e in run["evals"]:
            m = {f["family"]: f["pass_at_1"] for f in e["report"]["by_family"]}
            ys.append(float(m.get(fam, 0.0)))
        short = fam.split("_")[0]
        fam_series.append({"label": f"{short} ({_fmt_pct(ys[-1])})", "xs": e_steps, "ys": ys, "color": col, "width": 2.3})

    w, h = 880, 344
    svg = _svg_header(w, h, "Qwen2.5-Coder-7B 5-Family RL Results")
    svg += f'<line x1="{w / 2:.1f}" x2="{w / 2:.1f}" y1="22" y2="{h - 22}" stroke="{RULE}" stroke-width="1"/>\n'
    svg += _line_plot(
        8,
        8,
        424,
        328,
        title="Qwen2.5-Coder-7B held-out pass@k",
        subtitle="5 families, 50 held-out tasks x 4 samples, fused Triton kernel",
        x_max=max_s,
        y_max=1.0,
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        x_ticks=e_steps,
        y_fmt="pct",
        band={"xs": e_steps, "lo": lo, "hi": hi, "color": C_FOREST},
        series=[
            {"label": f"pass@4 ({p4[0]*100:.1f}% to {p4[-1]*100:.1f}%)", "xs": e_steps, "ys": p4, "color": C_SLATE, "width": 2.5},
            {"label": f"pass@1 ({p1[0]*100:.1f}% to {p1[-1]*100:.1f}%)", "xs": e_steps, "ys": p1, "color": C_FOREST, "width": 2.6},
        ],
    )
    svg += _line_plot(
        448,
        8,
        424,
        328,
        title="Held-out pass@1 across all 5 families",
        subtitle="async_keyed_queue 7.5% to 57.5%; ledger 70.0% to 97.5%",
        x_max=max_s,
        y_max=1.0,
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        x_ticks=e_steps,
        y_fmt="pct",
        series=fam_series,
        legend_cols=3,
    )
    svg += "</svg>\n"
    (ASSETS / "headline_7b_5fam.svg").write_text(svg, encoding="utf-8")


def render_systems_bench() -> None:
    p_sb = RESULTS / "sandbox_bench.json"
    p_tr = RESULTS / "triton_h100_bench.json"
    if not (p_sb.exists() and p_tr.exists()):
        return
    sb = json.loads(p_sb.read_text())
    tr = json.loads(p_tr.read_text())

    c1 = {r["backend"]: r for r in sb if r["concurrency"] == 1}
    c16 = {r["backend"]: r for r in sb if r["concurrency"] == 16}
    c8 = {r["backend"]: r for r in sb if r["concurrency"] == 8}

    w, h = 880, 344
    svg = _svg_header(w, h, "Triton Logprob Kernel and tri Run Benchmarks")
    svg += f'<line x1="{w / 2:.1f}" x2="{w / 2:.1f}" y1="22" y2="{h - 22}" stroke="{RULE}" stroke-width="1"/>\n'

    # Left panel: Throughput bars (cleanly compares tri v1, v2, Seatbelt, and Modal)
    svg += '<text x="26" y="34" class="title">tri run throughput and c=1 latency</text>\n'
    svg += '<text x="26" y="51" class="sub">Local Seatbelt tri-server vs. tri v1/v2 and remote cloud Modal</text>\n'
    rows = [
        ("local_subprocess (unsandboxed)", c1["local_subprocess"], c16["local_subprocess"], "c=16", C_STONE),
        ("tri-server + Seatbelt (WAL, batch)", c1["shapens_tri_native_seatbelt"], c16["shapens_tri_native_seatbelt"], "c=16", C_FOREST),
        ("tri serve v2 (WAL, batch)", c1["shapens_tri_http_v2_batch"], c16["shapens_tri_http_v2_batch"], "c=16", C_SLATE),
        ("tri serve v1 (12-req legacy)", c1["shapens_tri_http_v1"], c16["shapens_tri_http_v1"], "c=16", C_PLUM),
        ("Modal Sandbox (remote cloud gVisor)", c1["modal_sandbox_ephemeral"], c8["modal_sandbox_ephemeral"], "c=8", C_TERRA),
    ]
    max_tp = max(r[2]["throughput_per_sec"] for r in rows)
    bar_w = 392.0
    for idx, (label, r1, rpk, tag, col) in enumerate(rows):
        y = 84 + idx * 49
        tp = rpk["throughput_per_sec"]
        bw = max(6.0, (tp / max_tp) * bar_w)
        svg += f'<text x="26" y="{y}" class="lbl">{_e(label)}</text>\n'
        svg += f'<text x="418" y="{y}" text-anchor="end" class="ax" style="fill:{col if col == C_FOREST else INK};font-weight:600">{tp:.1f}/s ({tag})</text>\n'
        svg += f'<rect x="26" y="{y + 7}" width="{bar_w:.1f}" height="8" rx="2" fill="{PAPER_DEEP}"/>\n'
        svg += f'<rect x="26" y="{y + 7}" width="{bw:.1f}" height="8" rx="2" fill="{col}"/>\n'
        svg += (
            f'<text x="26" y="{y + 28}" class="ax">'
            f'c=1 p50 {r1["p50_ms"]:.1f} ms, p95 {r1["p95_ms"]:.0f} ms</text>\n'
        )

    # Right panel: Triton H100 kernel vs PyTorch
    svg += '<text x="466" y="34" class="title">Triton fused logprob kernel (H100)</text>\n'
    svg += f'<text x="466" y="51" class="sub">Qwen 152k vocab [2, 2048, 152064] bf16, fwd err {tr["fwd_max_abs_err"]:.1e}</text>\n'

    bar_x = 466.0
    bar_r_w = 276.0
    pt_ms, tr_ms = tr["pytorch_ms"], tr["triton_ms"]
    svg += f'<text x="466" y="94" class="lbl">Forward + backward step time</text>\n'
    svg += f'<text x="854" y="94" text-anchor="end" class="ax" style="fill:{C_FOREST};font-weight:600">{tr["speedup_x"]}x faster</text>\n'
    svg += f'<rect x="{bar_x:.0f}" y="106" width="{bar_r_w:.0f}" height="14" rx="2" fill="{PAPER_DEEP}"/>\n'
    svg += f'<rect x="{bar_x:.0f}" y="106" width="{bar_r_w:.0f}" height="14" rx="2" fill="{C_STONE}" fill-opacity="0.55"/>\n'
    svg += f'<text x="854" y="117" text-anchor="end" class="ax">PyTorch {pt_ms:.2f} ms</text>\n'
    tw = (tr_ms / pt_ms) * bar_r_w
    svg += f'<rect x="{bar_x:.0f}" y="128" width="{bar_r_w:.0f}" height="14" rx="2" fill="{PAPER_DEEP}"/>\n'
    svg += f'<rect x="{bar_x:.0f}" y="128" width="{tw:.1f}" height="14" rx="2" fill="{C_FOREST}"/>\n'
    svg += f'<text x="854" y="139" text-anchor="end" class="ax" style="fill:{C_FOREST};font-weight:600">Triton {tr_ms:.2f} ms</text>\n'

    pt_gb, tr_gb = tr["pytorch_peak_gb"], tr["triton_peak_gb"]
    svg += f'<text x="466" y="190" class="lbl">Peak HBM allocated</text>\n'
    svg += f'<text x="854" y="190" text-anchor="end" class="ax" style="fill:{C_FOREST};font-weight:600">-{tr["vram_saved_gb"]:.2f} GB</text>\n'
    svg += f'<rect x="{bar_x:.0f}" y="202" width="{bar_r_w:.0f}" height="14" rx="2" fill="{PAPER_DEEP}"/>\n'
    svg += f'<rect x="{bar_x:.0f}" y="202" width="{bar_r_w:.0f}" height="14" rx="2" fill="{C_STONE}" fill-opacity="0.55"/>\n'
    svg += f'<text x="854" y="213" text-anchor="end" class="ax">PyTorch {pt_gb:.2f} GB</text>\n'
    gw = (tr_gb / pt_gb) * bar_r_w
    svg += f'<rect x="{bar_x:.0f}" y="224" width="{bar_r_w:.0f}" height="14" rx="2" fill="{PAPER_DEEP}"/>\n'
    svg += f'<rect x="{bar_x:.0f}" y="224" width="{gw:.1f}" height="14" rx="2" fill="{INK}"/>\n'
    svg += f'<text x="854" y="235" text-anchor="end" class="ax" style="fill:{INK};font-weight:600">Triton {tr_gb:.2f} GB</text>\n'

    svg += f'<line x1="466" x2="854" y1="268" y2="268" stroke="{RULE}" stroke-width="1"/>\n'
    svg += f'<text x="466" y="290" class="ax">bwd max abs err {tr["bwd_max_abs_err"]:.2e}, used for policy and k3 KL ref</text>\n'
    svg += '<text x="466" y="308" class="ax">eliminated 4.64 GiB fp32 softmax OOM on 7B long rollouts</text>\n'

    svg += "</svg>\n"
    (ASSETS / "systems_bench.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    render_goodhart()
    render_async_mis()
    render_7b_headline()
    render_sparse_curriculum()
    render_systems_bench()
    for p in sorted(ASSETS.glob("*.svg")):
        ET.parse(p)
    print("rendered and XML-validated all charts in", ASSETS)


if __name__ == "__main__":
    main()
