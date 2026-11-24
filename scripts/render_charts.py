#!/usr/bin/env python3
"""Generate clean, GitHub dark/light-compatible SVG charts from saved loop run JSONs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "docs" / "assets"

BG = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
FG = "#e6edf3"
MUTED = "#8b949e"
C_GREEN = "#3fb950"
C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_AMBER = "#d29922"
C_RED = "#f85149"
C_CYAN = "#39c5cf"


def _svg_header(w: int, h: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="{title}">\n'
        f'<style>'
        f"text {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}"
        f".title {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
        f"font-size: 14px; font-weight: 600; fill: {FG}; }}"
        f".sub {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
        f"font-size: 11px; fill: {MUTED}; }}"
        f".ax {{ font-size: 10px; fill: {MUTED}; }}"
        f".lbl {{ font-size: 11px; fill: {FG}; }}"
        f".grid {{ stroke: {BORDER}; stroke-width: 1; stroke-dasharray: 2 3; }}"
        f"</style>\n"
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" '
        f'fill="{BG}" stroke="{BORDER}"/>\n'
    )


def _line_plot(
    x0: float,
    y0: float,
    pw: float,
    ph: float,
    *,
    title: str,
    subtitle: str,
    x_max: float,
    y_max: float,
    x_ticks: list[int],
    y_fmt: str,
    series: list[dict[str, Any]],
    band: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="8" '
        f'fill="{CARD}" stroke="{BORDER}"/>'
    )
    parts.append(f'<text x="{x0 + 16}" y="{y0 + 24}" class="title">{title}</text>')
    parts.append(f'<text x="{x0 + 16}" y="{y0 + 40}" class="sub">{subtitle}</text>')

    pl, pr, pt, pb = x0 + 48, x0 + pw - 18, y0 + 56, y0 + ph - 52
    iw, ih = pr - pl, pb - pt

    def sx(v: float) -> float:
        return pl + (v / max(x_max, 1e-9)) * iw

    def sy(v: float) -> float:
        return pb - (min(max(v, 0.0), y_max) / max(y_max, 1e-9)) * ih

    for i in range(5):
        v = (y_max * i) / 4.0
        yy = sy(v)
        parts.append(f'<line x1="{pl:.1f}" x2="{pr:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        lbl = f"{round(v * 100)}%" if y_fmt == "pct" else f"{v:.2f}"
        parts.append(f'<text x="{pl - 8:.1f}" y="{yy + 3:.1f}" text-anchor="end" class="ax">{lbl}</text>')

    for xt in x_ticks:
        xx = sx(xt)
        parts.append(f'<text x="{xx:.1f}" y="{pb + 15:.1f}" text-anchor="middle" class="ax">s{xt}</text>')

    if band:
        xs, lo, hi, col = band["xs"], band["lo"], band["hi"], band["color"]
        top = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, hi, strict=True))
        bot = " ".join(
            f"{sx(x):.1f},{sy(y):.1f}"
            for x, y in reversed(list(zip(xs, lo, strict=True)))
        )
        parts.append(f'<polygon points="{top} {bot}" fill="{col}" fill-opacity="0.16"/>')

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
            f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{s.get("width", 2.2)}"'
            f'{dash}{op} stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if s.get("dots", True):
            for x, y in zip(xs, ys, strict=True):
                parts.append(
                    f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{s.get("r", 3.2)}" fill="{col}"/>'
                )

    # Legend row(s) — supports up to 5 items cleanly across 2 rows
    lx, ly = x0 + 16, y0 + ph - 14
    cols = 3 if len(series) > 4 else 2
    col_w = (pw - 24) / cols
    for idx, s in enumerate(series):
        col = s["color"]
        name = s["label"]
        cx = lx + (idx % cols) * col_w
        row_idx = idx // cols
        total_rows = (len(series) + cols - 1) // cols
        cy = ly - (total_rows - 1 - row_idx) * 14
        if s.get("dash"):
            parts.append(
                f'<line x1="{cx:.1f}" x2="{cx + 10:.1f}" y1="{cy - 3:.1f}" y2="{cy - 3:.1f}" '
                f'stroke="{col}" stroke-width="2" stroke-dasharray="3 2"/>'
            )
        else:
            parts.append(f'<circle cx="{cx + 4:.1f}" cy="{cy - 3:.1f}" r="3.5" fill="{col}"/>')
        parts.append(f'<text x="{cx + 14:.1f}" y="{cy:.1f}" class="ax">{name}</text>')

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

    svg = _svg_header(860, 320, "Hardened vs Naive RLVR Goodhart Divergence")
    svg += _line_plot(
        14,
        14,
        410,
        292,
        title="Goodhart Divergence: Train Reward vs True Pass@1",
        subtitle="Qwen2.5-Coder-3B · 20 GRPO steps · 1×H100 · Held-out 30×4",
        x_max=20,
        y_max=1.0,
        x_ticks=[0, 5, 10, 15, 20],
        y_fmt="pct",
        band={"xs": e_steps, "lo": h_lo, "hi": h_hi, "color": C_GREEN},
        series=[
            {"label": "Hardened pass@1 (38.3%)", "xs": e_steps, "ys": h_p1, "color": C_GREEN, "width": 2.6},
            {"label": "Naive pass@1 (19.2%)", "xs": e_steps, "ys": n_p1, "color": C_PURPLE, "width": 2.4},
            {"label": "Naive train rew (100%)", "xs": steps, "ys": [s["mean_reward"] for s in naive["steps"]], "color": C_RED, "width": 1.4, "dash": "3 3", "dots": False, "opacity": 0.8},
            {"label": "Hardened train rew", "xs": steps, "ys": [s["mean_reward"] for s in hard["steps"]], "color": C_GREEN, "width": 1.4, "dash": "3 3", "dots": False, "opacity": 0.75},
        ],
    )
    svg += _line_plot(
        436,
        14,
        410,
        292,
        title="Per-Family Held-Out Pass@1 Breakdown",
        subtitle="Hardened learns multi-file ledger fixes; Naive collapses to 0%",
        x_max=20,
        y_max=1.0,
        x_ticks=[0, 5, 10, 15, 20],
        y_fmt="pct",
        series=[
            {"label": "Hardened rate_limiter (92.5%)", "xs": e_steps, "ys": _fam(hard, "rate_limiter_window"), "color": C_BLUE, "width": 2.5},
            {"label": "Hardened ledger (22.5%)", "xs": e_steps, "ys": _fam(hard, "ledger_invariants"), "color": C_GREEN, "width": 2.5},
            {"label": "Naive rate_limiter (57.5%)", "xs": e_steps, "ys": _fam(naive, "rate_limiter_window"), "color": C_AMBER, "width": 1.8, "dash": "4 3", "r": 2.5},
            {"label": "Naive ledger (0.0%)", "xs": e_steps, "ys": _fam(naive, "ledger_invariants"), "color": C_RED, "width": 1.8, "dash": "4 3", "r": 2.5},
        ],
    )
    svg += "</svg>\n"
    (ASSETS / "ablation_goodhart.svg").write_text(svg, encoding="utf-8")


def render_async_mis() -> None:
    p_van = RESULTS / "grpo_3b_async_vanilla.json"
    p_mis = RESULTS / "grpo_3b_async_mis.json"
    if not (p_van.exists() and p_mis.exists()):
        return
    van = json.loads(p_van.read_text())
    mis = json.loads(p_mis.read_text())
    steps = [s["step"] for s in mis["steps"]]
    e_steps = [e["step"] for e in mis["evals"]]
    max_s = max(steps)
    m_p1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in mis["evals"]]
    v_p1 = [e["report"]["pass_at_k"]["1"]["mean"] for e in van["evals"]]
    m_hi = [s["tim"]["high_ratio_fraction"] for s in mis["steps"]]
    v_hi = [s["tim"]["high_ratio_fraction"] for s in van["steps"]]
    m_mask = [s["tim"]["masked_token_fraction"] for s in mis["steps"]]
    y_hi = max(0.05, max(m_hi + v_hi + m_mask) * 1.25)

    svg = _svg_header(860, 320, "Async Off-Policy Staleness: Vanilla GRPO vs MIS")
    svg += _line_plot(
        14,
        14,
        410,
        292,
        title="Async Off-Policy (lag=3 steps): Held-Out Pass@1",
        subtitle="Qwen2.5-Coder-3B · 5 families · lr=8e-5 · Held-out 50×4",
        x_max=max_s,
        y_max=max(0.5, max(m_p1 + v_p1) * 1.2),
        x_ticks=[0, 5, 10, 15, max_s],
        y_fmt="pct",
        series=[
            {"label": f"MIS (r<=1.5): {m_p1[-1]*100:.1f}%", "xs": e_steps, "ys": m_p1, "color": C_GREEN, "width": 2.6},
            {"label": f"Vanilla GRPO: {v_p1[-1]*100:.1f}%", "xs": e_steps, "ys": v_p1, "color": C_RED, "width": 2.4, "dash": "4 3"},
        ],
    )
    svg += _line_plot(
        436,
        14,
        410,
        292,
        title="Off-Policy Ratio Tail & MIS Token Masking",
        subtitle="Fraction of assistant tokens with importance ratio spikes",
        x_max=max_s,
        y_max=y_hi,
        x_ticks=[1, 5, 10, 15, max_s],
        y_fmt="pct",
        series=[
            {"label": "MIS masked tokens (r>1.5)", "xs": steps, "ys": m_mask, "color": C_CYAN, "width": 2.2},
            {"label": "Vanilla high-ratio (r>2.0)", "xs": steps, "ys": v_hi, "color": C_RED, "width": 2.2, "dash": "3 2"},
            {"label": "MIS high-ratio (r>2.0)", "xs": steps, "ys": m_hi, "color": C_AMBER, "width": 1.8},
        ],
    )
    svg += "</svg>\n"
    (ASSETS / "ablation_async_mis.svg").write_text(svg, encoding="utf-8")


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
    colors = [C_BLUE, C_GREEN, C_CYAN, C_AMBER, C_PURPLE]
    fam_series = []
    for fam, col in zip(fams, colors, strict=True):
        ys = []
        for e in run["evals"]:
            m = {f["family"]: f["pass_at_1"] for f in e["report"]["by_family"]}
            ys.append(float(m.get(fam, 0.0)))
        short = fam.split("_")[0]
        fam_series.append({"label": f"{short} ({ys[-1]*100:.0f}%)", "xs": e_steps, "ys": ys, "color": col, "width": 2.2})

    svg = _svg_header(860, 330, "Qwen2.5-Coder-7B 5-Family RLVR Results")
    svg += _line_plot(
        14,
        14,
        410,
        302,
        title="Qwen2.5-Coder-7B · Overall Held-Out Pass@k",
        subtitle="5 families · 50 held-out tasks × 4 samples · Fused Triton kernel",
        x_max=max_s,
        y_max=1.0,
        x_ticks=e_steps,
        y_fmt="pct",
        band={"xs": e_steps, "lo": lo, "hi": hi, "color": C_GREEN},
        series=[
            {"label": f"pass@4 ({p4[0]*100:.1f}% -> {p4[-1]*100:.1f}%)", "xs": e_steps, "ys": p4, "color": C_BLUE, "width": 2.5},
            {"label": f"pass@1 ({p1[0]*100:.1f}% -> {p1[-1]*100:.1f}%)", "xs": e_steps, "ys": p1, "color": C_GREEN, "width": 2.6},
        ],
    )
    svg += _line_plot(
        436,
        14,
        410,
        302,
        title="7B Held-Out Pass@1 Across All 5 Coding Families",
        subtitle="async_keyed_queue jumps 7.5% -> 57.5%; ledger 70% -> 97.5%",
        x_max=max_s,
        y_max=1.0,
        x_ticks=e_steps,
        y_fmt="pct",
        series=fam_series,
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

    # Pick conc=1 p50 & p95 for the 3 sandbox backends
    c1 = {r["backend"]: r for r in sb if r["concurrency"] == 1}
    c8 = {r["backend"]: r for r in sb if r["concurrency"] == 8}

    svg = _svg_header(860, 290, "Systems Benchmarks: Sandbox Tail Latency & Custom Triton Kernel")
    # Left card: Sandbox latency bars (log-friendly visual)
    svg += f'<rect x="14" y="14" width="410" height="262" rx="8" fill="{CARD}" stroke="{BORDER}"/>\n'
    svg += f'<text x="30" y="38" class="title">Sandbox Grader Latency &amp; Throughput</text>\n'
    svg += f'<text x="30" y="54" class="sub">p50 / p95 (conc=1) &amp; conc=8 throughput · all_correct=True</text>\n'
    rows = [
        ("local_subprocess (no isol.)", c1["local_subprocess"], c8["local_subprocess"], C_BLUE),
        ("shapens/tri HTTP (CoW)", c1["shapens_tri_http"], c8["shapens_tri_http"], C_GREEN),
        ("Modal Sandbox (ephemeral)", c1["modal_sandbox_ephemeral"], c8["modal_sandbox_ephemeral"], C_AMBER),
    ]
    max_ms = max(r["p95_ms"] for r in c1.values())
    for idx, (label, r1, r8, col) in enumerate(rows):
        y = 84 + idx * 62
        w50 = max(6.0, (r1["p50_ms"] / max_ms) * 260.0)
        w95 = max(6.0, (r1["p95_ms"] / max_ms) * 260.0)
        svg += f'<text x="30" y="{y}" class="lbl">{label}</text>\n'
        svg += f'<text x="408" y="{y}" text-anchor="end" class="ax">{r8["throughput_per_sec"]:.1f}/s @c8</text>\n'
        svg += f'<rect x="30" y="{y + 8}" width="{w95:.1f}" height="12" rx="3" fill="{col}" fill-opacity="0.28"/>\n'
        svg += f'<rect x="30" y="{y + 8}" width="{w50:.1f}" height="12" rx="3" fill="{col}"/>\n'
        svg += (
            f'<text x="30" y="{y + 34}" class="ax">'
            f'p50={r1["p50_ms"]:.0f}ms · p95={r1["p95_ms"]:.0f}ms</text>\n'
        )

    # Right card: Triton H100 kernel vs PyTorch
    svg += f'<rect x="436" y="14" width="410" height="262" rx="8" fill="{CARD}" stroke="{BORDER}"/>\n'
    svg += f'<text x="452" y="38" class="title">Custom Triton Fused LogProb Kernel (H100)</text>\n'
    svg += f'<text x="452" y="54" class="sub">Qwen 152k vocab [2, 2048, 152064] bf16 · fwd err {tr["fwd_max_abs_err"]:.1e}</text>\n'

    # Latency comparison bars
    pt_ms, tr_ms = tr["pytorch_ms"], tr["triton_ms"]
    svg += f'<text x="452" y="88" class="lbl">Forward + Backward Step Time ({tr["speedup_x"]}× faster)</text>\n'
    svg += f'<rect x="452" y="96" width="280" height="16" rx="4" fill="{MUTED}" fill-opacity="0.35"/>\n'
    svg += f'<text x="740" y="108" class="ax">PyTorch {pt_ms:.2f}ms</text>\n'
    tw = (tr_ms / pt_ms) * 280.0
    svg += f'<rect x="452" y="118" width="{tw:.1f}" height="16" rx="4" fill="{C_GREEN}"/>\n'
    svg += f'<text x="{460 + tw:.1f}" y="130" class="lbl">Triton {tr_ms:.2f}ms</text>\n'

    # Peak VRAM comparison bars
    pt_gb, tr_gb = tr["pytorch_peak_gb"], tr["triton_peak_gb"]
    svg += f'<text x="452" y="168" class="lbl">Peak HBM Allocated (-{tr["vram_saved_gb"]:.2f} GB saved)</text>\n'
    svg += f'<rect x="452" y="176" width="280" height="16" rx="4" fill="{MUTED}" fill-opacity="0.35"/>\n'
    svg += f'<text x="740" y="188" class="ax">PyTorch {pt_gb:.2f} GB</text>\n'
    gw = (tr_gb / pt_gb) * 280.0
    svg += f'<rect x="452" y="198" width="{gw:.1f}" height="16" rx="4" fill="{C_CYAN}"/>\n'
    svg += f'<text x="{460 + gw:.1f}" y="210" class="lbl">Triton {tr_gb:.2f} GB</text>\n'
    svg += f'<text x="452" y="248" class="ax">bwd max abs err: {tr["bwd_max_abs_err"]:.2e} · zero [N,V] fp32 softmax tensor</text>\n'

    svg += "</svg>\n"
    (ASSETS / "systems_bench.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    render_goodhart()
    render_async_mis()
    render_7b_headline()
    render_systems_bench()
    print("rendered charts to", ASSETS)


if __name__ == "__main__":
    main()
