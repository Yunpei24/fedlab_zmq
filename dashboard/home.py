"""Clean landing page for the FedLab ZMQ research dashboard."""

from __future__ import annotations

import html
import pathlib
import re
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st


_TRACKS = (
    (
        "R1",
        "Fairness baselines",
        "No attack",
        "Client accuracy, variance, Worst-20 and tail gap",
        "exp1_",
    ),
    (
        "R2",
        "Fairness under Byzantine attacks",
        "Robustness",
        "ALIE, Min-Max, Min-Sum, sign flip and IPM",
        "exp2_",
    ),
    (
        "R3",
        "Privacy–fairness trade-off",
        "Differential privacy",
        "Privacy accountant, utility and client-level fairness",
        "exp3_",
    ),
)


def _load_styles() -> None:
    css_path = pathlib.Path(__file__).with_name("home.css")
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _track_counts(result_dirs: list[pathlib.Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for run_dir in result_dirs:
        path = run_dir.as_posix()
        for track, _, _, _, prefix in _TRACKS:
            if f"/{prefix}" in path or path.startswith(prefix):
                counts[track] += 1
    return counts


def _latest_update(result_dirs: list[pathlib.Path]) -> str:
    if not result_dirs:
        return "No result yet"
    latest = max((run_dir / "metrics.json").stat().st_mtime for run_dir in result_dirs)
    return datetime.fromtimestamp(latest).strftime("%d %b %Y · %H:%M")


def _experiment_families(result_dirs: list[pathlib.Path]) -> int:
    families = set()
    for run_dir in result_dirs:
        for part in run_dir.parts:
            if re.match(r"^exp\d+_", part):
                families.add(part)
    return len(families)


def _metric_card(value: str, label: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="home-metric">
          <div class="home-metric-value">{html.escape(value)}</div>
          <div class="home-metric-label">{html.escape(label)}</div>
          <div class="home-metric-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _focus_card(index: str, title: str, text: str, tags: tuple[str, ...]) -> None:
    tag_markup = "".join(f'<span class="home-tag">{html.escape(tag)}</span>' for tag in tags)
    st.markdown(
        f"""
        <div class="home-focus-card">
          <div class="home-focus-index">{html.escape(index)}</div>
          <div class="home-focus-title">{html.escape(title)}</div>
          <div class="home-focus-text">{html.escape(text)}</div>
          <div class="home-tags">{tag_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _step(number: str, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="home-step">
          <div class="home-step-number">{html.escape(number)}</div>
          <div>
            <div class="home-step-title">{html.escape(title)}</div>
            <div class="home-step-text">{html.escape(text)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home(result_dirs: list[pathlib.Path], results_dir: pathlib.Path) -> None:
    """Render the dashboard landing page using only live repository state."""

    _load_styles()
    run_count = len(result_dirs)
    track_counts = _track_counts(result_dirs)
    latest_update = _latest_update(result_dirs)
    family_count = _experiment_families(result_dirs)
    status_class = "ready" if run_count else "empty"
    status_text = f"{run_count} runs indexed" if run_count else "Waiting for first run"

    st.markdown(
        f"""
        <section class="home-hero">
          <div class="home-eyebrow">UM6P · FEDERATED LEARNING RESEARCH</div>
          <div class="home-hero-grid">
            <div>
              <h1>FedLab <span>ZMQ</span></h1>
              <p class="home-lead">
                A research console for studying fairness, Byzantine robustness,
                differential privacy and energy efficiency in federated learning.
              </p>
              <div class="home-pill-row">
                <span>Client-level fairness</span>
                <span>Byzantine robustness</span>
                <span>Differential privacy</span>
                <span>Energy efficiency</span>
              </div>
            </div>
            <div class="home-status-panel">
              <div class="home-status-line">
                <span class="home-status-dot {status_class}"></span>
                <span>{html.escape(status_text)}</span>
              </div>
              <div class="home-status-label">Results source</div>
              <div class="home-status-path">{html.escape(str(results_dir))}</div>
              <div class="home-status-label">Latest update</div>
              <div class="home-status-value">{html.escape(latest_update)}</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    with metric_columns[0]:
        _metric_card(str(run_count), "Completed run artifacts", "metrics.json files indexed")
    with metric_columns[1]:
        _metric_card(str(family_count), "Experiment families", "Distinct R-series protocols")
    with metric_columns[2]:
        _metric_card(str(sum(track_counts.values())), "R1–R3 runs", "Comparable protocol lanes")
    with metric_columns[3]:
        _metric_card("Enabled", "Multi-seed reporting", "Mean ± sample standard deviation")

    st.markdown(
        """
        <div class="home-section-heading">
          <div class="home-kicker">CURRENT RESEARCH FOCUS</div>
          <h2>FairPartFAR-DP</h2>
          <p>
            Preserve FAR's inclusion of honest atypical clients while controlling
            user-level influence, privacy loss and training cost.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="home-pipeline" role="img" aria-label="FairPartFAR-DP conceptual pipeline">
          <span>User-level clipping</span><b>→</b>
          <span>Robust reference F</span><b>→</b>
          <span>Controlled tilting</span><b>→</b>
          <span>Partial training</span><b>→</b>
          <span>Gaussian DP</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    focus_columns = st.columns(4)
    focus_content = (
        (
            "01",
            "Fairness",
            "Measure whether global progress is shared across clients, not only whether average accuracy rises.",
            ("Worst-20", "Variance", "Tail gap"),
        ),
        (
            "02",
            "Robustness",
            "Evaluate robust references and FAR reweighting against honest heterogeneity and stealthy Byzantine updates.",
            ("CM(NNM)", "RFA", "ALIE · IPM"),
        ),
        (
            "03",
            "Privacy",
            "Track explicit adjacency, clipping sensitivity, Gaussian noise and multi-round privacy accounting.",
            ("User-level", "RDP", "ε, δ"),
        ),
        (
            "04",
            "Efficiency",
            "Validate partial training with end-to-end metrics instead of reporting communication savings in isolation.",
            ("Joules-to-Accuracy", "Bytes", "Toubkal"),
        ),
    )
    for column, content in zip(focus_columns, focus_content):
        with column:
            _focus_card(*content)

    st.markdown(
        """
        <div class="home-section-heading compact">
          <div class="home-kicker">EXPERIMENTAL PROGRAM</div>
          <h2>Comparable evidence, one question at a time</h2>
          <p>Each track isolates a distinct scientific claim before mechanisms are combined.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    track_rows = []
    for track, objective, condition, evidence, _ in _TRACKS:
        count = track_counts.get(track, 0)
        track_rows.append(
            {
                "Track": track,
                "Research question": objective,
                "Condition": condition,
                "Primary evidence": evidence,
                "Indexed runs": count,
                "Status": "Available" if count else "Not indexed",
            }
        )
    st.dataframe(pd.DataFrame(track_rows), width="stretch", hide_index=True)

    st.markdown(
        """
        <div class="home-section-heading compact">
          <div class="home-kicker">WORKFLOW</div>
          <h2>From run artifacts to defensible comparisons</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    step_columns = st.columns(4)
    steps = (
        ("1", "Choose a protocol", "Keep dataset partition, seeds, model and threat assumptions explicit."),
        ("2", "Run the matrix", "Execute comparable methods under the same experimental condition."),
        ("3", "Inspect results", "Use Results for curves, privacy, robustness and client-level fairness."),
        ("4", "Aggregate seeds", "Use Summary Table for mean ± standard deviation and CSV export."),
    )
    for column, step in zip(step_columns, steps):
        with column:
            _step(*step)

    if not result_dirs:
        st.info(
            "No experiment artifact is indexed yet. Set the results folder in the sidebar "
            "or complete a run that writes metrics.json."
        )

