#!/usr/bin/env python3
"""Repair sparse-round plots and reorganize the existing notebook in place.

This is a surgical migration, not a notebook generator. It preserves user
cells, retains unobserved rounds as NaN, and never executes a kernel or run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks/LDP_Gradient_FAR_Positioning_Analysis.ipynb"
LEGACY_SECTIONS = {
    "3f5e5aca": "Bilan de disponibilité",
    "aabb002c5028d687": "Positioning v3",
    "bad28341a7cb6dd0": "G0d-F",
    "e8e6cd5af2aa8831": "G0e",
}

ROUND_COVERAGE_SOURCE = '''def _positioning_round_coverage_note(ax, frame, metric):
    """Mark observations without filling or joining gaps in client evaluation."""
    numeric = pd.to_numeric(frame[metric], errors="coerce")
    observed = sorted(frame.loc[np.isfinite(numeric), "round"].unique())
    expected = sorted(frame["round"].unique())
    if observed and len(observed) < len(expected):
        ax.set_xticks(observed)
        ax.set_xlabel(
            f"Tour — {len(observed)}/{len(expected)} tours évalués ; aucune imputation"
        )


'''


def _replace_once(source: str, old: str, new: str) -> str:
    if new in source:
        return source
    if source.count(old) != 1:
        raise AssertionError(f"Expected exactly one source anchor: {old[:100]!r}")
    return source.replace(old, new, 1)


def _repair_plot_sources(cells: dict) -> None:
    source = cells["180afe8c731ef31f"].source
    if "def _positioning_round_coverage_note(" not in source:
        source = source.replace("def _mean_sd_by_round(", ROUND_COVERAGE_SOURCE + "def _mean_sd_by_round(", 1)
    source = _replace_once(
        source,
        'ax.plot(group["round"], group["mean"], linewidth=2, label=label)',
        'ax.plot(group["round"], group["mean"], linewidth=2, marker="o", markersize=4, label=label)',
    )
    source = _replace_once(
        source,
        '''                    group.loc[mask, "round"],
                    group.loc[mask, "mean"] - group.loc[mask, "sd"],
                    group.loc[mask, "mean"] + group.loc[mask, "sd"],
                    alpha=0.14,''',
        '''                    group["round"],
                    group["mean"] - group["sd"],
                    group["mean"] + group["sd"],
                    where=mask, alpha=0.14,''',
    )
    source = _replace_once(
        source,
        '        ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")',
        '        _positioning_round_coverage_note(ax, view, metric)\n'
        '        ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")',
    )
    cells["180afe8c731ef31f"].source = source

    source = cells["rcig-review-alpha-accuracy"].source
    source = _replace_once(
        source,
        'linestyle="--" if alpha == 0 else "-", linewidth=2)',
        'linestyle="--" if alpha == 0 else "-", linewidth=2, marker="o", markersize=4)',
    )
    source = _replace_once(
        source,
        '''                    ax.fill_between(stats.index[mask],
                                    (stats["mean"]-stats["std"])[mask],
                                    (stats["mean"]+stats["std"])[mask], alpha=.12)''',
        '''                    ax.fill_between(stats.index,
                                    stats["mean"]-stats["std"],
                                    stats["mean"]+stats["std"], where=mask, alpha=.12)''',
    )
    source = _replace_once(
        source,
        '            ax.legend(loc="best", fontsize=10)',
        '            _positioning_round_coverage_note(ax, group[group.alpha.isin(allowed)], metric)\n'
        '            ax.legend(loc="best", fontsize=10)',
    )
    cells["rcig-review-alpha-accuracy"].source = source

    source = cells["rcig-review-controls-table"].source
    source = _replace_once(
        source, 'ax.plot(values.index, values, linewidth=2,',
        'ax.plot(values.index, values, linewidth=2, marker="o", markersize=4,',
    )
    source = _replace_once(
        source, '        ax.legend(loc="best", fontsize=9)',
        '        _positioning_round_coverage_note(ax, group, metric)\n'
        '        ax.legend(loc="best", fontsize=9)',
    )
    cells["rcig-review-controls-table"].source = source

    source = cells["rcig-review-alpha-dp"].source
    source = _replace_once(
        source, 'ax.plot(stats.index, stats, linewidth=2, label=f"α = {alpha:g}")',
        'ax.plot(stats.index, stats, linewidth=2, marker="o", markersize=4, label=f"α = {alpha:g}")',
    )
    cells["rcig-review-alpha-dp"].source = source

    note = (
        "\n\n**Points réellement observés.** Les métriques clientes (dont Worst-20 et gap) "
        "sont enregistrées tous les deux tours et au dernier tour : ici 1, 3, 5, …, "
        "19 et 20. Les marqueurs rendent visibles ces observations isolées. "
        "Les trous restent non mesurés : aucune valeur n'est interpolée ou imputée, "
        "et les lignes/bandes ne franchissent pas ces trous. L'accuracy et la loss "
        "test restent disponibles aux 20 tours."
    )
    if "**Points réellement observés.**" not in cells["rcig-review-alpha-intro"].source:
        cells["rcig-review-alpha-intro"].source += note


def apply_repairs(notebook):
    """Mutate a NotebookNode; return the exact edit/removal audit (idempotent)."""
    original = {cell.id: cell.source for cell in notebook.cells}
    kept, removed, skipping = [], [], False
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and re.match(r"^## \d+\.", cell.source):
            skipping = cell.id in LEGACY_SECTIONS
        if skipping:
            removed.append(cell.id)
        else:
            kept.append(cell)
    notebook.cells = kept
    cells = {cell.id: cell for cell in kept}
    _repair_plot_sources(cells)

    # Move only the motivation block. Its later evaluation block stays last.
    starts = [i for i, cell in enumerate(kept) if cell.cell_type == "markdown"
              and re.match(r"^## (?:2\.1|8\.1) Motivation initiale", cell.source)]
    if len(starts) != 1:
        raise AssertionError(f"Expected one motivation block, found {starts}")
    start = starts[0]
    end = next((i for i in range(start + 1, len(kept))
                if kept[i].cell_type == "markdown" and kept[i].source.startswith("## ")), len(kept))
    motivation = kept[start:end]
    del kept[start:end]
    before_explorer = next(i for i, cell in enumerate(kept)
                           if cell.cell_type == "markdown" and cell.source.startswith("## 9."))
    kept[before_explorer:before_explorer] = motivation

    for cell in kept:
        source = cell.source
        source = re.sub(r"(#{2,3}\s+)2\.1(?=[.\s])", r"\g<1>8.1", source)
        source = re.sub(r"(#{2,3}\s+)15(?=[.\s])", r"\g<1>11", source)
        source = re.sub(r"(?i)(section\s+)2\.1\b", r"\g<1>8.1", source)
        source = re.sub(r"(?i)(section\s+)15\b", r"\g<1>11", source)
        cell.source = source

    # Clear affected outputs only, including calls to the repaired plot helper.
    edited = [cell.id for cell in kept if original.get(cell.id) != cell.source]
    invalidated = []
    helper_changed = "180afe8c731ef31f" in edited or "rcig-review-alpha-accuracy" in edited
    for cell in kept:
        affected_call = helper_changed and any(name in cell.source for name in (
            "_plot_one_trajectory(", "plot_all_alpha_trajectories(",
        ))
        if cell.cell_type == "code" and (cell.id in edited or affected_call):
            cell.outputs = []
            cell.execution_count = None
            invalidated.append(cell.id)
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
    ids = [cell.id for cell in kept]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate notebook cell IDs")
    nbformat.validate(notebook)
    return {"edited_cell_ids": edited, "removed_cell_ids": removed,
            "cleared_output_cell_ids": invalidated,
            "moved_motivation_cell_ids": [cell.id for cell in motivation]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    args = parser.parse_args()
    path = args.notebook.resolve()
    notebook = nbformat.read(path, as_version=4)
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    audit_root = ROOT / "output/analysis/ldp_gradient_far_positioning_notebook/revision_layout_round_coverage"
    audit_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = audit_root / f"before_{stamp}.ipynb"
    shutil.copy2(path, backup)
    audit = apply_repairs(notebook)
    nbformat.write(notebook, path)
    audit.update(input_sha256=before_hash, output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 notebook=str(path), backup=str(backup), cells=len(notebook.cells), kernel_executed=False)
    (audit_root / f"repair_{stamp}.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
