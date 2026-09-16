"""Regression checks without a notebook kernel or experiment execution."""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "notebook_positioning_layout_repair",
    ROOT / "scripts/notebook_positioning_layout_repair.py",
)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def current_notebook():
    return nbformat.read(repair.DEFAULT_NOTEBOOK, as_version=4)


def test_layout_migration_is_idempotent_and_preserves_unrelated_sources():
    notebook = current_notebook()
    before = {cell.id: cell.source for cell in notebook.cells}
    audit = repair.apply_repairs(notebook)
    after = {cell.id: cell.source for cell in notebook.cells}
    allowed_changes = set(audit["edited_cell_ids"]) | set(audit["removed_cell_ids"])
    assert all(after[key] == source for key, source in before.items() if key not in allowed_changes)
    assert not (set(repair.LEGACY_SECTIONS) & set(after))
    headings = [cell.source.splitlines()[0] for cell in notebook.cells
                if cell.cell_type == "markdown" and cell.source.startswith("## ")]
    motivation = next(i for i, heading in enumerate(headings) if heading.startswith("## 8.1 "))
    assert headings[motivation - 1].startswith("## 8.")
    assert headings[motivation + 1].startswith("## 9.")
    assert headings[-1].startswith("## 11. Développement et évaluation de RCIG")
    assert all("section 2.1" not in cell.source.lower() for cell in notebook.cells)
    assert all("section 15" not in cell.source.lower() for cell in notebook.cells)
    once = copy.deepcopy(notebook)
    second = repair.apply_repairs(notebook)
    assert not second["edited_cell_ids"]
    assert not second["removed_cell_ids"]
    assert notebook == once


def test_sparse_fairness_points_are_visible_without_imputation(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notebook = current_notebook()
    repair.apply_repairs(notebook)
    cells = {cell.id: cell for cell in notebook.cells}
    scope = {"pd": pd, "np": np, "plt": plt, "display": lambda *args: None,
             "Markdown": str}
    # Only the two function definitions are evaluated, never notebook cells.
    for cell_id, function_name in [
        ("180afe8c731ef31f", "_positioning_round_coverage_note"),
        ("rcig-review-alpha-accuracy", "plot_all_alpha_trajectories"),
    ]:
        node = next(item for item in ast.parse(cells[cell_id].source).body
                    if isinstance(item, ast.FunctionDef) and item.name == function_name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "plot-regression", "exec"), scope)

    observed_rounds = list(range(1, 20, 2)) + [20]
    values = np.array([float(r) if r in observed_rounds else np.nan for r in range(1, 21)])
    data = pd.DataFrame({"reference": "F_CC", "n_clients": 10, "C_local": 8.,
                         "horizon": 20, "alpha": 0., "round": np.arange(1, 21),
                         "gap_pp": values})
    figures = []
    scope.update(alpha_all=data, ALPHA_REFERENCES_TO_PLOT=["F_CC"],
                 ALPHA_CLIENTS_TO_PLOT=[10], _save_alpha_figure=lambda fig, name: figures.append(fig))
    scope["plot_all_alpha_trajectories"]("gap_pp", "Gap (pp)")
    assert len(figures) == 2  # Shared alpha=0 in both signed-alpha panels.
    for fig in figures:
        axis = fig.axes[0]
        line = axis.lines[0]
        assert line.get_marker() == "o"
        np.testing.assert_equal(line.get_ydata(), values)
        np.testing.assert_equal(line.get_xdata()[np.isfinite(line.get_ydata())], observed_rounds)
        assert "11/20" in axis.get_xlabel()
        plt.close(fig)
    np.testing.assert_equal(data.gap_pp.to_numpy(), values)


def test_legacy_updater_applies_layout_repair_before_preservation_check():
    source = (ROOT / "scripts/update_ldp_gradient_far_positioning_notebook_rcig.py").read_text()
    assert "layout_audit = apply_repairs(notebook)" in source
    assert source.index("layout_audit = apply_repairs(notebook)") < source.index("new_by_id =")
