"""Execute the revised display cells only, preserving user cells and run data."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/LDP_Gradient_FAR_Positioning_Analysis_Editable.ipynb"
OUT = ROOT / "output/analysis/notebook_requested_details_20260916"


def h(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    nb = nbformat.read(PATH, as_version=4)
    original = copy.deepcopy(nb)
    snapshot = h(PATH)
    by_id = {c.id: c for c in nb.cells}
    # Existing dependencies run unchanged, not a notebook regeneration.
    dependencies = {"74cf56be", "29876ebf", "09037e85", "180afe8c731ef31f"}
    affected = {"rcig-review-controls-table", "294b03073a3f9c6a", "7a7e19849d9a2299",
                "rcig-review-weight-bridge", "0ec66adcd7272fc7", "de8df6638be93b94"}
    for c in nb.cells:
        if c.cell_type != "code": continue
        source = c.source
        if c.id.startswith("requested-"):
            affected.add(c.id)
        if source.startswith('_plot_weight_metric(') or source.startswith('if len(weight_summary):'):
            affected.add(c.id)
        if source.startswith('attack_focus = final_df[') or source.startswith('def _plot_attack_reference('):
            dependencies.add(c.id)
    chosen = [c for c in nb.cells if c.id in dependencies | affected]
    for c in chosen: ast.parse(c.source)
    # Freeze historical input hashes, not the unrelated live V29 campaign.
    metrics = sorted((ROOT / "results/ldp_gradient_far/positioning_v3").rglob("metrics.json"))
    source_hashes = {str(p.relative_to(ROOT)): h(p) for p in metrics}
    small = nbformat.v4.new_notebook(cells=copy.deepcopy(chosen), metadata=copy.deepcopy(nb.metadata))
    for c in small.cells:
        c.outputs = []; c.execution_count = None
    check = nbformat.v4.new_code_cell('''
# Requirements verified from the executed data, not hard-coded conclusions.
e0 = control_table[(control_table.Phase == "E : exploratoire") &
                   (control_table["Bruit DP"] == "non") & (control_table["α"] == 0)]
assert len(e0) == 2 and set(e0.Disponible) == {"oui"}
f0 = control_table[(control_table.Phase == "F : confirmation") &
                   (control_table["Bruit DP"] == "non") & (control_table["α"] == 0)]
assert len(f0) == 2 and set(f0.Disponible) == {"absent"}
assert set(weight_summary.Seeds) == {3}
assert (weight_summary["n × poids min"] <= 1 + 1e-5).all()
assert (weight_summary["n × poids max"] >= 1 - 1e-5).all()
assert (weight_summary["n × Σ poids²"] >= 1 - 1e-5).all()
assert len(privacy_calibration_check) == 6
assert privacy_calibration_check["Écart ε enregistré"].abs().max() < 1e-8
import json
print("REQUIREMENTS_PASS", json.dumps({
    "nodp_alpha0_E_rows": len(e0), "nodp_alpha0_F_rows_available": int((f0.Disponible == "oui").sum()),
    "weight_runs": len(weight_run_medians), "weight_rounds": len(weights_rounds),
    "privacy_checks": len(privacy_calibration_check),
    "max_epsilon_error": float(privacy_calibration_check["Écart ε enregistré"].abs().max())
}))
''', id='temporary-revision-checks')
    small.cells.append(check)
    OUT.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(small, timeout=1200, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    except Exception:
        nbformat.write(small, OUT / "execution_failed.ipynb")
        raise
    assert h(PATH) == snapshot, "Notebook edited concurrently; refusing overwrite"
    assert source_hashes == {str(p.relative_to(ROOT)): h(p) for p in metrics}
    figures = []
    for c in small.cells:
        assert not any(o.output_type == "error" for o in c.get("outputs", [])), c.id
        if c.id not in affected: continue
        by_id[c.id].outputs = c.outputs
        by_id[c.id].execution_count = c.execution_count
        for k, output in enumerate(c.outputs):
            if "image/png" in output.get("data", {}):
                p = OUT / f"{c.id}_{k}.png"
                p.write_bytes(base64.b64decode(output.data["image/png"]))
                figures.append(str(p))
    for a, b in zip(original.cells, nb.cells):
        assert a.source == b.source and a.id == b.id
        if a.id not in affected: assert a == b, a.id
    nbformat.validate(nb)
    nbformat.write(nb, PATH)
    report = dict(notebook=str(PATH), executed_cells=[c.id for c in chosen],
        outputs_replaced=sorted(affected), other_cells_preserved=True,
        source_metrics_unchanged=True, metrics_files=len(metrics), figures=figures,
        checks=[o.get("text", "") for o in small.cells[-1].outputs],
        errors=0, before_execution_sha256=snapshot, after_execution_sha256=h(PATH))
    (OUT / "execution.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
