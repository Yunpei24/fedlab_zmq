"""Execute the edited analysis cells, then merge outputs only, with hash guard."""
import base64
import copy
import hashlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "notebooks/LDP_Gradient_FAR_Positioning_Analysis_Editable.ipynb"
validation = ROOT / "output/validation/notebook_dmm_revision_20260913"
data = nbformat.read(path, as_version=4)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
by_id = {cell.id: cell for cell in data.cells}
target_ids = [
    "dmm-audit-f-protocol", "180afe8c731ef31f", "e1d4df10ed669f19", "33c22da713380143",
    "dmm-bilan-reference", "dmm-bilan-rcig", "dmm-bilan-decision",
]
prep = nbformat.v4.new_code_cell('''
# Read the existing tidy export only for the unchanged section-3 figures.
# The following audit independently reloads the 18 original metrics.json files.
rounds_df = pd.read_csv(EXPORT_ROOT / "rounds_tidy.csv", low_memory=False)
assert rounds_df.dp_enabled.dtype == bool
''', id="dmm-execution-prep")
scratch = nbformat.v4.new_notebook(cells=[copy.deepcopy(by_id["74cf56be"]), prep]
    + [copy.deepcopy(by_id[i]) for i in target_ids])
scratch.metadata.kernelspec = {"name": "fedlab-venv", "display_name": "Python (fedlab-venv)", "language": "python"}
scratch.cells.append(nbformat.v4.new_code_cell('''
# Check the cached trajectory export against the freshly loaded original results.
cross = trajectory_final.merge(f_protocol_runs, left_on=["n_clients", "seed", "alpha", "dp_enabled"],
    right_on=["Clients", "seed", "α", "DP"], validate="one_to_one")
assert len(cross) == 18
assert np.allclose(cross.test_accuracy_pct, cross["Accuracy (%)"], atol=1e-9, rtol=0)
assert np.allclose(cross.test_loss, cross["Loss finale"], atol=1e-9, rtol=0)
assert np.allclose(cross.weight_concentration, cross["Concentration finale"], atol=1e-9, rtol=0)
print("18 final-row matches verified against original metrics; cached graphs are consistent.")
''', id="dmm-execution-check"))
client = NotebookClient(scratch, timeout=300, kernel_name="fedlab-venv",
                        resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
client.execute()
executed = {c.id: c for c in scratch.cells}
for cell_id in target_ids:
    assert not any(o.output_type == "error" for o in executed[cell_id].outputs)
next_count = max((c.get("execution_count") or 0 for c in data.cells), default=0)
for cell_id in target_ids:
    next_count += 1
    original = by_id[cell_id]
    result = executed[cell_id]
    assert original.source == result.source
    original.outputs = result.outputs
    original.execution_count = next_count
    original.metadata["execution"] = result.metadata.get("execution", {})
    for idx, output in enumerate(result.outputs):
        png = output.get("data", {}).get("image/png")
        if png:
            (validation / f"{cell_id}_{idx}.png").write_bytes(base64.b64decode(png))
nbformat.validate(data)
assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Concurrent notebook edit: outputs not merged"
# Generated notebook outputs; unrelated cells are left intact.
nbformat.write(scratch, validation / "selected_cells_executed.ipynb")
nbformat.write(data, path)
print(json.dumps({"executed_cells": target_ids, "cell_count": len(data.cells), "errors": 0}, indent=2))
