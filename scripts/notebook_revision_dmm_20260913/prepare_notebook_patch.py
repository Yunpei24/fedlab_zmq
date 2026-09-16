"""Emit a minimal source patch; preserve every unrelated user cell and output."""
import copy
import difflib
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "notebooks/LDP_Gradient_FAR_Positioning_Analysis_Editable.ipynb"
BACKUP = ROOT / "output/validation/notebook_dmm_revision_20260913"
BACKUP.mkdir(parents=True, exist_ok=True)
old = TARGET.read_text()
digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
backup = BACKUP / f"original_{digest[:12]}.ipynb"
if not backup.exists():
    shutil.copy2(TARGET, backup)
data = json.loads(old)
original = copy.deepcopy(data)
by_id = {c["id"]: c for c in data["cells"]}

def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)

cell = by_id["180afe8c731ef31f"]
source = "".join(cell["source"])
start = source.index('display(Markdown(\n    r"**Lecture correcte.')
set_source(cell, source[:start] + '''display(Markdown(
    "**Lecture correcte.** Le contrôle sans DP à α=2 termine plus bas dans "
    "cette campagne. Le clipping local est conservé ; le clipping serveur "
    "ne s’active dans aucun des 18 runs (voir l’audit ci-dessus). Les poids "
    "finaux sont observés moins concentrés sous DP. Cette association est "
    "compatible avec une modification du tilting par le bruit, mais ne "
    "démontre pas un gain causal de la DP. Le seuil 2/n est ici diagnostiqué, "
    "pas imposé. La confirmation F à trois seeds manque de sans-DP/α=0 et "
    "l’appariement strict des tirages historiques n’est pas certifié. "
    "L’ancienne explication par un générateur unique batch/bruit était trop "
    "affirmative, notamment sur MPS. Le témoin sans-DP/α=0 de l’écran E "
    "(seed 137) est présenté séparément en section 3.2."
))
''')
cell = by_id["rcig-review-controls-intro"]
source = "".join(cell["source"])
old_phrase = """Dans E, les configurations sont identiques
hors α et mécanisme DP, mais une même seed ne garantit pas les mêmes batches
après chaque tirage de bruit : ce n'est pas un appariement aléatoire strict."""
assert old_phrase in source
set_source(cell, source.replace(old_phrase, """Dans E, les configurations sont identiques
hors α et mécanisme DP. Une même seed ne certifie cependant pas, à elle seule,
l’appariement des tirages : les traces disponibles ne l’établissent pas.
Il ne faut pas attribuer un éventuel désappariement à un générateur partagé
sans vérification de l’exécution historique sur MPS."""))

set_source(by_id["rcig-motivation-60e447f70b45f192"], """### 11.5 Statut de cette archive et verdict scientifique actuel

Cette section 11 conserve les étapes de motivation et de validation initiales.
Les statements « R2/R3 non exécutés » concernaient l’état initial du protocole
v2 ; ils ne décrivent plus l’ensemble des campagnes réalisées depuis.

**Le bilan actualisé se trouve en section 8.2**, calculé à partir de la campagne
RCIG N10 terminée : 132 runs, dont 120 comparaisons. La moyenne temporelle
améliore l’erreur de référence, sans gain final substantiel. RCIG sans gel et
vue récente ont la même accuracy enregistrée dans 21 des 24 paires.
Les attaques persistantes ne sont pas résolues par cette instanciation.

**Statut : diagnostic scientifique établi, pas nouvelle méthode publiable
validée.** Le diagnostic distinct `aggregation_role_n10_v1` compare uniforme,
RFA directe, FAR + RFA et FAR + moyenne temporelle sous DP, sur les mêmes
messages privés et en entraînement complet. Son instantané est en section 8.2.
Il n’autorise aucune revendication anticipée de supériorité.
""")

new_specs = {
    "51a8d33cea6d1b93": [("dmm-audit-f-protocol", "audit_protocol.py")],
    "rcig-motivation-901d302253613f82": [
        ("dmm-bilan-reference", "bilan_reference.py"),
        ("dmm-bilan-rcig", "bilan_rcig.py"),
        ("dmm-bilan-decision", "bilan_decision.py"),
    ],
}
cells = []
for cell in data["cells"]:
    cells.append(cell)
    for cell_id, filename in new_specs.get(cell["id"], []):
        assert cell_id not in by_id, "Revision already integrated"
        cells.append({
            "cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {"editable": True, "deletable": True, "tags": ["dmm-review-20260914"]},
            "outputs": [], "source": (Path(__file__).parent / filename).read_text().splitlines(keepends=True),
        })
data["cells"] = cells
modified_ids = {"180afe8c731ef31f", "rcig-review-controls-intro", "rcig-motivation-60e447f70b45f192"}
new_by_id = {c["id"]: c for c in cells}
for cell in original["cells"]:
    if cell["id"] not in modified_ids:
        assert cell == new_by_id[cell["id"]]
assert digest == hashlib.sha256(TARGET.read_bytes()).hexdigest(), "Notebook changed concurrently"
new = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
diff = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True), n=3))
print("*** Begin Patch")
print("*** Update File: " + str(TARGET.relative_to(ROOT)))
for line in diff[2:]:
    if line.startswith("@@"):
        print("@@")
    else:
        print(line, end="")
print("*** End Patch")
