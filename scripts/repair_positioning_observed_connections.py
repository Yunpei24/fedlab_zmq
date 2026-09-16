"""Surgical display-only migration of the current editable notebook."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
import nbformat

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'notebooks/LDP_Gradient_FAR_Positioning_Analysis_Editable.ipynb'
OUT=ROOT/'output/analysis/notebook_observed_connections_20260915'


def transform(nb):
    before=copy.deepcopy(nb)
    cells={c.id:c for c in nb.cells}
    c=cells['180afe8c731ef31f']
    if 'display-only observed-point connection' not in c.source:
        c.source=c.source.replace('Mark observations without filling or joining gaps in client evaluation.',
          'Mark observations; segments are visual guides, never imputed measurements.')
        c.source=c.source.replace('f"Tour — {len(observed)}/{len(expected)} tours évalués ; aucune imputation"',
          'f"Tour — {len(observed)}/{len(expected)} évalués ; points mesurés, segments indicatifs"')
        old='            group = group.sort_values("round")'
        assert c.source.count(old)==1
        c.source=c.source.replace(old,old+'\n            # display-only observed-point connection; do not mutate rounds_df.\n'
            '            group = group.loc[(group["n"] > 0) & np.isfinite(group["mean"])].copy()')
        c.source=c.source.replace('mask = group["n"] >= 2','mask = (group["n"] >= 2) & np.isfinite(group["sd"])')
    c=cells['rcig-review-alpha-accuracy']
    if 'display-only observed-point connection' not in c.source:
        old='                stats = part.groupby("round")[metric].agg(["mean", "std", "count"])'
        assert c.source.count(old)==1
        c.source=c.source.replace(old,old+'\n                # display-only observed-point connection; no interpolated rows.\n'
          '                stats = stats.loc[(stats["count"] > 0) & np.isfinite(stats["mean"])].copy()')
        c.source=c.source.replace('mask = stats["count"] > 1','mask = (stats["count"] > 1) & np.isfinite(stats["std"])')
    note=('**Lecture des points et des segments.** La loss cliente, le Worst-20 et le gap sont '
      'mesurés aux tours 1, 3, 5, …, 19 et 20. Les anciens tracés étaient coupés par les valeurs '
      'manquantes des autres tours : ce n’était pas une erreur des métriques. Les points '
      'mesurés sont maintenant reliés par des segments pour guider la lecture. **Aucune valeur '
      'intermédiaire n’est ajoutée aux données ou aux statistiques** ; les segments et les bandes '
      'entre observations ne sont pas de nouvelles évaluations. Les bandes représentent '
      'moyenne ± écart-type, pas un IC95. Accuracy et loss test sont disponibles aux 20 tours.')
    for cid in ('rcig-review-alpha-intro',):
        c=cells[cid]
        if '**Points réellement observés.**' in c.source:
            start=c.source.index('**Points réellement observés.**')
            # Old note was the final paragraph; preserve all text before it.
            c.source=c.source[:start]+note
        elif '**Lecture des points et des segments.**' not in c.source:
            c.source+='\n\n'+note
    intro=next(c for c in nb.cells if c.cell_type=='markdown' and c.source.startswith('## 3. Trajectoires'))
    if '**Lecture des points et des segments.**' not in intro.source:
        intro.source+='\n\n'+note
    changed=[c.id for c,o in zip(nb.cells,before.cells) if c.source!=o.source]
    assert len(nb.cells)==len(before.cells)
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,c.id,'exec')
    nbformat.validate(nb)
    return changed


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    nb=nbformat.read(PATH,as_version=4)
    original=PATH.read_bytes()
    changed=transform(nb)
    check=copy.deepcopy(nb)
    assert not transform(check) and check==nb
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=OUT/f'before_{stamp}.ipynb'
    shutil.copy2(PATH,backup)
    nbformat.write(nb,PATH)
    audit=dict(notebook=str(PATH),backup=str(backup),edited_cells=changed,
        original_sha256=hashlib.sha256(original).hexdigest(),
        updated_sha256=hashlib.sha256(PATH.read_bytes()).hexdigest(),
        outputs_pending_reexecution=True,metrics_changed=False)
    (OUT/'repair.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
