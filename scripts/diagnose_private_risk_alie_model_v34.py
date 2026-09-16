"""Local gradient/report ALIE factorial with validation-model effects; MPS only."""
import fcntl
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.audit_private_risk_continuation_v33 import exact
from privacy.fair_objective import require_mps, losses
from privacy.capped_private_risk import weights
from privacy.private_risk_winsor_v32 import winsor
from privacy.stable_weighted_rfa import weighted_rfa

SOURCE = ROOT / 'results/ldp_gradient_far/private_risk_continuation_v33'
OUT = ROOT / 'results/ldp_gradient_far/private_risk_alie_model_v34'
DOC = ROOT / 'output/analysis/Private_Risk_V34_ALIE_Local_Attribution_Protocol.md'
DEST = ROOT / 'output/analysis/Private_Risk_V34_ALIE_Local_Attribution'
SEEDS = (170501, 170502)
METHODS = ('risk_mean', 'risk_rfa', 'risk_winsor')
CELLS = ('00', '10', '01', '11')


def phi(r):
    if not math.isfinite(r) or not 0 <= r <= 1:
        raise ValueError('Half-Brier risk must belong to [0,1]')
    return r + 2*r*r if r <= .5 else 3*r - .5


def contrast(v):
    if set(v) != set(CELLS):
        raise ValueError('All four cells required; no imputation')
    return dict(gradient_original_report=v['10']-v['00'],
                report_original_gradient=v['01']-v['00'],
                report_forged_gradient=v['11']-v['10'],
                gradient_forged_report=v['11']-v['01'],
                interaction=v['11']-v['10']-v['01']+v['00'])


def summary(ev):
    out = {k: float(v) for k, v in exact(ev, range(2, 10)).items()}
    cs = ev['clients'][2:]
    for k in ('ce_loss', 'brier_loss'):
        out[k] = st.mean(c[k] for c in cs)
    out['balanced_accuracy_pct'] = 100*st.mean(c['balanced_accuracy'] for c in cs)
    out['J_val'] = st.mean(phi(c['brier_loss']) for c in cs)
    return out


def validation_gradient(model, data, evaluation):
    """Gradient of mean Phi(mean half-Brier), never sent to the mechanism."""
    require_mps()
    ps = [p for p in model.parameters() if p.requires_grad]
    grad = torch.zeros(sum(p.numel() for p in ps), device='mps')
    for cid in range(2, 10):
        coeff = 1 + 4*min(evaluation['clients'][cid]['brier_loss'], .5)
        ids = data['val'][cid]
        assert len(ids) == 1200
        for ix in ids.split(256):
            val = losses(model(data['x'][ix]), data['y'][ix], 'brier').sum()
            gs = torch.autograd.grad(val*(coeff/(8*len(ids))), ps)
            grad += torch.cat([g.detach().flatten() for g in gs])
    assert bool(torch.isfinite(grad).all())
    return grad


def aggregate(x, r, method):
    lam, _ = weights(r, .5)
    if method == 'risk_mean':
        return (lam[:, None]*x).sum(0), dict(solver_gap=0.)
    if method == 'risk_rfa':
        a, di = weighted_rfa(x, lam)
        return a, dict(solver_gap=di['unsmoothed_objective_gap_upper'])
    a, di, _ = winsor(x, r, C=2., f_budget=2)
    return a, dict(solver_gap=di['pilot_solver']['unsmoothed_objective_gap_upper'],
                   clipped_count=di['clipped_count'], factors=di['factors'], radius=di['radius'])


def run():
    require_mps()
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    stamp = dict(manifest['stamp'])
    auditpath = ROOT/'output/analysis/Private_Risk_V33_Continuation_Audit_Partial_Failure.json'
    audit = json.loads(auditpath.read_text())
    # Require both completed ALIE candidate trajectories from the final-step audit.
    assert audit['audit_passed'] and audit['local_gate_passed'] is False
    assert audit['source_stamp'] == stamp and audit['test_evaluated'] is False
    for p, digest in audit['inputs'].items():
        assert base.digest(ROOT/p) == digest
    assert {r['seed'] for r in audit['final_step_replays']
            if r['attack'] == 'persistent_alie' and r['method'] == 'risk_winsor'
            and r['exact_validation_counts']} == set(SEEDS)
    assert json.loads((SOURCE/'status.json').read_text())['status'] == 'failed'
    for p in (Path(__file__), DOC, auditpath,
              ROOT/'tests/test_private_risk_alie_model_v34.py',
              ROOT/'scripts/audit_private_risk_continuation_v33.py'):
        stamp[str(p.relative_to(ROOT))] = base.digest(p)
    for seed in SEEDS:
        f = SOURCE/f'seed{seed}__persistent_alie__risk_winsor'
        status = json.loads((f/'orchestration_status.json').read_text())
        assert status['status'] == 'completed' and status['device'] == 'mps'
        for fn, k in (('metrics.json', 'metrics_sha256'), ('checkpoint.pt', 'checkpoint_sha256')):
            assert base.digest(f/fn) == status[k]
            stamp[str((f/fn).relative_to(ROOT))] = status[k]
        stamp[str((f/'orchestration_status.json').relative_to(ROOT))] = base.digest(f/'orchestration_status.json')
    base.verify_stamp(stamp)
    signature = dict(source_stamp=stamp, seeds=list(SEEDS), methods=list(METHODS),
                     cells=list(CELLS), eta=.5, diagnostic=True, confirmation=False)
    if (OUT/'results.json').exists():
        old = json.loads((OUT/'results.json').read_text())
        assert old['signature'] == signature
        print('V34 already complete; source signatures verified')
        return old
    if (OUT/'manifest.json').exists():
        assert json.loads((OUT/'manifest.json').read_text()) == signature
    else:
        base.save(OUT/'manifest.json', signature)
    profile = json.loads((ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28/manifest.json').read_text())['profile']
    rows, states, contrasts = [], [], []
    for seed in SEEDS:
        folder = SOURCE/f'seed{seed}__persistent_alie__risk_winsor'
        cp = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
        assert cp['signature']['source_stamp'] == manifest['stamp']
        assert cp['rows'][-1]['step'] == 12 and cp['rows'][-1]['attack']['active']
        data, model = base.prepare(profile, seed), base.new_model(profile, seed)
        model.load_state_dict(cp['pre_round_model'])
        before = base.evaluate(model, data, 'val')
        bs = summary(before)
        grad = validation_gradient(model, data, before)
        x0, r0 = cp['last_private_messages'].to('mps'), cp['last_reports'].to('mps')
        x1, r1 = x0.clone(), r0.clone()
        x1[:2] = x0[2:].mean(0) + 1.5*x0[2:].std(0, correction=0)
        r1[:2] = 1.
        assert torch.equal(x1[2:], x0[2:]) and torch.equal(r1[2:], r0[2:])
        for i, x in enumerate(x0):
            assert base.ids_hash(x) == cp['rows'][-1]['clients'][i]['private_message_hash']
        mu = cp['last_clean_means'].to('mps')
        lam, _ = weights(r0, .5)
        p = lam[2:]/lam[2:].sum()
        target = (p[:, None]*mu[2:]).sum(0)
        states.append(dict(seed=seed, evaluation=before, honest=bs,
                           oracle_gradient_norm=float(torch.linalg.vector_norm(grad))))
        for method in METHODS:
            vectors, mr = {}, {}
            for cell in CELLS:
                x = x0 if cell[0] == '0' else x1
                r = r0 if cell[1] == '0' else r1
                a, diag = aggregate(x, r, method)
                row = dict(seed=seed, method=method, cell=cell, device='mps', diagnostics=diag,
                           evaluated=False, byzantine_risk_mass=float(weights(r, .5)[0][:2].sum()))
                if not bool(torch.isfinite(a).all()):
                    raise FloatingPointError('Nonfinite aggregate')
                if diag['solver_gap'] > .001:
                    row['missing_reason'] = 'original_40_iteration_certificate_exceeds_0.001'
                    rows.append(row)
                    continue
                model.load_state_dict(cp['pre_round_model'])
                base.apply_gradient(model, a, .5)
                ev = base.evaluate(model, data, 'val')
                hs = summary(ev)
                alignment = float((grad*a).sum())
                hs.update(aggregate_error_squared=float((a-target).square().sum()),
                          oracle_alignment=alignment, first_order_delta_J=-.5*alignment,
                          actual_delta_J=hs['J_val']-bs['J_val'],
                          remainder_J=hs['J_val']-bs['J_val']+.5*alignment)
                if method == 'risk_winsor' and cell == '11':
                    torch.testing.assert_close(.5*a, cp['last_step'].to('mps'), rtol=2e-5, atol=2e-6)
                    for k, val in model.state_dict().items():
                        torch.testing.assert_close(val, cp['model'][k].to('mps'), rtol=2e-5, atol=2e-6)
                    old = cp['rows'][-1]['validation']
                    for v, w in zip(ev['clients'], old['clients']):
                        assert v['class_hits'] == w['class_hits'] and v['class_count'] == w['class_count']
                    for k in ('ce_loss', 'brier_loss'):
                        assert math.isclose(ev[k], old[k], rel_tol=2e-6, abs_tol=2e-6)
                    row['observed_V33_step_replayed'] = True
                row.update(evaluated=True, evaluation=ev, honest=hs)
                rows.append(row)
                vectors[cell], mr[cell] = a, hs
                print(f"V34 {seed} {method} {cell}: acc={hs['accuracy_pct']:.4f} W={hs['worst20_pct']:.4f} J={hs['J_val']:.6f}", flush=True)
                base.save(OUT/'status.json', dict(status='running', pid=os.getpid(), device='mps',
                    attempted=len(rows), expected=24, updated_unix=time.time()))
                base.save(OUT/'partial.json', dict(signature=signature, rows=rows, states=states))
            if len(vectors) == 4:
                G, R = vectors['10']-vectors['00'], vectors['01']-vectors['00']
                I = vectors['11']-vectors['10']-vectors['01']+vectors['00']
                D = vectors['11']-vectors['00']
                torch.testing.assert_close(D, G+R+I, rtol=3e-5, atol=2e-6)
                decomp = sum(v.square().sum() for v in (G, R, I)) + 2*sum((v*w).sum() for v,w in ((G,R),(G,I),(R,I)))
                torch.testing.assert_close(D.square().sum(), decomp, rtol=3e-5, atol=2e-6)
                contrasts.append(dict(seed=seed, method=method, identities_passed=True,
                    metrics={k: contrast({c: mr[c][k] for c in CELLS}) for k in mr['00']}))
        del model, data, cp, grad
        torch.mps.empty_cache()
    assert len(rows) == 24
    assert sum(r.get('observed_V33_step_replayed', False) for r in rows) == 2
    base.verify_stamp(stamp)
    result = dict(signature=signature, rows=rows, states=states, contrasts=contrasts,
                  attempted=24, evaluated=sum(r['evaluated'] for r in rows), device='mps',
                  independent_confirmation=False, global_validation=False,
                  V33_changed=False, new_private_gradient_draws=0, test_evaluated=False,
                  oracle_feeds_mechanism=False, research_exports_private=False)
    base.save(OUT/'results.json', result)
    base.save(DEST.with_suffix('.json'), result)
    base.save(OUT/'status.json', dict(status='completed', pid=os.getpid(), device='mps',
              attempted=24, evaluated=result['evaluated'], results_sha256=base.digest(OUT/'results.json')))
    return result


def report(result):
    lines = ['# V34 — effet local des gradients et rapports ALIE sur le modèle', '',
        f"**{result['evaluated']}/24 pas virtuels évalués sur MPS ; deux états de calibration V33 déjà contaminés.**", '',
        'Chaque cellule repart du même modèle pré-pas 12 pour une seed. RFA et moyenne sont aussi évaluées à cet état V32. '
        'Ce tableau ne compare pas leurs trajectoires propres et ne décompose pas causalement les 12 pas.', '',
        '00 : aucune falsification au pas courant ; 10 : gradients seuls ; 01 : rapports seuls ; 11 : les deux. '
        'Les modèles 00 ne sont pas des trajectoires sans attaque : leur passé est déjà attaqué.', '',
        'Métriques sur les huit clients honnêtes, validation uniquement. E est l’erreur quadratique par rapport '
        'à la moyenne honnête clippée pondérée. J est un risque équitable de validation, pas la variance des accuracies.', '',
        '| Seed | Règle | Cellule | Acc (%) | Worst-20 (%) | Gap (pp) | Var (pp²) | CE | Brier | J | E |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|']
    for row in result['rows']:
        prefix = f"| {row['seed']} | {row['method']} | {row['cell']} |"
        if row['evaluated']:
            lines.append(prefix+' '+' | '.join(f"{row['honest'][k]:.6f}" for k in
                ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','ce_loss','brier_loss','J_val','aggregate_error_squared'))+' |')
        else:
            lines.append(prefix+' Non évalué : certificat solveur | — | — | — | — | — | — | — |')
    lines += ['', '## Effet des rapports falsifiés lorsque les gradients sont déjà falsifiés', '',
        'Différences 11 moins 10, à modèle et gradients identiques. Positif est favorable pour accuracy/Worst-20, '
        'défavorable pour les losses et E. Un effet local favorable ne rend pas le rapport byzantin fiable.', '',
        '| Seed | Règle | Δ Acc (pp) | Δ Worst-20 (pp) | Δ CE | Δ J | Δ E |',
        '|--:|:--|--:|--:|--:|--:|--:|']
    for c in result['contrasts']:
        lines.append(f"| {c['seed']} | {c['method']} | "+' | '.join(f"{c['metrics'][k]['report_forged_gradient']:+.6f}"
            for k in ('accuracy_pct','worst20_pct','ce_loss','J_val','aggregate_error_squared'))+' |')
    lines += ['', '## Direction et pas fini', '',
        'Prédiction : −0,5 ⟨∇J_val, A⟩. Reste = variation réelle de J moins cette prédiction. '
        'Il est mesuré, pas borné par une constante de lissité estimée. Le gradient est un oracle hors mécanisme.', '',
        '| Seed | Règle | Cellule | Variation réelle J | Premier ordre | Reste |',
        '|--:|:--|:--|--:|--:|--:|']
    for r in result['rows']:
        if r['evaluated']:
            lines.append(f"| {r['seed']} | {r['method']} | {r['cell']} | "+' | '.join(f"{r['honest'][k]:+.7f}"
                for k in ('actual_delta_J','first_order_delta_J','remainder_J'))+' |')
    lines += ['', 'Les deux sorties V32/11 reproduisent les pas appliqués et les comptages de validation de V33. '
        'Aucune nouvelle seed ni donnée de test, aucun nouveau tirage de gradient privé, aucun seuil ajusté. '
        'Les exports oracle et de validation ne sont pas privés. V29/V33 restent en échec ; V30 reste fermé.', '',
        '[Protocole fixé avant calcul](Private_Risk_V34_ALIE_Local_Attribution_Protocol.md) · '
        '[Cellules, contrastes complets et empreintes](Private_Risk_V34_ALIE_Local_Attribution.json)']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'diagnostic.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            result = run()
            report(result)
            print(json.dumps(dict(attempted=result['attempted'], evaluated=result['evaluated'], global_validation=False)))
        except Exception as exc:
            base.save(OUT/'status.json', dict(status='failed', pid=os.getpid(), device='mps', error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
