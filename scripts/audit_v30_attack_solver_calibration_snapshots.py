#!/usr/bin/env python3
"""MPS-only, frozen 40-iteration solver preflight on old V28 private messages.

No model/data evaluation, no new gradient query, no use of V29 confirmation
messages and no attack training. These are single-snapshot interventions,
not trajectories of attack/recovery and not additional independent seeds.
"""
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
from privacy.fair_objective import require_mps
from privacy.private_risk_attacks import inject
from privacy.private_risk_message_safety import sanitize
from privacy.scheduled_private_risk import aggregate
from privacy.stable_weighted_rfa import stable_norm
from scripts import run_fair_objective_screen as base

PARENT = ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
DEST = ROOT/'output/analysis/V30_Attack_Solver_V28_Snapshot_Preflight'
METHODS = ('erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa')
SEEDS = (170501, 170502)
CONDITIONS = ('none', 'abrupt_bf', 'persistent_alie', 'slow_ipm')


def main():
    require_mps()
    manifest = json.loads((PARENT/'manifest.json').read_text())
    stamp = dict(manifest['source_stamp']); base.verify_stamp(stamp)
    assert json.loads((PARENT/'status.json').read_text())['status'] == 'completed'
    files = [Path(__file__), ROOT/'privacy/private_risk_attacks.py',
        ROOT/'privacy/private_risk_message_safety.py', ROOT/'privacy/scheduled_private_risk.py',
        ROOT/'privacy/stable_weighted_rfa.py', ROOT/'privacy/capped_private_risk.py',
        ROOT/'configs/ldp_gradient_far/full_population_private_risk_attacks_v30.yaml',
        ROOT/'output/analysis/Full_Population_Private_Risk_Attacks_V30_Protocol.md']
    stamp.update({str(p.relative_to(ROOT)): base.digest(p) for p in files})
    rows, sources = [], {}
    for seed in SEEDS:
        for source_method in METHODS:
            name = f'seed{seed}__{source_method}'; folder = PARENT/name
            status = json.loads((folder/'orchestration_status.json').read_text())
            assert status['status'] == 'completed' and status['device'] == 'mps'
            paths = {'checkpoint.pt': 'checkpoint_sha256', 'metrics.json': 'metrics_sha256',
                     'simulator_oracle.json': 'oracle_sha256'}
            hashes = {filename: base.digest(folder/filename) for filename in paths}
            assert all(hashes[filename] == status[k] for filename, k in paths.items())
            sources[name] = hashes
            cp = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
            assert cp['round'] == 120 and cp['source_stamp'] == manifest['source_stamp']
            messages = cp['last_private_messages'].to('mps')
            reports = None if cp['last_reports'] is None else cp['last_reports'].to('mps')
            assert messages.shape == (10, 61706) and bool(torch.isfinite(messages).all())
            # Apply the matching risk/no-risk RFA even to a source trained by a mean.
            # This is explicitly a local counterfactual, not its trained RFA trajectory.
            method = 'risk_rfa' if source_method.startswith('risk_') else 'erm_rfa'
            for attack in CONDITIONS:
                # Public round 60 instantiates each declared attack at full amplitude.
                # It is NOT a claim that these pre-round120 messages came from round60.
                sent, rr, threat = inject(messages, reports, attack=attack, round_number=60)
                sent, rr, safety = sanitize(sent, rr)
                assert safety['invalid_message_rows'] == safety['nonfinite_risk_reports'] == 0
                step, diag = aggregate(sent, rr, kind=method, round_number=120, horizon=120)
                center = step/diag['eta']; solver = diag['solver']
                assert bool(torch.isfinite(center).all()) and solver['iterations'] == 40
                effective = torch.tensor(solver['stationary_weights'], device='mps', dtype=torch.float32)
                rebuilt = (effective[:, None]*sent).sum(0)
                mass = sum(diag['objective_weights'][:2])
                assert mass <= 3/7+1e-6
                gap = solver['unsmoothed_objective_gap_upper']
                rows.append(dict(seed=seed, source_method=source_method, evaluated_solver_method=method,
                    source_pre_round=120, attack=attack, attack_instantiation_round=60, device='mps',
                    objective_byzantine_mass=mass, effective_byzantine_mass=sum(solver['stationary_weights'][:2]),
                    objective_gap_upper=gap, precision_threshold=.001, precision_pass=gap <= .001,
                    reconstruction_error=float(stable_norm(center-rebuilt)),
                    forged_message_norm=threat['forged_norm'], aggregation_norm=float(stable_norm(center)),
                    numerical_solver_diagnostic=solver, accuracy_evaluated=False))
            del cp, messages, reports; torch.mps.empty_cache()
    assert len(rows) == 32
    base.verify_stamp(stamp)
    payload = dict(device='mps', parent_campaign='full_population_private_risk_calibration_v28',
        source_states=8, calibration_seeds=list(SEEDS), evaluations=32, rows=rows, source_files=sources,
        source_stamp=stamp, candidate_changed=False, v29_confirmation_used=False,
        attack_training_launched=False, end_to_end_robustness_validated=False,
        precision_passes=sum(r['precision_pass'] for r in rows),
        scope='One fixed message set per old calibration state; no accuracy, no trajectory or statistical power claim')
    base.save(DEST.with_suffix('.json'), payload)
    lines = ['# V30 — précontrôle numérique sur les messages de calibration V28', '',
        '**Aucun entraînement attaqué lancé ; aucun résultat V29 utilisé.**', '',
        'Huit états connus de calibration (deux seeds, quatre trajectoires) ; quatre conditions par état. '
        'Les messages privés ont été produits avant le tour 120 de V28. Les formules publiques d’attaque '
        'sont instanciées à leur amplitude du tour 60 ; ce décalage est un contrefactuel local, pas un replay temporel.', '',
        'Chaque ensemble est traité par la RFA correspondante, même si sa trajectoire source utilisait la moyenne. '
        'Le solveur reste à 40 itérations. Aucune précision n’est ajustée après ces résultats.', '',
        '| Seed | Trajectoire source | Attaque | Borne d’écart d’objectif | Seuil 0,001 | Masse d’objectif corrompue | Masse effective |',
        '|--:|:--|:--|--:|:--|--:|--:|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['source_method']} | {r['attack']} | {r['objective_gap_upper']:.7g} | "
            f"{'respecté' if r['precision_pass'] else 'non satisfait'} | {r['objective_byzantine_mass']:.5f} | {r['effective_byzantine_mass']:.5f} |")
    lines += ['', f"**{payload['precision_passes']}/32 contrôles numériques satisfaits.**", '',
        'Un dépassement de la borne supérieure ne prouve pas que l’erreur réelle dépasse le seuil. '
        'Un passage ne démontre ni maintien de l’accuracy ni protection de la fairness. '
        'Les masses effectives reconstruisent le point renvoyé par le solveur et ne sont pas les coefficients de risque.', '',
        'Cette vérification ne remplace pas le gate propre V29 et n’ouvre pas V30. '
        'Aucun nouveau hyperparamètre, candidat, rayon ou nombre d’itérations n’est choisi. '
        'Les 32 cellules ne sont pas 32 réplications indépendantes.', '',
        '[Calculs et empreintes](V30_Attack_Solver_V28_Snapshot_Preflight.json) · '
        '[Protocole conditionnel](Full_Population_Private_Risk_Attacks_V30_Protocol.md).']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(device='mps', evaluations=32, precision_passes=payload['precision_passes'],
        max_objective_gap=max(r['objective_gap_upper'] for r in rows),
        by_attack={a: dict(passed=sum(r['precision_pass'] for r in rows if r['attack']==a),
                         count=sum(r['attack']==a for r in rows)) for a in CONDITIONS},
        training_launched=False, report=str(DEST.with_suffix('.md')))), flush=True)


if __name__ == '__main__':
    main()
