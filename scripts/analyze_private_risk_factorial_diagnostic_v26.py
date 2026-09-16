#!/usr/bin/env python3
"""Independent real-model audit and complete conditional contrasts of V26."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
from scripts import run_private_risk_factorial_diagnostic_v26 as run
from scripts import run_fair_objective_screen as base
from scripts.run_private_risk_confirmation_v12 import audit_evaluation
from privacy.fair_objective import require_mps, losses
from privacy.stable_weighted_rfa import weighted_rfa

REPORT = ROOT / 'output/analysis/Private_Risk_Factorial_Diagnostic_V26_Analyse'


def phi(r):
    return torch.where(r <= .5, r+2*r.square(), 3*r-.5)


def independent_population(model, data):
    """Different block size/order of gradient scaling from the production probe."""
    rs, gs = [], []
    params = tuple(p for p in model.parameters() if p.requires_grad)
    for ids in data['train']:
        r = torch.zeros((), device='mps')
        gradient_sum = torch.zeros(sum(p.numel() for p in params), device='mps')
        for batch in ids.split(256):
            loss_sum = losses(model(data['x'][batch]), data['y'][batch]).sum()
            gradients = torch.autograd.grad(loss_sum, params)
            gradient_sum += torch.cat([g.detach().flatten() for g in gradients])
            r += loss_sum.detach()
        rs.append(r/len(ids)); gs.append(gradient_sum/len(ids))
    return torch.stack(rs), torch.stack(gs)


def delta(a, b, keys):
    return {k: a[k]-b[k] for k in keys}


def contrast(candidate, baseline, family):
    return dict(family=family, seed=candidate['seed'], round=candidate['round'],
                candidate=candidate['treatment'], baseline=baseline['treatment'],
                effects=delta(candidate['effects'], baseline['effects'],
                  ('error_to_population_target_sq', 'error_to_clipped_batch_target_sq',
                   'predicted_J_decrease', 'actual_J_decrease', 'finite_step_remainder')),
                validation=delta(candidate['validation_after'], baseline['validation_after'],
                  ('accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2',
                   'balanced_accuracy_pct', 'ce_loss', 'brier_loss')))


def conditional_contrasts(rows):
    index = {(r['treatment']['weight'], r['treatment']['message'], r['treatment']['aggregator']): r for r in rows}
    assert len(index) == len(rows) == 12
    out = []
    for message in ('private', 'clean_clipped_oracle'):
        for agg in ('mean', 'rfa'):
            for w1, w0, family in [('oracle', 'private', 'remove_risk_noise'),
                                   ('uniform', 'private', 'uniform_vs_private_weights'),
                                   ('oracle', 'uniform', 'oracle_vs_uniform_weights')]:
                out.append(contrast(index[w1, message, agg], index[w0, message, agg], family))
    for weight in ('private', 'oracle', 'uniform'):
        for agg in ('mean', 'rfa'):
            out.append(contrast(index[weight, 'clean_clipped_oracle', agg], index[weight, 'private', agg], 'remove_gradient_noise'))
        for message in ('private', 'clean_clipped_oracle'):
            out.append(contrast(index[weight, message, 'rfa'], index[weight, message, 'mean'], 'rfa_vs_mean'))
    assert len(out) == 24
    return out


def interactions(rows):
    """Full risk-noise × gradient-noise interaction at each aggregator."""
    ix = {(r['treatment']['weight'], r['treatment']['message'], r['treatment']['aggregator']): r for r in rows}
    output = []
    for agg in ('mean', 'rfa'):
        terms = [ix['oracle', 'clean_clipped_oracle', agg], ix['private', 'clean_clipped_oracle', agg],
                 ix['oracle', 'private', agg], ix['private', 'private', agg]]
        values = {k: terms[0]['effects'][k]-terms[1]['effects'][k]-terms[2]['effects'][k]+terms[3]['effects'][k]
                  for k in ('actual_J_decrease', 'error_to_population_target_sq', 'finite_step_remainder')}
        output.append(dict(aggregator=agg, definition='(oracle-private weights) at clean messages minus same contrast at private messages', values=values))
    output.append(dict(aggregator='rfa_minus_mean', definition='three-way interaction, difference of the preceding two interactions',
                       values={k: output[1]['values'][k]-output[0]['values'][k] for k in output[0]['values']}))
    return output


def close_float(a, b):
    assert math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-8), (a, b)


def audit_state(folder, data, model, stamp):
    snapshot = torch.load(folder/'state.pt', map_location='cpu', weights_only=True)
    assert snapshot['source_stamp'] == stamp and not snapshot['privacy_protected']
    node_status = json.loads((folder/'orchestration_status.json').read_text())
    assert node_status['status'] == 'completed' and node_status['valid_probes'] == 12
    assert node_status['state_sha256'] == base.digest(folder/'state.pt')
    model.load_state_dict(snapshot['before'])
    run.assert_model_equal(model, snapshot['before'])
    risk_independent, gradients = independent_population(model, data)
    raw = snapshot['raw_risks'].to('mps')
    torch.testing.assert_close(risk_independent, raw, rtol=2e-5, atol=2e-7)
    # Fixed, pre-audit float32 tolerance for different batch size/accumulation.
    torch.testing.assert_close(gradients, snapshot['raw_population_gradients'].to('mps'), rtol=5e-4, atol=2e-7)
    coeff = 1+2*(raw/.5).clamp(max=1)
    oracle_lam = coeff/coeff.sum()
    target = snapshot['target']
    fullg = snapshot['raw_population_gradients'].to('mps')
    h = (oracle_lam[:, None]*fullg).sum(0)
    gj = (coeff[:, None]*fullg).mean(0)
    clean = snapshot['clean_messages'].to('mps')
    hb = (oracle_lam[:, None]*clean).sum(0)
    for name, value in [('h', h), ('GJ', gj), ('h_batch', hb), ('coefficients', coeff), ('oracle_weights', oracle_lam)]:
        torch.testing.assert_close(value, target[name].to('mps'), rtol=0, atol=0)
    j0 = float(phi(raw).mean())
    close_float(j0, target['J'])
    before_val = base.evaluate(model, data, 'val')
    run.same(before_val, snapshot['validation_before'], 'independent pre-probe validation')
    rows, risks = [], []
    paths = sorted(folder.glob('probe_*.json'))
    assert len(paths) == 12 and set(p.name for p in paths) == set(node_status['probe_sha256'])
    for path in paths:
        assert node_status['probe_sha256'][path.name] == base.digest(path)
        r = run.validate_probe(path, stamp)
        saved = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        assert saved['source_stamp'] == stamp and saved['treatment'] == r['treatment']
        t = r['treatment']
        messages = snapshot['private_messages'].to('mps') if t['message'] == 'private' else clean
        if t['weight'] == 'uniform':
            lam = torch.ones(len(raw), device='mps')/len(raw)
        else:
            risk = snapshot['private_reports'].to('mps') if t['weight'] == 'private' else raw
            a = 1+2*(risk.clamp(0, 1)/.5).clamp(max=1)
            lam = a/a.sum()
        A, solver = weighted_rfa(messages, lam) if t['aggregator'] == 'rfa' else ((lam[:, None]*messages).sum(0), None)
        assert torch.equal(A, saved['aggregate'].to('mps'))
        run.same(solver, r['aggregation']['solver'], 'independent solver diagnostics')
        run.same(lam.cpu().tolist(), r['aggregation']['objective_weights'], 'independent objective weights')
        eta = 2. if r['round'] <= 60 else .5
        if r['is_host_treatment']:
            assert torch.equal(eta*A, snapshot['host_step'].to('mps'))
        model.load_state_dict(snapshot['before'])
        base.apply_gradient(model, eta*A, 1.)
        # Independent risk recomputation with original query chunking, no noise.
        observed = []
        with torch.no_grad():
            for ids in data['train']:
                total = torch.zeros(1, device='mps')
                for block in ids.split(512):
                    total += losses(model(data['x'][block]), data['y'][block]).sum()/len(ids)
                observed.append(total.squeeze(0))
        observed = torch.stack(observed)
        assert torch.equal(observed, saved['after_risks'].to('mps'))
        j1 = float(phi(observed).mean())
        pred = eta*float(torch.dot(gj, A))
        expected = dict(error_to_population_target_sq=float((A-h).square().sum()),
                        error_to_clipped_batch_target_sq=float((A-hb).square().sum()),
                        eta=eta, step_norm=float(torch.linalg.vector_norm(eta*A)),
                        predicted_J_decrease=pred, actual_J_decrease=j0-j1,
                        J_before=j0, J_after=j1, finite_step_remainder=j1-j0+pred)
        for key, value in expected.items():
            close_float(value, r['effects'][key])
        val = base.evaluate(model, data, 'val')
        audit_evaluation(val)
        run.same(val, r['validation_after'], 'independent post-probe validation')
        for key, value in r['validation_change'].items():
            close_float(value, val[key]-before_val[key])
        rows.append(r)
    return dict(seed=snapshot['seed'], round=snapshot['round'], rows=rows,
                contrasts=conditional_contrasts(rows), interactions=interactions(rows),
                audit=dict(passed=True, independent_population_block_size=256,
                  gradient_tolerance=dict(rtol=5e-4, atol=2e-7), all_aggregates_bitwise=True,
                  all_post_step_risks_bitwise=True, all_validations_exact=True),
                state_sha256=base.digest(folder/'state.pt'),
                probe_sha256={p.name: base.digest(p) for p in paths})


def report(evidence):
    states = evidence['states']
    lines = ['# V26 — deux bruits, pondération et RFA : diagnostic causal', '',
        '**96 interventions d’un pas ; 8 états de 2 trajectoires de calibration.** '
        'Audit MPS indépendant terminé. Aucun test final, aucune attaque, aucune promotion de méthode.', '',
        'Les deux trajectoires hôtes reproduisent exactement V19, y compris les modèles terminaux bit à bit. '
        'Les risques, gradients de population, agrégats, risques après intervention et validations ont été recalculés. '
        'La dérivée de population a aussi été recalculée par blocs de 256 au lieu de 512, avec une tolérance float32 annoncée.', '',
        '## Lecture des contrastes', '',
        'Chaque différence est **intervention moins contrôle**, au même modèle, au même tour et sur les mêmes batchs. '
        'Δerreur² < 0 est favorable ; Δbaisse J > 0 signifie une baisse supplémentaire du vrai risque de population après le pas. '
        'Δaccuracy/Worst-20 > 0 est favorable. J est la moyenne du potentiel équitable des risques clients demi-Brier ; '
        'il n’est ni la CE ni une accuracy. Les validations portent sur les 1200 exemples réservés par client.', '',
        'Retirer un bruit est un **oracle non privé**, pas un mécanisme candidat à ε=4. '
        'L’uniforme emploie ici le bruit de gradient du canal risque, et non tout le budget ERM. '
        'Les états d’une même seed sont dépendants : aucun IC ne traite les huit états ou les 96 probes comme des runs indépendants.', '',
        '## Trois contrastes décisifs au même état', '']
    selections = [
        ('Retirer le bruit du rapport, en conservant gradients privés et RFA',
         lambda c: c['family']=='remove_risk_noise' and c['candidate']['message']=='private' and c['candidate']['aggregator']=='rfa'),
        ('Retirer le bruit des gradients, en conservant poids privés et RFA',
         lambda c: c['family']=='remove_gradient_noise' and c['candidate']['weight']=='private' and c['candidate']['aggregator']=='rfa'),
        ('RFA moins moyenne, avec poids et gradients privés',
         lambda c: c['family']=='rfa_vs_mean' and c['candidate']['weight']=='private' and c['candidate']['message']=='private')]
    for title, select in selections:
        lines += ['### '+title, '', '| Seed | Tour | Δerreur² vers h | Δbaisse J réelle | Δaccuracy (pp) | ΔWorst-20 (pp) |',
                  '|--:|--:|--:|--:|--:|--:|']
        selected = []
        for state in states:
            cs = [c for c in state['contrasts'] if select(c)]
            assert len(cs)==1
            c=cs[0]; selected.append(c)
            lines.append(f"| {c['seed']} | {c['round']} | {c['effects']['error_to_population_target_sq']:+.6g} | {c['effects']['actual_J_decrease']:+.7f} | {c['validation']['accuracy_pct']:+.4f} | {c['validation']['worst20_pct']:+.4f} |")
        lines += ['', f"Baisse J améliorée dans {sum(c['effects']['actual_J_decrease']>0 for c in selected)}/8 états, comptage descriptif uniquement.", '']
    lines += ['## Toutes les interventions, sans sélection', '',
              'Le reste = baisse prédite − baisse réalisée mesure l’écart de pas fini ; il ne doit pas être ignoré. '
              'Poids `oracle` : risque exact de population. Message `clean_clipped_oracle` : même batch, même clipping, sans bruit gaussien.', '']
    for state in states:
        lines += [f"### Seed {state['seed']}, avant tour {state['round']}", '',
            '| Poids | Messages | Agrégateur | Erreur² vers h | Baisse J prédite | Baisse J réelle | Reste | Acc. val. (%) | Worst-20 (%) |',
            '|:--|:--|:--|--:|--:|--:|--:|--:|--:|']
        for r in state['rows']:
            t,e,v=r['treatment'],r['effects'],r['validation_after']
            lines.append(f"| {t['weight']} | {t['message']} | {t['aggregator']} | {e['error_to_population_target_sq']:.6g} | {e['predicted_J_decrease']:+.7f} | {e['actual_J_decrease']:+.7f} | {e['finite_step_remainder']:+.7f} | {v['accuracy_pct']:.4f} | {v['worst20_pct']:.4f} |")
        lines += ['']
    lines += ['## Limites et suite autorisée', '',
        'Ces résultats identifient une marge au niveau d’un pas, pas une solution privée réalisable ni une amélioration end-to-end. '
        'Les deux seeds et les états étaient connus. Aucun changement de seuil, de coefficient ou de calendrier ne découle automatiquement de ce diagnostic. '
        'La sélection éventuelle d’un nouveau mécanisme nécessite de traiter son biais, son drift, son coût DP et sa robustesse, '
        'puis une nouvelle confirmation indépendante. V25 reste négatif.', '',
        'Le JSON contient les 192 contrastes conditionnels et les interactions bruit des poids × bruit des gradients × agrégateur, '
        'ainsi que toutes les métriques de validation, dont loss, variance, gap et balanced accuracy.', '',
        '[Protocole](Private_Risk_Factorial_Diagnostic_V26_Protocol.md) · '
        '[Preuves numériques](Private_Risk_Factorial_Diagnostic_V26_Analyse.json) · '
        '[Décision V25](Public_Temporal_Noise_Confirmation_V25_Decision.md).']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main():
    require_mps()
    manifest = json.loads((run.OUT/'manifest.json').read_text())
    stamp = manifest['source_stamp']; base.verify_stamp(stamp)
    status = json.loads((run.OUT/'status.json').read_text())
    assert status['status']=='completed' and status['valid_probes']==96 and status['replay_audits_passed']
    audit_stamp = {str(Path(__file__).relative_to(ROOT)): base.digest(__file__),
                   'tests/test_private_risk_factorial_v26_audit.py': base.digest(ROOT/'tests/test_private_risk_factorial_v26_audit.py')}
    states=[]; host_audits=[]
    for seed in manifest['config']['seeds']:
        host = json.loads((run.OUT/f'seed{seed}/replay_audit.json').read_text())
        assert host['passed'] and host['final_model_bitwise_equal'] and host['source_stamp']==stamp
        assert host['replay_checkpoint_sha256']==base.digest(run.OUT/f'seed{seed}/replay_checkpoint.pt')
        host_audits.append(host)
        data = base.prepare(manifest['profile'], seed)
        model = base.new_model(manifest['profile'], seed)
        for k in manifest['config']['pre_round_states']:
            base.verify_stamp(stamp); base.verify_stamp(audit_stamp)
            states.append(audit_state(run.OUT/f'seed{seed}/round{k}', data, model, stamp))
            print(f'V26 independent audit seed{seed} t{k}: 12/12 probes passed', flush=True)
        del data, model; torch.mps.empty_cache()
    ev=dict(audit_passed=True, states=states, host_audits=host_audits, source_stamp=stamp, audit_stamp=audit_stamp,
            total_probes=96, total_conditional_contrasts=sum(len(s['contrasts']) for s in states),
            oracle_only=True, privacy_protected=False, test_evaluated=False, global_validation=False,
            automatic_promotion=False, completed_unix=time.time())
    assert ev['total_conditional_contrasts']==192
    base.verify_stamp(stamp); base.verify_stamp(audit_stamp)
    base.save(REPORT.with_suffix('.json'), ev); report(ev)
    print('V26 independent audit passed; no candidate promoted.', flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--audit', action='store_true', required=True); p.parse_args(); main()
