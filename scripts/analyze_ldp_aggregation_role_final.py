#!/usr/bin/env python3
"""Read-only scientific audit of the locked 96-run aggregation-role campaign.

Only this script's analysis directory/report are written. The training runner is
imported for its validation checks, never for execute/summarize/launch/run.
Temporal observations are first reduced within seed; candidates and trajectories
are not treated as additional independent replications.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts import run_ldp_aggregation_role_ablation as runner

OUT = ROOT / 'output/analysis/aggregation_role_n10_final'
REPORT = ROOT / 'output/analysis/Aggregation_Role_N10_Final_Analysis.md'
ARMS = runner.ARMS
NOISE = ('homogeneous', 'heteroscedastic')
SCENARIOS = ('none', 'bf_persistent', 'ipm_persistent', 'alie_persistent')
SEEDS = (920001, 920002, 920003)
METRICS = {
    'test_accuracy_pct': ('test_accuracy', 100),
    'client_accuracy_pct': ('client_accuracy_mean', 100),
    'test_loss': ('test_loss', 1),
    'client_loss': ('client_loss_mean', 1),
    'variance_pp2': ('client_accuracy_variance_pct2', 1),
    'worst20_pct': ('worst20_accuracy_pct', 1),
    'gap_pp': ('best20_worst20_gap_pct', 1),
    'balanced_accuracy_pct': ('mean_client_balanced_accuracy_pct', 1),
}
TERMS = ('honest_effective_noise_sq', 'honest_tilt_sq', 'byzantine_centered_sq',
         'cross_noise_tilt', 'cross_noise_byzantine', 'cross_tilt_byzantine')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = list(values)
    assert len(values) == 3, f'Expected three independent seeds, got {len(values)}'
    assert all(math.isfinite(v) for v in values)
    mean, sd = st.mean(values), st.stdev(values)
    half = 4.302652729911275 * sd / math.sqrt(3)
    return dict(n=3, mean=mean, sample_sd=sd, values=values,
                descriptive_t95_low=mean-half, descriptive_t95_high=mean+half)


def fmt(s, digits=2):
    return f"{s['mean']:.{digits}f} ± {s['sample_sd']:.{digits}f}"


def csv_dump(name, rows):
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (OUT / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def window_rows(rows, scenario, window='primary'):
    if window == 'primary':
        lo, hi = (13, 40) if scenario == 'none' else (17, 40)
    elif window == 'pre_attack':
        lo, hi = 13, 16
    elif window == 'persistent':
        lo, hi = 25, 40
    elif window == 'transition':
        lo, hi = 18, 24
    else:
        raise ValueError(window)
    return [r for r in rows if lo <= r['round_num'] <= hi]


def main():
    matrix = runner.matrix()
    campaign = runner.OUTPUT
    lock_path = campaign / 'campaign_lock.json'
    lock = json.loads(lock_path.read_text())
    runner.verify(lock, matrix)
    recomputed_privacy = runner.legacy.privacy(matrix)
    assert recomputed_privacy == lock['privacy'], 'Accountant recalculation differs from campaign lock'
    assert runner.status(matrix) == dict(completed=96, total=96, active=[], failed=[], missing=0)
    all_metrics = list(campaign.rglob('metrics.json'))
    assert len(all_metrics) == 96
    data, hashes, audit_rows, finals = {}, {}, [], []
    round_csv, candidate_csv, headroom_csv = [], [], []
    max_identity_abs = max_identity_relative = max_vector_residual = 0.0
    count_identity = 0
    last_finish = ''
    for task in runner.tasks(matrix):
        dest = runner.directory(task)
        status = json.loads((dest / 'orchestration_status.json').read_text())
        assert status['status'] == 'completed'
        assert status['device'] == 'mps' and status['fallback'] == 0
        assert sha(dest / 'resolved_config.yaml') == status['config_sha256']
        cfg = runner.config_for(matrix, task, lock)
        path, x, pairing = runner.validate(task, cfg, lock)
        assert sha(path) == status['metrics_sha256']
        assert sha(dest / 'simulator_randomness_private_audit.jsonl') == status['trace_sha256']
        # Compare every resolved-config field, not only the algorithm subsection.
        assert runner.yaml.safe_load((dest / 'resolved_config.yaml').read_text()) == cfg
        for artifact in [path, dest/'orchestration_status.json', dest/'resolved_config.yaml',
                         dest/'simulator_randomness_private_audit.jsonl', dest/'runtime_imports.json']:
            hashes[str(artifact.relative_to(ROOT))] = sha(artifact)
        key = tuple(task[k] for k in ('noise', 'scenario', 'arm', 'seed'))
        data[key] = x
        rows = x['rounds']
        final = dict(task)
        for label, (field, scale) in METRICS.items():
            if field not in rows[-1] and label == 'client_loss':
                continue  # No invented train loss or absent client loss.
            final[label] = rows[-1][field] * scale
        finals.append(final)
        last_finish = max(last_finish, status['finished_at'])
        audit_rows.append(dict(task, valid=True, metrics=str(path.relative_to(ROOT)),
                               rounds=40, device='mps:0', pairing=pairing,
                               epsilon_max=rows[-1]['privacy_epsilon_max'],
                               epsilon_mean=rows[-1]['privacy_epsilon_mean'],
                               delta=rows[-1]['privacy_delta']))
        for r in rows:
            assert r['privacy_adjacency'] == 'replace_one'
            assert r['privacy_sampling_scheme'] == 'fixed_without_replacement'
            assert r['privacy_level'] == 'sample' and r['privacy_trust_model'] == 'local'
            assert r['privacy_delta'] == 1e-5
            assert r['rcig_private_gradient_compute_device'].startswith('mps')
            assert r['rcig_server_aggregation_device'] == 'cpu'
            expected_target_clients = 10 if task['scenario'] == 'none' or r['round_num'] < 17 else 8
            assert r['aggregation_role_honest_evaluation_clients'] == expected_target_clients
            for arm in ARMS:
                pre = f'cohort_{arm}_'
                target_error = r[pre + 'aggregate_error_sq']
                terms_sum = math.fsum(r[pre + k] for k in TERMS)
                residual = abs(terms_sum-target_error)
                max_identity_abs = max(max_identity_abs, residual)
                max_identity_relative = max(max_identity_relative, residual/max(1, target_error))
                max_vector_residual = max(max_vector_residual, r[pre+'decomposition_residual_sq'])
                assert residual <= 1e-9*max(1, target_error)
                count_identity += 1
                round_csv.append(dict(task, candidate=arm, round=r['round_num'],
                    aggregate_error_sq=target_error,
                    **{k: r[pre+k] for k in TERMS},
                    identity_absolute_residual=residual,
                    byzantine_mass=r[pre+'byzantine_mass']))
        for window in ('primary', 'pre_attack', 'transition', 'persistent'):
            selected = window_rows(rows, task['scenario'], window)
            for arm in ARMS:
                pre = f'cohort_{arm}_'
                entry = dict(task, host_arm=task['arm'], candidate=arm, window=window,
                             round_start=selected[0]['round_num'], round_end=selected[-1]['round_num'])
                for field in ('aggregate_error_sq', 'byzantine_mass', 'concentration', 'max_weight',
                              'entropy', *TERMS):
                    entry[field] = st.mean(r[pre+field] for r in selected)
                for ref in ('rfa', 'midpoint'):
                    entry[f'reference_{ref}_error_sq'] = st.mean(r[f'cohort_reference_{ref}_error_sq'] for r in selected)
                entry['far_weights_l1_rfa_vs_midpoint'] = st.mean(r['cohort_far_weights_l1_rfa_vs_midpoint'] for r in selected)
                candidate_csv.append(entry)
            if task['arm'] == 'uniform':
                for base in ('far_rfa', 'far_midpoint'):
                    errors = [r['cohort_'+base+'_aggregate_error_sq'] for r in selected]
                    oracle = [min(r['cohort_'+a+'_aggregate_error_sq'] for a in ARMS) for r in selected]
                    p_errors = [r['cohort_reference_midpoint_error_sq'] for r in selected]
                    headroom_csv.append(dict(noise=task['noise'], scenario=task['scenario'],
                        seed=task['seed'], host_arm='uniform', window=window, baseline=base,
                        baseline_error_sq=st.mean(errors),
                        hindsight_best_of_four_error_sq=st.mean(oracle),
                        hindsight_envelope_reduction_pct=100*(1-st.mean(oracle)/st.mean(errors)),
                        midpoint_if_direct_error_sq=st.mean(p_errors),
                        midpoint_endpoint_reduction_pct=100*(1-st.mean(p_errors)/st.mean(errors)),
                        inference='oracle_bound_or_endpoint_not_deployed_radial_algorithm'))

    runner.verify(lock, matrix)
    OUT.mkdir(parents=True, exist_ok=True)
    # Aggregate statistically at the seed level only.
    groups, paired, cohort_groups = {}, {}, {}
    for noise, scenario, arm in itertools.product(NOISE, SCENARIOS, ARMS):
        rs = sorted((r for r in finals if (r['noise'],r['scenario'],r['arm'])==(noise,scenario,arm)), key=lambda r:r['seed'])
        groups[f'{noise}/{scenario}/{arm}'] = {m:stats(r[m] for r in rs) for m in METRICS if m in rs[0]}
    for noise, scenario, candidate, baseline in itertools.product(NOISE, SCENARIOS, ARMS, ARMS):
        if candidate == baseline:
            continue
        key = f'{noise}/{scenario}/{candidate}_minus_{baseline}'
        paired[key] = {}
        for metric in groups[f'{noise}/{scenario}/{candidate}']:
            a = {r['seed']:r[metric] for r in finals if (r['noise'],r['scenario'],r['arm'])==(noise,scenario,candidate)}
            b = {r['seed']:r[metric] for r in finals if (r['noise'],r['scenario'],r['arm'])==(noise,scenario,baseline)}
            paired[key][metric] = stats(a[s]-b[s] for s in SEEDS)
    for noise, scenario, host, window, candidate in itertools.product(NOISE, SCENARIOS, ARMS, ('primary','pre_attack','transition','persistent'), ARMS):
        rs = sorted((r for r in candidate_csv if (r['noise'],r['scenario'],r['host_arm'],r['window'],r['candidate'])==(noise,scenario,host,window,candidate)),key=lambda r:r['seed'])
        key = f'{noise}/{scenario}/host={host}/{window}/{candidate}'
        cohort_groups[key] = {k:stats(r[k] for r in rs) for k in ('aggregate_error_sq','byzantine_mass','reference_rfa_error_sq','reference_midpoint_error_sq','far_weights_l1_rfa_vs_midpoint',*TERMS)}
    cohort_paired = []
    for noise, scenario, host, window, candidate, baseline in itertools.product(NOISE,SCENARIOS,ARMS,('primary','persistent'),ARMS,ARMS):
        if candidate == baseline:
            continue
        a = {r['seed']:r for r in candidate_csv if (r['noise'],r['scenario'],r['host_arm'],r['window'],r['candidate'])==(noise,scenario,host,window,candidate)}
        b = {r['seed']:r for r in candidate_csv if (r['noise'],r['scenario'],r['host_arm'],r['window'],r['candidate'])==(noise,scenario,host,window,baseline)}
        ds = stats(a[s]['aggregate_error_sq']-b[s]['aggregate_error_sq'] for s in SEEDS)
        pct = stats(100*(1-a[s]['aggregate_error_sq']/b[s]['aggregate_error_sq']) for s in SEEDS)
        cohort_paired.append(dict(noise=noise,scenario=scenario,host_arm=host,window=window,
                                 candidate=candidate,baseline=baseline,delta_error_sq=ds,
                                 relative_reduction_pct=pct))
    evidence = dict(created_at=datetime.now(timezone.utc).isoformat(), campaign=str(campaign.relative_to(ROOT)),
        audit=dict(valid_runs=96, completed_statuses=96, rounds=3840, same_cohort_identities=count_identity,
                   private_device='mps', server_postprocessing='cpu_float64', mps_fallback=0,
                   source_lock_matches=True, locked_source_count=len(lock['sources']),
                   all_status_config_metrics_trace_hashes_match=True,
                   paired_client_round_draws=38400, last_completed_at=last_finish,
                   max_scalar_identity_absolute_residual=max_identity_abs,
                   max_scalar_identity_scaled_residual=max_identity_relative,
                   max_vector_identity_residual_sq=max_vector_residual),
        privacy=lock['privacy'], privacy_recomputed_matches_lock=True,
        analysis_script_sha256=sha(Path(__file__)), runs=audit_rows, artifact_sha256=hashes,
        campaign_lock_sha256=sha(lock_path), source_sha256=lock['sources'],
        final_groups=groups, final_paired_differences=paired,
        same_cohort_groups=cohort_groups, same_cohort_paired=cohort_paired,
        protocol_windows=dict(clean_primary='13–40', attacked_primary='17–40', persistent='25–40'),
        epistemic_limitations=[
            'Three seeds; no independent replication from rounds, candidates or host trajectories.',
            'Clean-batch post-clipping gradient target is not the true population/full-data gradient.',
            'In attacked runs the oracle aggregate target uses all10 clients before round17 then fixed8 honest clients; fairness excludes the two designated attackers throughout. Whole40-round target is not a fixed honest objective.',
            'No saved A/P/target vectors or A-minus-P inner products: radial interpolation and cumulative vector bias are not reconstructible.',
            'Reference midpoint considered directly is an offline endpoint diagnostic, not an end-to-end deployed fifth arm.',
            'Best-of-four hindsight envelope is oracle-selected, not a deployable selection policy.',
            'No clean/DP causal attribution: all96 are DP.',
            'No monotone mapping from target MSE to final model accuracy or fairness is assumed.',
        ])
    (OUT/'Evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    csv_dump('final_per_seed.csv',finals)
    csv_dump('same_cohort_per_seed.csv',candidate_csv)
    csv_dump('same_cohort_error_identities.csv',round_csv)
    csv_dump('headroom_uniform_host_per_seed.csv',headroom_csv)
    readable = []
    for key, values in groups.items():
        for metric, s in values.items():
            readable.append(dict(group=key,metric=metric,**{k:v for k,v in s.items() if k!='values'}))
    csv_dump('final_mean_sd.csv',readable)
    write_report(evidence,headroom_csv)
    print(json.dumps(evidence['audit'],indent=2))
    print(str(REPORT))


def write_report(e, headroom):
    g, p, c = e['final_groups'], e['final_paired_differences'], e['same_cohort_groups']
    labels={'uniform':'Uniforme','rfa_direct':'RFA directe','far_rfa':'FAR(RFA)','far_midpoint':'FAR(moyenne temporelle)'}
    nlabels={'homogeneous':'Homogène','heteroscedastic':'Hétéroscédastique'}
    slabels={'none':'Propre','bf_persistent':'BF ×10','ipm_persistent':'IPM','alie_persistent':'ALIE'}
    lines=['# Audit final — aggregation_role_n10_v1','',
           '## 1. Verdict et intégrité','',
           f"**96/96 runs valides et terminés**, dernière fin enregistrée : {e['audit']['last_completed_at']}. Aucun lancement ni modification des résultats pendant cet audit.", '',
           f"{len(e['source_sha256'])} sources verrouillées inchangées ; statuts/configurations/metrics/traces contrôlés par SHA-256 ; 3 840 tours et 15 360 identités de décomposition vérifiés. Les 38 400 enregistrements client–tour ont des tirages appariés. Calcul privé sur MPS, fallback interdit et absent ; post-traitement serveur CPU float64 prévu au protocole.", '',
           f"Résidu scalaire absolu maximal de l'identité des erreurs : {e['audit']['max_scalar_identity_absolute_residual']:.3e} ; norme carrée maximale du résidu vectoriel enregistré : {e['audit']['max_vector_identity_residual_sq']:.3e}.", '',
           '**Conclusion principale :** une référence précise n’implique ni un agrégat précis ni un meilleur modèle. La campagne localise bien une fragilité de FAR sous BF, mais montre également qu’une baisse de l’erreur quadratique au gradient de batch n’est pas, à elle seule, un certificat d’utilité. Aucun agrégateur n’est déclaré universellement meilleur.', '',
           '## 2. Protocole et unité statistique','',
           'Fashion-MNIST, LeNet-5 tanh, 10 clients, Dirichlet par client β=0,1, 6 000 exemples/client ; B=120, un gradient privé par client/tour, T=40, pas serveur 0,2. **Pas de multi-époques.** C=4 par exemple, U=16 au serveur ; distance FAR brute, α=0,1 après warmup uniforme 1–12 commun aux quatre bras. RFA numérique sans bucketing/NNM ; la moyenne temporelle est un contrôle passé, pas RCIG avec détection déployée.', '',
           'Trois seeds 920001/920002/920003. Les tableaux donnent **moyenne ± écart-type d’échantillon** (ddof=1). Pour les mécanismes, moyenne temporelle par seed d’abord, puis moyenne/écart-type entre trois seeds. Les IC95 t descriptifs sont dans Evidence.json ; ils sont larges avec n=3 et ne corrigent pas les comparaisons multiples. Les tours, candidats et trajectoires hôtes ne sont pas de nouvelles réplications.', '',
           'Attaques sur clients 0/1 aux tours 17–40. Fenêtre primaire sans attaque : 13–40 ; avec attaque : 17–40 ; persistance : 25–40. Le jeu global de test est constant. La fairness est évaluée sur 10 clients sans attaque, 8 honnêtes avec attaque dès le premier tour : les différences propre/attaqué de fairness mélangent donc changement du modèle et changement d’ensemble évalué.', '',
           '**Attention : la cible oracle d’agrégat n’a pas la même chronologie que la fairness.** Le code utilise le masque `is_byzantine` actif : 10 clients aux tours 1–16, puis 8 aux tours 17–40. Les différences de MSE entre avant/après attaque mélangent donc également changement de cible et attaque. Les comparaisons appariées à tour/scénario identiques sont valides ; la fenêtre primaire attaquée 17–40 possède un ensemble H fixe de 8 clients. Pour insérer les erreurs dans une preuve sur une loss honnête fixe f_H sur tous les 40 tours, il faut réévaluer avec le même H dès le départ, ou ajouter explicitement un terme de dérive de cible. Le gradient propre de batch clippé reste différent du gradient complet ∇f_H par sampling et biais de clipping.', '',
           'Privacy : sample-level client-side, replace-one, batch fixe sans remise ; ε maximal = 4,0000368051, δ=10⁻⁵. σ=1,4655220866 ; pour les clients à bruit doublé, σ=2,9310441732 et ε=0,9347316947. La privacy est appariée entre méthodes à régime/seed/scénario fixés ; les clients du régime hétéroscédastique n’ont pas tous le même ε. Tous les runs sont DP : aucun résultat causal « effet du bruit » n’est isolé ici.', '',
           '## 3. Résultats finaux du modèle','',
           '| Bruit | Scénario | Règle appliquée | Test Acc. (%) | Client Acc. (%) | Test loss | Variance (pp²) | Worst-20 (%) | Gap B20–W20 (pp) | Balanced Acc. (%) |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for noise,scenario,arm in itertools.product(NOISE,SCENARIOS,ARMS):
        v=g[f'{noise}/{scenario}/{arm}']
        lines.append('| '+' | '.join([nlabels[noise],slabels[scenario],labels[arm],*[fmt(v[m],3 if m=='test_loss' else 2) for m in ['test_accuracy_pct','client_accuracy_pct','test_loss','variance_pp2','worst20_pct','gap_pp','balanced_accuracy_pct']]])+' |')
    lines += ['', '## 4. Comparaisons appariées end-to-end','',
              'Différences candidat − baseline, calculées seed par seed. Le signe positif est favorable pour accuracy/Worst-20, défavorable pour le gap.', '',
              '| Bruit | Scénario | Comparaison | Δ Test Acc. (pp) | Δ Worst-20 (pp) | Δ Gap (pp) |',
              '|---|---|---|---:|---:|---:|']
    for noise,scenario in itertools.product(NOISE,SCENARIOS):
        for candidate,base in [('rfa_direct','uniform'),('far_rfa','uniform'),('far_rfa','rfa_direct'),('far_midpoint','far_rfa')]:
            v=p[f'{noise}/{scenario}/{candidate}_minus_{base}']
            lines.append('| '+' | '.join([nlabels[noise],slabels[scenario],labels[candidate]+' − '+labels[base],*[fmt(v[m]) for m in ('test_accuracy_pct','worst20_pct','gap_pp')]])+' |')
    lines += ['', '## 5. Même cohorte : quelle marge d’amélioration de A ?','',
              'Ancrage préfixé : **trajectoire uniforme**, chaque candidat calculé sur exactement les mêmes messages privés X au même tour. Les colonnes ci-dessous sont des erreurs quadratiques vis-à-vis de la moyenne des gradients honnêtes propres de batch après clipping ; ce ne sont pas des losses du modèle. RFA « référence » est exactement le candidat RFA directe. P est la référence temporelle passée : sa colonne mesure son erreur si l’on l’utilisait directement, mais aucun bras end-to-end ne l’applique seul.', '',
              '| Bruit | Scénario | Uniforme | RFA directe | FAR(RFA) | FAR(P) | P seul, diagnostic | L1 des poids FAR(RFA)/FAR(P) |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for noise,scenario in itertools.product(NOISE,SCENARIOS):
        vs=[c[f'{noise}/{scenario}/host=uniform/primary/{arm}'] for arm in ARMS]
        lines.append('| '+' | '.join([nlabels[noise],slabels[scenario],*[fmt(v['aggregate_error_sq'],4) for v in vs],fmt(vs[0]['reference_midpoint_error_sq'],4),fmt(vs[0]['far_weights_l1_rfa_vs_midpoint'],4)])+' |')
    lines += ['', '### 5.1 Comparaisons fixes et enveloppe oracle','',
              '| Bruit | Scénario | Gain MSE de RFA directe vs FAR(RFA), % | Gain MSE de FAR(P) vs FAR(RFA), % | Enveloppe oracle parmi les 4, % |',
              '|---|---|---:|---:|---:|']
    for noise,scenario in itertools.product(NOISE,SCENARIOS):
        def match(candidate):
            return next(r['relative_reduction_pct'] for r in e['same_cohort_paired'] if (r['noise'],r['scenario'],r['host_arm'],r['window'],r['candidate'],r['baseline'])==(noise,scenario,'uniform','primary',candidate,'far_rfa'))
        h=stats(r['hindsight_envelope_reduction_pct'] for r in headroom if (r['noise'],r['scenario'],r['window'],r['baseline'])==(noise,scenario,'primary','far_rfa'))
        lines.append('| '+' | '.join([nlabels[noise],slabels[scenario],fmt(match('rfa_direct')),fmt(match('far_midpoint')),fmt(h)])+' |')
    lines += ['', 'L’enveloppe choisit après coup le plus petit MSE parmi quatre candidats à chaque tour : c’est un **diagnostic optimiste non déployable**, pas une politique proposée, ni une amélioration d’accuracy. Les résultats sur toutes les trajectoires hôtes, ainsi que la fenêtre persistante 25–40, sont exportés sans choisir l’hôte favorable.', '',
              '### 5.2 Pourquoi cela ne valide pas encore un contrôle radial','',
              'Les scalaires archivés permettent de comparer quatre A et la qualité de P. Ils ne conservent pas les vecteurs A, P, h, ni tous les produits scalaires entre eux. On ne peut donc pas reconstruire l’erreur de toute la courbe P+γ(A−P), le biais vectoriel cumulé, la covariance effective ou la quantité de signal retiré. Il faut instrumenter un replay hors mécanisme ou une nouvelle campagne distincte pour ces mesures. Aucune nouvelle garantie de contrôle radial n’est déduite de ces 96 runs.', '',
              '## 6. Décomposition exacte de l’erreur appliquée','',
              'Pour h=moyenne des g_i honnêtes après clipping, les diagnostics satisfont exactement :', '',
              '```text',
              'A − h = N + H + B',
              'N = Σ_{i honnêtes} λ_i (X_i − g_i)',
              'H = Σ_{i honnêtes} λ_i (g_i − h)',
              'B = Σ_{i byzantins} λ_i (X_i − h)',
              '||A−h||² = ||N||² + ||H||² + ||B||²',
              '           + 2<N,H> + 2<N,B> + 2<H,B>.',
              '```', '',
              'N comprend l’effet du clipping serveur : ce n’est pas une gaussienne centrée par définition. H n’est pas uniquement une variance statistique et B n’est pas la masse byzantine. Les coefficients de RFA directe sont ceux de la dernière itération de Weiszfeld, non une softmax. Les termes croisés peuvent être négatifs, donc supprimer B n’implique pas mécaniquement soustraire ||B||² à l’erreur.', '',
              '| Bruit | Attaque | Candidat sur hôte uniforme | Bruit : N² | Repondération : H² | Byzantine : B² | Croisé N,H | Croisé N,B | Croisé H,B | Total |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for noise,scenario,arm in itertools.product(NOISE,SCENARIOS,ARMS):
        v=c[f'{noise}/{scenario}/host=uniform/primary/{arm}']
        lines.append('| '+' | '.join([nlabels[noise],slabels[scenario],labels[arm],*[f"{v[k]['mean']:.5f}" for k in TERMS],f"{v['aggregate_error_sq']['mean']:.5f}"])+ ' |')
    lines += ['', '## 7. Interprétation : acquis, contradictions et non-identifiabilité','',
              '**Observation — BF homogène.** La robustesse de RFA utilisée directement est perdue en partie lorsqu’elle sert seulement de référence à un FAR positif : RFA directe augmente l’accuracy par rapport à FAR(RFA), réduit le MSE sur la même cohorte et donne moins de masse aux deux attaquants. C’est une localisation concrète du problème à l’étape référence → poids → agrégat dans ce régime.', '',
              '**Observation — IPM.** RFA directe a un MSE de batch beaucoup plus faible mais une accuracy finale plus faible, particulièrement sous bruit hétéroscédastique. Le MSE enregistré est donc insuffisant pour classer l’utilité. La cible est un gradient de batch clippé, pas ∇f exact ; l’orientation et l’accumulation du biais, le pas et le bruit influencent les trajectoires. Ces facteurs restent des explications possibles, pas une attribution causale identifiée.', '',
              '**Décomposition observée sous IPM.** Sur l’hôte uniforme hétéroscédastique, pour RFA directe : N²=11,9230, B²=3,9270 et le terme croisé 2⟨N,B⟩=−12,7602 ; le total est seulement 3,1229. La petite erreur résulte donc en grande partie d’une annulation entre contribution byzantine et perturbation honnête, pas simplement d’une petite contribution byzantine. Ce constat algébrique ne suffit pas à expliquer causalement la différence d’accuracy, mais interdit de lire le MSE seul comme un certificat de robustesse.', '',
              '**Observation — ALIE.** RFA directe attribue pratiquement toute sa masse aux deux messages byzantins identiques. Le point choisi peut néanmoins avoir un MSE voisin d’une moyenne dans certains régimes : masse byzantine, distance à la cible et accuracy sont trois diagnostics différents. On ne peut inférer une robustesse générale de la seule accuracy finale.', '',
              '**Observation — propre hétéroscédastique.** FAR(RFA) et FAR(P) améliorent ici accuracy et Worst-20 par rapport à l’uniforme, tout en augmentant le MSE de batch. Cela contredit l’idée d’imposer systématiquement une réduction du MSE de batch comme substitut exact de l’amélioration du modèle.', '',
              '**Inférence défendable.** Il est légitime d’étudier une correction appliquée à A plutôt qu’à F seule, mais la correction doit montrer simultanément une erreur appliquée utile, un coût honnête acceptable et des résultats end-to-end. Un meilleur P seul est une piste instrumentale, pas une marge démontrée pour la politique complète.', '',
              '**Non identifiable.** Supériorité générale de RCIG ; robustesse d’une ellipsoïde ; amélioration de convergence par réduction du MSE de batch ; contribution propre du bruit DP ; récupération après arrêt d’attaque ; covariance directionnelle effective et biais cumulé. Aucun de ces points n’est établi par cette matrice.', '',
              '## 8. Suite minimale, distincte des résultats actuels','',
              '1. Fixer le gradient cible (batch honnête clippé pour l’audit, ∇f pour la preuve) et enregistrer les écarts entre les deux ; ne pas les confondre.',
              '2. Sur cohortes strictement identiques, comparer FAR intact, lissage simple d’A, clipping isotrope d’A et contrôle statistique d’A ; inclure RFA directe comme baseline. Geler les seuils sur seeds hors test.',
              '3. Distinguer la référence FAR (CM/RFA/trMean, comparaisons séparées) du prédicteur passé P. RCIG peut devenir un prédicteur candidat ; il ne doit pas être supposé obligatoire ni déjà meilleur qu’un lissage simple.',
              '4. Mesurer le coût honnête, les fausses corrections, les produits scalaires d’erreur, le biais cumulé et les attaques progressives/persistantes ; ajouter une phase de récupération dans une nouvelle matrice.',
              '5. Exiger une non-infériorité propre et un gain end-to-end répliqué ; si les gains de mécanisme n’atteignent pas le modèle, ne pas promouvoir la variante.', '',
              'Aucun run de ces nouvelles variantes n’a été lancé par ce script d’analyse.', '',
              '## 9. Sources locales reproductibles','',
              '- [Protocole verrouillé](LDP_Aggregation_Role_N10_Protocol.md)',
              '- [Matrice](../../configs/ldp_gradient_far/aggregation_role_n10_v1.yaml)',
              '- [Preuves machine et statistiques](aggregation_role_n10_final/Evidence.json)',
              '- [Métriques finales par seed](aggregation_role_n10_final/final_per_seed.csv)',
              '- [Moyennes et écarts-types](aggregation_role_n10_final/final_mean_sd.csv)',
              '- [Candidats appariés par seed, hôte et fenêtre](aggregation_role_n10_final/same_cohort_per_seed.csv)',
              '- [15 360 identités d’erreur](aggregation_role_n10_final/same_cohort_error_identities.csv)',
              '- [Marge oracle et endpoint temporel](aggregation_role_n10_final/headroom_uniform_host_per_seed.csv)',
              '- [Script reproductible](../../scripts/analyze_ldp_aggregation_role_final.py)', '']
    REPORT.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
