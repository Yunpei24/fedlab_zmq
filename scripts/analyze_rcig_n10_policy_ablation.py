#!/usr/bin/env python3
"""Read-only audit/reduction of the completed N10 pilot; never trains or promotes.

Writes derived JSON, Markdown tables and figures outside the source-locked runs.
Seed is the replication unit; time points and arms are not independent seeds.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import yaml
from scripts import run_rcig_n10_policy_ablation as run

DEST = ROOT / "output/analysis/rcig_n10_policy_ablation_final"
SUFFIX = "_squared_l2_error_to_clean_honest_center_oracle"
NOISE = {"homogeneous": "Homogène", "heteroscedastic": "Hétéroscédastique"}
SCENARIO = {"none": "Sans attaque", "bf_persistent": "BF ×10", "ipm_persistent": "IPM", "alie_persistent": "ALIE"}
ARMS = {"recent": "Récente", "rolling": "RCIG sans gel", "freeze": "RCIG avec gel", "midpoint": "Moyenne simple", "rfa": "FAR + RFA"}
METRICS = {
    "test_acc": ("test_accuracy", 100),
    "client_acc": ("client_accuracy_mean", 100),
    "test_loss": ("test_loss", 1),
    "variance": ("client_accuracy_variance_pct2", 1),
    "worst20": ("worst20_accuracy_pct", 1),
    "gap": ("best20_worst20_gap_pct", 1),
    "balanced_acc": ("mean_client_balanced_accuracy_pct", 1),
    "balanced_worst20": ("worst20_balanced_accuracy_pct", 1),
    "client_loss": ("client_loss_mean", 1),
}
WINDOWS = {"clean_ready": (13, 16), "attacked_all": (17, 40), "transition": (18, 24), "persistent": (25, 40), "ready_all": (13, 40)}


def stats(values):
    a = list(values)
    assert a and all(math.isfinite(x) for x in a)
    return dict(n=len(a), mean=st.mean(a), sd=st.stdev(a) if len(a) > 1 else None, min=min(a), max=max(a))


def fmt(values, digits=2):
    s = stats(values)
    return f"{s['mean']:.{digits}f} ± {s['sd']:.{digits}f}" if s['sd'] is not None else f"{s['mean']:.{digits}f}"


def select(rows, window):
    a, b = WINDOWS[window]
    return [r for r in rows if a <= r['round_num'] <= b]


def window_stats(rows, threshold):
    d = {
        "mean_logit_span": st.mean(r['far_logit_range'] for r in rows),
        "mean_max_weight": st.mean(r['far_max_weight'] for r in rows),
        "mean_concentration": st.mean(10*r['far_weight_l2_squared'] for r in rows),
        "mean_entropy": st.mean(r['weight_entropy'] for r in rows),
        "server_clip_fraction": st.mean(r['far_server_clip_rate'] for r in rows),
        "byzantine_mass": st.mean(r['byzantine_weight_mass_oracle'] for r in rows),
        "weight_cap_violations": sum(not r['far_weight_cap_respected'] for r in rows),
    }
    if not rows[0].get('rcig_ready', False):
        return d
    for name in ['reference', 'identity_new', 'identity_old', 'midpoint', 'full', 'isotropic', 'euclidean']:
        d['error_'+name] = st.mean(r['rcig_'+name+SUFFIX] for r in rows)
    d.update(
        alarm_fraction=st.mean(bool(r['rcig_gate_active']) for r in rows),
        frozen_fraction=st.mean(bool(r['rcig_persistent_frozen_after_commit']) for r in rows),
        r_over_c=st.mean(r['rcig_full_innovation_stat']/threshold for r in rows),
        r_over_c_max=max(r['rcig_full_innovation_stat']/threshold for r in rows),
        full_trust=st.mean(r['rcig_full_newer_trust'] for r in rows),
        covariance_anisotropy=max(r['rcig_max_covariance_anisotropy_ratio'] for r in rows),
        relative_stat_full_iso_max=max(abs(r['rcig_full_innovation_stat']-r['rcig_isotropic_innovation_stat'])/r['rcig_full_innovation_stat'] for r in rows),
        midpoint_beats_full=sum(r['rcig_midpoint'+SUFFIX] < r['rcig_full'+SUFFIX] - 1e-12 for r in rows),
        n_rounds=len(rows),
    )
    return d


def main():
    m = run.matrix()
    stamp = json.loads((run.OUTPUT/'campaign_lock.json').read_text())
    threshold_doc = json.loads(run.ARTIFACT.read_text())
    run.verify_sources(stamp)
    hashes = {}
    records, raw = [], {}
    pairing = []
    finished = []
    calibration = defaultdict(lambda: defaultdict(list))
    for i, task in enumerate(run.tasks(m), 1):
        folder = run.directory(task)
        status = json.loads((folder/'orchestration_status.json').read_text())
        assert status['status'] == 'completed', folder
        assert status['device'] == 'mps' and status['fallback'] == 0
        thresholds = threshold_doc['thresholds'][task['noise']] if task['phase'] == 'comparison' else None
        cfg = run.config_for(m, task, stamp, thresholds)
        assert yaml.safe_load((folder/'resolved_config.yaml').read_text()) == cfg
        assert run.hash_file(folder/'resolved_config.yaml') == status['config_sha256']
        path, payload, paired = run.validate(task, cfg, stamp)
        assert run.hash_file(path) == status['metrics_sha256']
        assert run.hash_file(folder/'simulator_randomness_private_audit.jsonl') == status['trace_sha256']
        for p in [path, folder/'orchestration_status.json', folder/'resolved_config.yaml', folder/'simulator_randomness_private_audit.jsonl', folder/'runtime_imports.json']:
            hashes[str(p.relative_to(ROOT))] = run.hash_file(p)
        pairing.append(paired)
        finished.append(status['finished_at'])
        rows = payload['rounds']
        assert len(rows) == 40
        # Independent cross-check of fairness formulas against per-client values.
        for row in rows:
            acc = row['client_accuracy_values_oracle']
            k = math.ceil(.2*len(acc))
            assert len(acc) == (10 if task['scenario'] == 'none' else 8)
            expected = {
                'client_accuracy_mean': st.mean(acc),
                'client_accuracy_variance_pct2': st.pvariance(acc)*10000,
                'worst20_accuracy_pct': st.mean(sorted(acc)[:k])*100,
                'best20_worst20_gap_pct': (st.mean(sorted(acc)[-k:])-st.mean(sorted(acc)[:k]))*100,
            }
            for key, val in expected.items():
                assert abs(row[key]-val) < 1e-8, (path, key)
        if task['phase'] == 'calibration':
            for mode in run.MODES:
                calibration[task['noise']][mode].append(max(r[f'rcig_{mode}_innovation_stat'] for r in rows[12:]))
        else:
            rec = dict(task, metrics_path=str(path.relative_to(ROOT)))
            rec['final'] = {name: rows[-1][key]*scale for name, (key, scale) in METRICS.items()}
            rec['windows'] = {name: window_stats(select(rows, name), thresholds['full']) for name in WINDOWS}
            rec['alarm_rounds'] = [r['round_num'] for r in rows[12:] if r.get('rcig_gate_active')]
            rec['frozen_rounds'] = [r['round_num'] for r in rows[12:] if r.get('rcig_persistent_frozen_after_commit')]
            rec['freeze_events'] = [r['round_num'] for r in rows if r.get('rcig_persistent_action') == 'freeze_on_activation']
            rec['persistent_actions'] = [[r['round_num'],r.get('rcig_persistent_action')] for r in rows[12:]]
            rec['max_logged_frozen_age'] = max(r.get('rcig_n10_reference_age_rounds',0) for r in rows)
            rec['privacy'] = {key:rows[-1][key] for key in ['privacy_epsilon_max','privacy_epsilon_mean','privacy_delta','privacy_model_noise_multiplier_min','privacy_model_noise_multiplier_max']}
            records.append(rec)
            raw[(task['noise'],task['scenario'],task['arm'],task['seed'])] = rows
        if i % 24 == 0:
            print(f'Audit {i}/132', flush=True)
    for noise, modes in calibration.items():
        for mode, values in modes.items():
            assert values == threshold_doc['seed_maxima'][noise][mode]
            assert max(values) == threshold_doc['thresholds'][noise][mode]
    for source, digest in threshold_doc['sources'].items():
        assert run.hash_file(ROOT/source) == digest
    assert len(records) == 120
    groups = defaultdict(list)
    for rec in records:
        groups[(rec['noise'],rec['scenario'],rec['arm'])].append(rec)
    assert len(groups) == 40 and all(len(g)==3 for g in groups.values())
    # Check independently recomputed numbers against the launcher's summary.
    previous = json.loads((run.OUTPUT/'summary_progress.json').read_text())['groups']
    for group, recs in groups.items():
        for alias,(key,scale) in METRICS.items():
            s = previous['/'.join(group)]['mean_sd'].get(key)
            if s:
                fresh=stats(r['final'][alias]/scale for r in recs)
                assert abs(fresh['mean']-s['mean']) < 1e-10
                assert abs(fresh['sd']-s['sd']) < 1e-10
    deltas=[]
    for rec in records:
        for control in ['recent','midpoint','rfa']:
            if rec['arm']==control:
                continue
            base=next(r for r in groups[(rec['noise'],rec['scenario'],control)] if r['seed']==rec['seed'])
            deltas.append({**{k:rec[k] for k in ['noise','scenario','seed','arm']},'control':control,
                           'delta':{metric:rec['final'][metric]-base['final'][metric] for metric in METRICS}})
    # Local diagnostics: same transcript, not different end-to-end trajectories.
    counterfactual=[]
    for rec in records:
        if rec['arm']!='recent':
            continue
        for window in ['ready_all'] if rec['scenario']=='none' else ['attacked_all','transition','persistent']:
            a=rec['windows'][window]
            counterfactual.append({**{k:rec[k] for k in ['noise','scenario','seed']},'window':window,
                'full_gain_pct':100*(1-a['error_full']/a['error_identity_new']),
                'midpoint_gain_pct':100*(1-a['error_midpoint']/a['error_identity_new']),
                'midpoint_beats_full_rounds':a['midpoint_beats_full'],'n_rounds':a['n_rounds']})
    DEST.mkdir(parents=True,exist_ok=True)
    evidence=dict(created_at=datetime.now(timezone.utc).isoformat(),validation=dict(
        valid_runs=132,calibration_runs=12,comparison_runs=120,rounds=5280,
        private_client_round_traces=sum(p['actual_draws_verified'] for p in pairing),
        paired_attack_clean_prefixes=sum(p.get('same_arm_clean_prefix') is True for p in pairing),
        last_finished_at=max(finished),source_files=len(stamp['sources']),
        config_metrics_trace_hashes_verified=True,fairness_recomputed=True,
        thresholds_recomputed=True,privacy_device_and_runtime_audited=True),
        privacy=stamp['privacy'],thresholds=threshold_doc['thresholds'],
        records=records,paired_deltas=deltas,counterfactual=counterfactual,source_sha256=hashes,
        interpretation='exploratory_no_promotion_no_new_training')
    (DEST/'Evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# RCIG N10 — tableaux complets vérifiés','',
        '132/132 runs valides. Moyenne ± écart-type inter-seeds (n = 3, ddof = 1). Les écarts appariés sont calculés seed par seed avant agrégation. Ni les tours ni les bras ne sont des répétitions indépendantes.','',
        'Le gap désigne Best-20 − Worst-20, pas max − min. Variance : dispersion entre clients en pp². Les barres ± des figures sont des écarts-types, pas des IC95.','']
    for noise in NOISE:
        for scenario in SCENARIO:
            lines += [f'## {NOISE[noise]} — {SCENARIO[scenario]}','',
                '| Référence | Test Acc. (%) | Client Acc. (%) | Test loss | Variance (pp²) | Worst-20 (%) | Gap (pp) |',
                '|---|---:|---:|---:|---:|---:|---:|']
            for arm in ARMS:
                g=groups[noise,scenario,arm]
                lines.append('| '+ARMS[arm]+' | '+' | '.join(fmt(r['final'][k] for r in g) for k in ['test_acc','client_acc','test_loss','variance','worst20','gap'])+' |')
            lines += ['', '| Référence | Balanced accuracy cliente (%) | Worst-20 balanced (%) | Loss cliente |','|---|---:|---:|---:|']
            for arm in ARMS:
                g=groups[noise,scenario,arm]
                lines.append('| '+ARMS[arm]+' | '+' | '.join(fmt(r['final'][k] for r in g) for k in ['balanced_acc','balanced_worst20','client_loss'])+' |')
            for control in ['recent','midpoint','rfa']:
                lines += ['',f'### Différences appariées : bras − {ARMS[control]}','',
                    '| Bras | Δ Test Acc. (pp) | Δ Worst-20 (pp) | Δ variance (pp²) | Δ gap (pp) |','|---|---:|---:|---:|---:|']
                for arm in ['rolling','freeze']:
                    ds=[d for d in deltas if (d['noise'],d['scenario'],d['arm'],d['control'])==(noise,scenario,arm,control)]
                    lines.append('| '+ARMS[arm]+' | '+' | '.join(fmt((d['delta'][k] for d in ds),3) for k in ['test_acc','worst20','variance','gap'])+' |')
    lines += ['','## Mécanisme : fenêtres préspécifiées','',
        'Erreur = distance euclidienne au carré de la référence déployée au centre honnête propre clippé, évalué hors mécanisme. Ce n’est ni la loss, ni une MSE divisée par la dimension. Non instrumentée pour le bras RFA.','']
    for window in ['ready_all','transition','persistent']:
        lines += [f'### {window} : tours {WINDOWS[window][0]}–{WINDOWS[window][1]}','',
            '| Bruit | Scénario | Référence | Erreur déployée | Alerte (%) | Gel (%) | r/c |','|---|---|---|---:|---:|---:|---:|']
        for group,g in groups.items():
            noise,scenario,arm=group
            if arm=='rfa' or (window=='ready_all') != (scenario=='none'):
                continue
            vals=[r['windows'][window] for r in g]
            lines.append('| '+NOISE[noise]+' | '+SCENARIO[scenario]+' | '+ARMS[arm]+' | '+
                fmt((v['error_reference'] for v in vals),4)+' | '+fmt((v['alarm_fraction']*100 for v in vals),2)+' | '+
                fmt((v['frozen_fraction']*100 for v in vals),2)+' | '+fmt((v['r_over_c'] for v in vals),4)+' |')
    lines += ['','## Poids et clipping, tours 17–40 (13–40 sans attaque)','',
        '| Bruit | Scénario | Référence | Masse byzantine | Poids max | 10 Σλ² | Logit-span | Clipping serveur (%) |','|---|---|---|---:|---:|---:|---:|---:|']
    for (noise,scenario,arm),g in groups.items():
        w='ready_all' if scenario=='none' else 'attacked_all'
        vals=[r['windows'][w] for r in g]
        lines.append('| '+NOISE[noise]+' | '+SCENARIO[scenario]+' | '+ARMS[arm]+' | '+
            ' | '.join(fmt((v[k] for v in vals),4) for k in ['byzantine_mass','mean_max_weight','mean_concentration','mean_logit_span'])+' | '+
            fmt((v['server_clip_fraction']*100 for v in vals),2)+' |')
    lines += ['','## Résultats de chaque seed','',
        '| Bruit | Scénario | Référence | Seed | Test Acc. (%) | Client Acc. (%) | Test loss | Variance (pp²) | Worst-20 (%) | Gap (pp) |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for rec in records:
        lines.append('| '+NOISE[rec['noise']]+' | '+SCENARIO[rec['scenario']]+' | '+ARMS[rec['arm']]+' | '+str(rec['seed'])+' | '+
            ' | '.join(f"{rec['final'][k]:.4f}" for k in ['test_acc','client_acc','test_loss','variance','worst20','gap'])+' |')
    (DEST/'Tables.md').write_text('\n'.join(lines)+'\n')
    figures(groups,raw,threshold_doc)
    # All source observations must remain byte-for-byte unchanged.
    run.verify_sources(stamp)
    assert all(run.hash_file(ROOT/p)==h for p,h in hashes.items())
    print(json.dumps(evidence['validation'],indent=2))


def figures(groups, raw, thresholds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors=['#637586','#168c85','#d46f25','#9464b4','#c44569']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    scenarios=list(SCENARIO)
    for metric,title in [('test_acc','Accuracy test (%) — plus haut = mieux'),('worst20','Worst-20 (%) — plus haut = mieux')]:
        fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),sharey=True)
        for ax,noise in zip(axes,NOISE):
            for j,arm in enumerate(ARMS):
                vals=[stats(r['final'][metric] for r in groups[noise,s,arm]) for s in scenarios]
                ax.errorbar(np.arange(4)+(j-2)*.13,[v['mean'] for v in vals],yerr=[v['sd'] for v in vals],fmt='o',capsize=3,color=colors[j],label=ARMS[arm])
            ax.set_xticks(range(4),[SCENARIO[s] for s in scenarios])
            ax.set_title(NOISE[noise]);ax.grid(axis='y',alpha=.25)
        axes[0].set_ylabel(title)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False)
        fig.suptitle('Tour 40 · 3 seeds · moyenne ± écart-type (pas IC95)')
        fig.tight_layout(rect=(0,.1,1,.93));fig.savefig(DEST/f'{metric}.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),sharey=True)
    for ax,noise in zip(axes,NOISE):
        for j,arm in enumerate(list(ARMS)[:4]):
            vals=[stats(r['windows']['ready_all' if s=='none' else 'attacked_all']['error_reference'] for r in groups[noise,s,arm]) for s in scenarios]
            ax.errorbar(np.arange(4)+(j-1.5)*.16,[v['mean'] for v in vals],yerr=[v['sd'] for v in vals],fmt='o',capsize=3,color=colors[j],label=ARMS[arm])
        ax.set_title(NOISE[noise]);ax.set_xticks(range(4),[SCENARIO[s] for s in scenarios]);ax.grid(axis='y',alpha=.25)
    axes[0].set_ylabel('Erreur quadratique de référence — plus bas = mieux')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Référence réellement déployée · tours 17–40 (13–40 sans attaque)')
    fig.tight_layout(rect=(0,.1,1,.93));fig.savefig(DEST/'reference_error.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,scenario in zip(axes.flat,scenarios):
        for noise,color in zip(NOISE,['#168c85','#d46f25']):
            rs=[raw[noise,scenario,'recent',seed][12:] for seed in [920001,920002,920003]]
            arr=np.array([[r['rcig_full_innovation_stat']/thresholds['thresholds'][noise]['full'] for r in rows] for rows in rs])
            ts=np.arange(13,41);avg=arr.mean(0);sd=arr.std(0,ddof=1)
            ax.plot(ts,avg,color=color,label=NOISE[noise]);ax.fill_between(ts,avg-sd,avg+sd,color=color,alpha=.13)
        ax.axhline(1,color='black',ls='--',lw=1,label='Seuil d’alerte')
        if scenario != 'none':
            ax.axvline(17,color='gray',ls=':',lw=1)
        ax.set_title(SCENARIO[scenario]);ax.grid(alpha=.2);ax.set_ylabel('Statistique / seuil');ax.set_xlabel('Tour')
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Diagnostic sur le bras récent · moyenne ± écart-type entre 3 seeds')
    fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(DEST/'alarm_statistic.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    main()
