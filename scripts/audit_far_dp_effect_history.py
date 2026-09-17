#!/usr/bin/env python3
"""Read saved positioning runs, freeze evidence; no training and no old edits."""
import hashlib
import json
from pathlib import Path
import statistics
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/analysis/FAR_DP_Effect_Historical_Audit_20260917.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    folder = ROOT/'results/ldp_gradient_far/positioning_v3/e_byzantine_identification_screen'
    evidence = []
    for arm in ['nodp_uniform', 'nodp_fcc_ap2', 'dp_uniform', 'dp_fcc_ap2']:
        parent = folder/f'n10_c8__none__{arm}_seed137'
        config_path = parent/'resolved_config.yaml'
        metrics_paths = list(parent.rglob('metrics.json'))
        assert len(metrics_paths) == 1
        p = metrics_paths[0]
        cfg = yaml.safe_load(config_path.read_text())
        m = json.loads(p.read_text()); rows = m['rounds']; a = cfg['training']['algo_config']
        assert [r['round_num'] for r in rows] == list(range(1, 21))
        assert all(not r.get('attack_enabled') and r['num_survivors'] == 10 for r in rows)
        counts = {}
        for field in ['far_server_clip_rate', 'privacy_clip_rate_mean']:
            values = [r[field] for r in rows if isinstance(r.get(field), (float, int))]
            counts[field] = dict(recorded=len(values), min=min(values) if values else None,
                                 max=max(values) if values else None)
        final = rows[-1]
        evidence.append(dict(arm=arm, seed=137, dataset=m['dataset'],
            config_source=str(config_path.relative_to(ROOT)), config_sha256=sha(config_path),
            metrics_source=str(p.relative_to(ROOT)), metrics_sha256=sha(p),
            local_clip=a['clip_norm'], server_clip=a['far_server_clip_norm'],
            local_epochs_field=a['local_epochs'], local_optimizer_steps_actual=0,
            releases_per_round=a['fixed_steps_per_round'], rounds=len(rows),
            batch=a['fixed_batch_size'], N=a['privacy_public_dataset_size'],
            expected_exposure_passes=len(rows)*a['fixed_batch_size']/a['privacy_public_dataset_size'],
            expected_distinct_fraction=1-(1-a['fixed_batch_size']/a['privacy_public_dataset_size'])**len(rows),
            clipping=counts, eta=a['far_server_lr'], alpha=a['far_alpha'],
            epsilon_bound=final.get('privacy_epsilon_max'),
            final=dict(accuracy=100*final['test_accuracy'], worst20=final['worst20_accuracy_pct'],
                       gap=final['best20_worst20_gap_pct'], variance=final['client_accuracy_variance_pct2'],
                       loss=final['test_loss']),
            accuracy_gain_last_five_rounds_pp=100*(final['test_accuracy']-rows[14]['test_accuracy']),
            max_recorded_server_clipping=max(r['far_server_clip_rate'] for r in rows)))
    sources = ['algorithms/ldp_gradient_far.py', 'algorithms/dp_references.py',
               'privacy/local_dpsgd.py', 'core/seeding.py', 'run_experiment.py',
               'scripts/run_ldp_gradient_far_positioning.py', 'privacy/rdp.py',
               'notebooks/LDP_Gradient_FAR_Positioning_Analysis_Editable.ipynb']
    report = dict(scope='four saved E controls, n=10, seed137; not every historical campaign',
                  observations=evidence, sources_sha256={p:sha(ROOT/p) for p in sources},
                  no_new_training=True, historical_files_modified=False,
                  strict_pairing_of_all_historical_rng_draws='not certified by config equality',
                  scientific_conclusion='short horizon; server clipping configured but inactive in these controls')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['observations'], indent=2))


if __name__ == '__main__':
    main()
