"""Conditional V30 grid and exact-count criteria; this module never starts training.

Threat generation reuses private_risk_attacks.inject, not an oracle-gradient attack.
The statistical comparison reuses the frozen V29 rule; an eventual final auditor
must rederive it independently. Passing unit tests is not empirical validation.
"""
from fractions import Fraction as Q
import math
from privacy.full_population_private_risk_confirmation_v29 import SEEDS, METHODS, KEYS, compare

CONDITIONS = ('none', 'abrupt_bf', 'persistent_alie', 'slow_ipm')
ENDPOINTS = dict(abrupt_bf=90, persistent_alie=120, slow_ipm=90)
CONTROLS = ('erm_mean', 'erm_rfa')
HONEST_IDS = tuple(range(2, 10))


def jobs():
    # Complete the four matched clean replays before attacks for each seed.
    return [dict(seed=s, method=m, attack=a) for s in SEEDS for a in CONDITIONS for m in METHODS]


def require_clean_pass(audit, status, manifest_sha256):
    """Reject partial, negative, stale-manifest or runner-only clean evidence."""
    d = audit.get('decision') or {}
    required = (
        audit.get('audit_passed') is True,
        audit.get('gate_evaluated') is True,
        audit.get('expected_runs') == 24,
        audit.get('valid_runs') == 24,
        audit.get('manifest_sha256') == manifest_sha256,
        d.get('clean_confirmation_passed') is True,
        d.get('primary_method') == 'risk_rfa',
        d.get('primary_batch') == 4800,
        d.get('wave') == 1,
        d.get('one_sided_alpha') == .0125,
        status.get('status') == 'completed',
        status.get('device') == 'mps',
        status.get('valid_runs') == 24,
    )
    if not all(required):
        raise RuntimeError('V30 closed: all 24 V29 runs and a positive independent final audit are required')
    expected = {(s, b, m) for s in SEEDS for b, m in (
        (4800, 'erm_mean'), (4800, 'erm_rfa'), (4800, 'risk_mean'),
        (4800, 'risk_rfa'), (240, 'erm_mean'), (240, 'erm_rfa'))}
    records = audit.get('records', [])
    found = {(r['job']['seed'], r['job']['batch'], r['job']['method']) for r in records}
    if len(records) != 24 or found != expected:
        raise RuntimeError('V30 closed: independent audit does not contain the exact V29 grid')
    contrasts = d.get('contrasts', [])
    controls = {(c['control_batch'], c['control_method']) for c in contrasts}
    gate_names = {'all_seed_gates', 'worst20_lower_positive', 'accuracy_lower_noninferior',
                  'mean_gap_nonincreasing', 'mean_variance_nonincreasing'}
    if (len(contrasts) != 4 or controls != {(4800, m) for m in CONTROLS} | {(240, m) for m in CONTROLS}
            or any(c.get('passed') is not True or set(c.get('gates', {})) != gate_names
                   or not all(v is True for v in c['gates'].values()) for c in contrasts)):
        raise RuntimeError('V30 closed: every prespecified clean comparison must pass')


def exact_honest_test(full):
    """Eight fixed identities, exact integer counts, two clients in each tail.

    All ten client records remain present to prevent relabelling/dropout selection.
    The full-test metrics need not equal honest-only metrics under attack.
    """
    clients = full['clients']
    if len(clients) != 10:
        raise ValueError('Exactly ten ordered test partitions required')
    acc = []
    for cid, c in enumerate(clients):
        ns, hs = c['class_count'], c['class_hits']
        if (c['N'] != 1000 or len(ns) != 10 or len(hs) != 10 or sum(ns) != 1000
                or any(not math.isfinite(x) or x < 0 or x != int(x) for x in ns+hs)
                or any(h > n for h, n in zip(hs, ns))):
            raise ValueError('Invalid fixed test counts')
        if cid in HONEST_IDS:
            acc.append(Q(int(sum(hs)), 10))
    mean = sum(acc)/8
    ordered = sorted(acc)
    return dict(accuracy_pct=mean, worst20_pct=sum(ordered[:2])/2,
        gap_best20_worst20_pp=sum(ordered[-2:])/2-sum(ordered[:2])/2,
        variance_pp2=sum((a-mean)**2 for a in acc)/8)


def decide(records):
    """All attacks AND matched damage/recovery limits; never select a best attack."""
    expected = {(j['seed'], j['method'], j['attack']) for j in jobs()}
    if len(records) != 64:
        raise ValueError('No attack gate before all 64 unique clean/attack runs')
    index, solvers = {}, {}
    for record in records:
        j = record['job']; key = (j['seed'], j['method'], j['attack'])
        if key in index or key not in expected or record.get('device') != 'mps':
            raise ValueError('Duplicate, unknown or non-MPS run')
        rows = record['rounds']
        if ([r['round'] for r in rows] != list(range(1, 121))
                or record['test_evaluation_rounds'] != [90, 120]
                or [r['round'] for r in rows if r['test'] is not None] != [90, 120]):
            raise ValueError('Missing fixed trajectory or matched endpoints')
        index[key] = {r['round']: exact_honest_test(r['test']) for r in rows if r['test'] is not None}
        if j['method'].endswith('_rfa'):
            gaps = [r['aggregation']['solver']['unsmoothed_objective_gap_upper'] for r in rows]
            if any(not math.isfinite(g) or g < 0 for g in gaps):
                raise ValueError('Invalid numerical solver diagnostic')
            solvers[key] = max(gaps)
    if set(index) != expected:
        raise ValueError('Incomplete fixed grid')
    results = {}
    for attack, t in ENDPOINTS.items():
        comparisons = {}
        for control in (*CONTROLS, 'risk_mean'):
            pairs = [{k: index[s, 'risk_rfa', attack][t][k]-index[s, control, attack][t][k]
                      for k in KEYS} for s in SEEDS]
            comparisons[control] = dict(primary=control in CONTROLS,
                pairs=[dict(seed=s, delta={k: float(v) for k, v in p.items()}) for s, p in zip(SEEDS, pairs)],
                **compare(pairs))
        clean = []
        for s in SEEDS:
            for endpoint in sorted({t, 120}):
                a = index[s, 'risk_rfa', attack][endpoint]
                b = index[s, 'risk_rfa', 'none'][endpoint]
                da, dw = a['accuracy_pct']-b['accuracy_pct'], a['worst20_pct']-b['worst20_pct']
                clean.append(dict(seed=s, round=endpoint, accuracy_delta_pp=float(da),
                    worst20_delta_pp=float(dw), passed=da >= -5 and dw >= -5))
        solver_max = max(solvers[s, 'risk_rfa', attack] for s in SEEDS)
        passed = (all(comparisons[c]['passed'] for c in CONTROLS)
                  and all(c['passed'] for c in clean) and solver_max <= .001)
        results[attack] = dict(primary_round=t, comparisons=comparisons,
            own_clean_comparisons=clean, solver_max_gap=solver_max,
            solver_gate=solver_max <= .001, passed=passed)
    # Control solver validity is checked too; do not benefit from a poor baseline solve.
    all_solver_max = max(solvers.values())
    return dict(attacks=results, all_rfa_solver_max_gap=all_solver_max,
        all_rfa_solver_gate=all_solver_max <= .001,
        attack_confirmation_passed=all(r['passed'] for r in results.values()) and all_solver_max <= .001,
        independent_audit_required=True, joint_objective_validated=False,
        wave=1, one_sided_alpha=.0125,
        scope='Fixed full-population mechanism, four paired seeds, three fixed attacks only')
