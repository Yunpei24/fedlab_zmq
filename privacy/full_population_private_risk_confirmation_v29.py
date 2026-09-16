"""Frozen V29 mechanisms and scalar confirmation criteria; no new DP theorem."""
from fractions import Fraction as Q
from functools import lru_cache
import math
import statistics as st
from privacy.fair_objective import calibrate, epsilon_bound, wor_rdp, per_example, release, require_mps
from privacy.full_population_private_risk_v28 import ledger as full_ledger, private_population_gradient

SEEDS = (180701, 180702, 180703, 180704)
METHODS = ('erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa')
ARMS = tuple((4800, m) for m in METHODS)+((240, 'erm_mean'), (240, 'erm_rfa'))
CONTROLS = ((4800, 'erm_mean'), (4800, 'erm_rfa'), (240, 'erm_mean'), (240, 'erm_rfa'))
KEYS = ('accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2')
TCRIT = 4.176534846104499


@lru_cache(maxsize=6)
def plan(batch, method):
    if type(batch) is not int or (batch, method) not in ARMS:
        raise ValueError('Only the six preregistered arms are allowed')
    if batch == 4800:
        return full_ledger(method)
    z = calibrate(q=.05, steps=120, epsilon=4., delta=1e-5)
    eps, order = epsilon_bound(q=.05, z=z, steps=120, delta=1e-5)
    return dict(N=4800, b=240, T=120, C=2., gradient_z=z, gradient_std=4*z/240,
        gradient_sensitivity=4/240, risk_z=None, risk_std=0., risk_sensitivity=None,
        risk_releases=0, gradient_releases=120, epsilon_realized=eps, order=order,
        epsilon_cap=4., delta=1e-5, adjacency='replace_one', sampling='fixed_without_replacement',
        rdp={a:120*wor_rdp(a,.05,z) for a in range(2,65)},
        accounting='V19 replication: generic WBK Theorem9, basic conversion, orders2..64',
        scope='sample-level per-client per-run messages only')


def prefix(plan_, t):
    if type(t) is not int or not 0 <= t <= 120:
        raise ValueError('Invalid prefix')
    if not t:
        return 0., 2
    return min((t*float(v)/120+math.log(1/plan_['delta'])/(int(a)-1),int(a))
               for a,v in plan_['rdp'].items())


def private_message(model, x, y, *, batch, noise_std, seed):
    require_mps()
    if type(batch) is not int or batch not in (240,4800) or len(x) != batch or len(y) != batch:
        raise ValueError('The query must contain exactly its public batch size')
    if model.training or not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError('Fixed evaluation model and positive Gaussian noise required')
    if batch == 4800:
        return private_population_gradient(model,x,y,C=2.,block_size=240,noise_std=noise_std,seed=seed)
    losses, rows, norms, _ = per_example(model,x,y,kind='brier',clip_norm=2.)
    query = rows.mean(0)  # Preserve the original V19 small-batch reduction.
    message = release(query,noise_std=noise_std,seed=seed)
    return message, query, dict(population_size=4800,batch_size=240,accumulation_blocks=1,
        gaussian_releases=1,per_example_clipped_count=int((norms>2).sum()),
        per_example_clipped_fraction=float((norms>2).float().mean()),
        raw_batch_brier_risk=float(losses.mean()),replace_one_sensitivity=4/240,
        local_optimizer_steps=0,noise_added_after_complete_mean=True,diagnostic_fields_not_private=True)


def exact_test_metrics(validation):
    clients = validation['clients']
    if len(clients) != 10:
        raise ValueError('Ten test partitions required')
    acc=[]
    for c in clients:
        ns, hs = c['class_count'], c['class_hits']
        if (c['N'] != 1000 or len(ns) != 10 or len(hs) != 10 or sum(ns) != 1000
                or any(not math.isfinite(x) or x<0 or x!=int(x) for x in ns+hs)
                or any(h>n for h,n in zip(hs,ns))):
            raise ValueError('Invalid test class counts')
        acc.append(Q(int(sum(hs)),10))
    mean=sum(acc)/10; ss=sorted(acc); worst=sum(ss[:2])/2
    values=dict(accuracy_pct=mean,worst20_pct=worst,
        gap_best20_worst20_pp=sum(ss[-2:])/2-worst,variance_pp2=sum((a-mean)**2 for a in acc)/10)
    for k,v in values.items():
        if not math.isfinite(validation[k]) or abs(float(v)-validation[k])>1e-9:
            raise ValueError('Stored metric does not match exact class counts')
    return values


def compare(pairs):
    if len(pairs)!=4 or any(set(p)!=set(KEYS) or any(not isinstance(v,Q) for v in p.values()) for p in pairs):
        raise ValueError('Four exact count-derived differences required')
    summaries={}
    for k in KEYS:
        xs=[p[k] for p in pairs]; mean=sum(xs)/4
        sd=math.sqrt(float(sum((v-mean)**2 for v in xs)/3))
        lower=float(mean)-TCRIT*sd/2
        summaries[k]=dict(n=4,mean=float(mean),sd=sd,lower_one_sided_9875=lower,df=3,
            exact_mean=dict(numerator=mean.numerator,denominator=mean.denominator))
    gates=dict(all_seed_gates=all(p['accuracy_pct']>=-1 and p['worst20_pct']>=1 for p in pairs),
        worst20_lower_positive=summaries['worst20_pct']['lower_one_sided_9875']>0,
        accuracy_lower_noninferior=summaries['accuracy_pct']['lower_one_sided_9875']>=-1,
        mean_gap_nonincreasing=sum(p['gap_best20_worst20_pp'] for p in pairs)<=0,
        mean_variance_nonincreasing=sum(p['variance_pp2'] for p in pairs)<=0)
    return dict(summaries=summaries,gates=gates,passed=all(gates.values()))


def decide(records):
    expected={(s,b,m) for s in SEEDS for b,m in ARMS}; index={}
    if len(records)!=24:
        raise ValueError('No gate before all 24 runs')
    for r in records:
        j=r['job']; key=(j['seed'],j['batch'],j['method'])
        if key in index or key not in expected:
            raise ValueError('Duplicate or unregistered arm')
        if (r['final']['round']!=120 or r['test_evaluation_rounds']!=[120]
                or not r['test_evaluated'] or r['device']!='mps'):
            raise ValueError('Fixed final test endpoint required')
        index[key]=exact_test_metrics(r['final']['test'])
    if set(index)!=expected:
        raise ValueError('Missing fixed controls')
    contrasts=[]
    for b,m in CONTROLS:
        ds=[{k:index[s,4800,'risk_rfa'][k]-index[s,b,m][k] for k in KEYS} for s in SEEDS]
        contrasts.append(dict(control_batch=b,control_method=m,
            pairs=[dict(seed=s,delta={k:float(v) for k,v in d.items()}) for s,d in zip(SEEDS,ds)],**compare(ds)))
    return dict(clean_confirmation_passed=all(c['passed'] for c in contrasts),contrasts=contrasts,
        primary_method='risk_rfa',primary_batch=4800,endpoint='final_test_round_120',
        wave=1,one_sided_alpha=.0125,t_critical_df3=TCRIT,independent_audit_required=True,
        attacks_evaluated=False,joint_objective_validated=False)
