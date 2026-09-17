#!/usr/bin/env python3
"""CPU-only, nested-Monte-Carlo FAR DP study. One SLURM task = one run.

prepare freezes a stage; run never downloads or submits jobs. Confirmation
requires a filled calibration lock. A local random key pairs experimental
draws and is NOT a publishable privacy artifact. No changes to old MPS code.
"""
import argparse
import contextlib
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import secrets
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from torchvision.datasets import MNIST, FashionMNIST
from datasets.partitioner import partition_dataset
from models.registry import get_model
from privacy.local_dpsgd import _per_sample_grads_vectorized
from privacy.far_dp_effect_mc import privacy_plan, stream_seed, nested_summary
from privacy.rdp import RDPAccountant
from robustness.aggregators import aggregate_vectors

DEFAULT = ROOT/'configs/ldp_gradient_far/far_dp_effect_toubkal_cpu.yaml'
SOURCE_FILES = ['scripts/run_far_dp_effect_cpu.py', 'privacy/far_dp_effect_mc.py',
                'privacy/local_dpsgd.py', 'privacy/rdp.py', 'robustness/aggregators.py',
                'robustness/tensor_ops.py', 'models/registry.py', 'datasets/partitioner.py',
                'datasets/registry.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value, tensor=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb' if tensor else 'w') as f:
        os.fchmod(f.fileno(), 0o600)
        if tensor:
            torch.save(value, f)
        else:
            json.dump(value, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


@contextlib.contextmanager
def exclusive(path):
    with Path(path).open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process owns {path}') from None
        yield


def load_config(path):
    m = yaml.safe_load(Path(path).read_text())
    assert m['schema_version'] == 1 and m['device'] == 'cpu'
    assert m['num_clients'] == 10 and m['train_per_client']+m['validation_per_client'] == 6000
    assert m['test_per_client'] == 1000 and m['model'] in {'lenet5_tanh', 'lenet5_relu'}
    assert len(set(m['outer_seeds'])) == 10
    assert not set(m['outer_seeds']) & set(m['calibration_seeds'])
    assert m['mc_repeats'] >= 2 and m['microbatch'] >= 1
    assert set(m['references']) == {'rfa', 'coordinate_median', 'trimmed_mean', 'centered_clipping'}
    assert len(m['alphas']) == len(set(m['alphas'])) and 0 in m['alphas']
    assert 0 < m['delta'] < 1 and 0 <= m['fcc_anchor_rate'] <= 1
    return m


def make_tasks(m, stage, selected_datasets=None, lock=None):
    datasets = selected_datasets or m['datasets']
    if not set(datasets) <= set(m['datasets']):
        raise ValueError('unknown dataset')
    tasks = []
    def add(dataset, seed, mc, **kw):
        task = dict(dataset=dataset, seed=seed, mc=mc, stage=stage, **kw)
        task['id'] = f'{dataset}_s{seed}_mc{mc}_{canonical(task)[:16]}'
        tasks.append(task)
    for dataset in datasets:
        if stage == 'smoke':
            s = m['smoke']
            for ref, epsilon in itertools.product(m['references'], [None, 4.]):
                add(dataset, m['calibration_seeds'][0], 0, reference=ref, alpha=.6,
                    epsilon=epsilon, clip=s['local_clip'], server_clip=s['server_clip'],
                    batch=s['batch'], rounds=s['rounds'], learning_rate=s['learning_rate'],
                    fcc_radius=s['fcc_radius'])
            add(dataset, m['calibration_seeds'][0], 0, reference='centered_clipping', alpha=0.,
                epsilon=None, clip=None, server_clip=None, batch=s['batch'], rounds=s['rounds'],
                learning_rate=s['learning_rate'], fcc_radius=s['fcc_radius'])
        elif stage == 'calibration':
            c = m['calibration']
            for seed, mc, batch, lr, arm in itertools.product(m['calibration_seeds'],
                    range(c['mc_repeats']), c['batches'], c['learning_rates'], ['raw', 'clip', 'dp']):
                add(dataset, seed, mc, reference='centered_clipping', alpha=0.,
                    epsilon=c['epsilon'] if arm == 'dp' else None,
                    clip=None if arm == 'raw' else c['local_clip'], server_clip=None,
                    batch=batch, rounds=c['rounds'], learning_rate=lr,
                    fcc_radius=c['diagnostic_fcc_radius'])
        else:
            if lock is None:
                raise ValueError('confirmation requires --lock with calibration choices')
            p = lock['datasets'][dataset]
            for name in ['batch', 'rounds', 'learning_rate', 'server_clip', 'fcc_radius']:
                if not isinstance(p.get(name), (int, float)) or p[name] <= 0:
                    raise ValueError(f'fill positive {dataset}.{name} in calibration lock')
            if int(p['batch']) != p['batch'] or p['batch'] > m['train_per_client']:
                raise ValueError('invalid locked batch')
            if int(p['rounds']) != p['rounds']:
                raise ValueError('integer horizon required')
            alpha_refs = [(a, f) for a in m['alphas'] for f in
                          (['centered_clipping'] if a == 0 else m['references'])]
            arms = [(c, e, u) for c, e, u in itertools.product(m['local_clips'],
                    [None]+m['epsilon_values'], [None, p['server_clip']])]
            arms += [(None, None, None), (None, None, p['server_clip'])]
            for (alpha, ref), (c, eps, u), seed, mc in itertools.product(alpha_refs, arms,
                    m['outer_seeds'], range(m['mc_repeats'])):
                add(dataset, seed, mc, reference=ref, alpha=alpha, epsilon=eps, clip=c,
                    server_clip=u, **{k:p[k] for k in ['batch', 'rounds', 'learning_rate', 'fcc_radius']})
    assert len({t['id'] for t in tasks}) == len(tasks)
    return tasks


def prepare(args):
    m = load_config(args.config)
    lock = yaml.safe_load(Path(args.lock).read_text()) if args.lock else None
    extra = {}
    if args.stage == 'confirmation':
        if lock is None or not lock.get('calibration_report'):
            raise ValueError('confirmation requires filled --lock and calibration_report')
        report = Path(lock['calibration_report'])
        if not report.is_absolute(): report = Path(args.lock).resolve().parent/report
        extra = dict(calibration_lock=lock, calibration_report_sha256=digest(report))
    tasks = make_tasks(m, args.stage, args.dataset, lock)
    out = Path(args.output_root).resolve(); out.mkdir(parents=True, exist_ok=True)
    with exclusive(out/'prepare.lock'):
        key = out/'.simulation_entropy'
        if not key.exists():
            fd = os.open(key, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'w') as f: f.write(secrets.token_hex(32))
        path = out/args.stage/'manifest.json'
        payload = dict(schema_version=1, config=m, stage=args.stage, tasks=tasks,
                       source_sha256={name:digest(ROOT/name) for name in SOURCE_FILES},
                       entropy_sha256=digest(key), privacy_scope='research_simulation_per_run_only', **extra)
        payload['signature'] = canonical(payload)
        if path.exists():
            old = json.loads(path.read_text())
            if old != payload: raise RuntimeError('manifest exists with different choices/sources; use a NEW root')
        else:
            save(path, payload)
    print(json.dumps(dict(stage=args.stage, tasks=len(tasks), manifest=str(path),
                         array_example=f'0-{min(999,len(tasks)-1)}%4'), indent=2))


def generator(seed):
    return torch.Generator(device='cpu').manual_seed(seed)


def raw_datasets(name, data_root, include_test=False, download=False):
    cls = {'mnist':MNIST, 'fashionmnist':FashionMNIST}[name]
    tr = cls(data_root, train=True, download=download)
    te = cls(data_root, train=False, download=download) if include_test else None
    return tr, te


def data_for_task(m, task, data_root):
    train, test = raw_datasets(task['dataset'], data_root, task['stage'] == 'confirmation')
    mean, std = (.1307, .3081) if task['dataset'] == 'mnist' else (.286, .353)
    data = dict(x=(train.data.float().unsqueeze(1)/255-mean)/std, y=train.targets,
                train=[], val=[], test=[])
    seed = task['seed']
    kwargs = dict(num_clients=10, partition=m['partition'], alpha=m['dirichlet_beta'], seed=seed)
    parts = partition_dataset(train, **kwargs)
    for cid, part in enumerate(parts):
        ids = torch.tensor(part.indices, dtype=torch.long)
        if len(ids) != 6000: raise ValueError('expected equal public client size 6000')
        perm = torch.randperm(len(ids), generator=generator(stream_seed('public-split', seed, 0, 'split', 0, cid)))
        ids = ids[perm]; data['train'].append(ids[:m['train_per_client']]); data['val'].append(ids[m['train_per_client']:])
    if test is not None:
        data.update(xt=(test.data.float().unsqueeze(1)/255-mean)/std, yt=test.targets)
        data['test'] = [torch.tensor(p.indices) for p in partition_dataset(test, **kwargs)]
    split_hashes = {name:[hashlib.sha256(ids.numpy().tobytes()).hexdigest() for ids in data[name]]
                    for name in ['train','val','test']}
    data['fingerprint'] = canonical(dict(splits=split_hashes,
        train_images=hashlib.sha256(train.data.numpy().tobytes()).hexdigest(),
        train_labels=hashlib.sha256(train.targets.numpy().tobytes()).hexdigest(),
        test_images=None if test is None else hashlib.sha256(test.data.numpy().tobytes()).hexdigest(),
        test_labels=None if test is None else hashlib.sha256(test.targets.numpy().tobytes()).hexdigest()))
    data['split_hashes'] = split_hashes
    return data


def release(model, x, y, clip, noise_std, noise_seed, microbatch):
    if any(p.device.type != 'cpu' for p in model.parameters()): raise ValueError('CPU only')
    if clip is None and noise_std: raise ValueError('DP without local clipping forbidden')
    params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    sums = [torch.zeros_like(p) for _,p in params]
    for xx, yy in zip(x.split(microbatch), y.split(microbatch)):
        gs, _ = _per_sample_grads_vectorized(model, [n for n,_ in params], xx, yy)
        norms = sum(g.reshape(len(xx),-1).square().sum(1) for g in gs).sqrt()
        factors = torch.ones_like(norms) if clip is None else (clip/norms.clamp_min(1e-12)).clamp(max=1)
        for s,g in zip(sums,gs): s.add_((g*factors.view((len(xx),)+(1,)*(g.ndim-1))).sum(0))
    mu = torch.cat([v.flatten() for v in sums])/len(x)
    # Separate generator, no global RNG advancement in either DP or no-DP arm.
    z = torch.randn(mu.shape, generator=generator(noise_seed))
    out = mu + noise_std*z
    if not torch.isfinite(out).all(): raise FloatingPointError('nonfinite private message')
    return out


def aggregate(messages, task, m, anchor):
    x = messages
    factors = torch.ones(len(x))
    u = task['server_clip']
    if u is not None:
        factors = (u/x.norm(dim=1).clamp_min(1e-12)).clamp(max=1); x = x*factors[:,None]
    r = aggregate_vectors(x, task['reference'], f=m['trimmed_mean_f_budget'], anchor=anchor,
                          tau=task['fcc_radius'], max_iter=100, tol=1e-6)
    d = (x-r).norm(dim=1); lam = torch.softmax(task['alpha']*d, dim=0)
    a = (lam[:,None]*x).sum(0)
    if not torch.isfinite(a).all(): raise FloatingPointError('nonfinite FAR aggregate')
    diagnostics = dict(lambda_min=float(lam.min()), lambda_max=float(lam.max()),
        concentration=float(len(x)*lam.square().sum()),
        entropy_normalized=float(-(lam*lam.clamp_min(1e-30).log()).sum()/math.log(len(x))),
        distance_span=float(d.max()-d.min()), logit_span=float(abs(task['alpha'])*(d.max()-d.min())),
        server_clip_fraction=float((factors<1).float().mean()),
        upload_norm_q50=float(torch.quantile(messages.norm(dim=1),.5)),
        upload_norm_q95=float(torch.quantile(messages.norm(dim=1),.95)),
        distance_q50=float(torch.quantile(d,.5)), distance_q95=float(torch.quantile(d,.95)),
        aggregate_norm=float(a.norm()))
    return a, r, diagnostics


@torch.no_grad()
def apply(model, a, lr):
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            size=p.numel(); p.sub_(lr*a[offset:offset+size].view_as(p)); offset+=size
    assert offset == len(a)


@torch.no_grad()
def evaluate(model, data, split, limit=None):
    x,y = (data['xt'],data['yt']) if split=='test' else (data['x'],data['y'])
    clients=[]
    for ids in data[split]:
        if limit: ids=ids[:limit]
        hits=0; loss=0.; counts=torch.zeros(10,dtype=torch.int64); correct=counts.clone()
        for idx in ids.split(256):
            out=model(x[idx]); labels=y[idx]; preds=out.argmax(1)
            hits+=int((preds==labels).sum()); loss+=float(torch.nn.functional.cross_entropy(out,labels,reduction='sum'))
            counts+=torch.bincount(labels,minlength=10)
            correct+=torch.bincount(labels[preds==labels],minlength=10)
        present=counts>0
        clients.append(dict(n=len(ids), hits=hits, accuracy_pct=100*hits/len(ids), loss=loss/len(ids),
            class_count=counts.tolist(), class_correct=correct.tolist(),
            balanced_accuracy_pct=100*float((correct[present].float()/counts[present]).mean())))
    values=[c['accuracy_pct'] for c in clients]; order=sorted(values)
    return dict(accuracy_pct=100*sum(c['hits'] for c in clients)/sum(c['n'] for c in clients),
        client_accuracy_pct=statistics.mean(values), loss=sum(c['loss']*c['n'] for c in clients)/sum(c['n'] for c in clients),
        variance_pp2=statistics.pvariance(values), worst20_pct=statistics.mean(order[:2]),
        best20_pct=statistics.mean(order[-2:]), gap_pp=statistics.mean(order[-2:])-statistics.mean(order[:2]),
        minmax_gap_pp=max(values)-min(values), balanced_accuracy_pct=statistics.mean(c['balanced_accuracy_pct'] for c in clients),
        clients=clients)


def run(args):
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest=json.loads(Path(args.manifest).read_text()); signature=manifest.pop('signature')
    if canonical(manifest)!=signature: raise RuntimeError('manifest signature mismatch')
    for name,sha in manifest['source_sha256'].items():
        if digest(ROOT/name)!=sha: raise RuntimeError('frozen source changed: '+name)
    task=manifest['tasks'][args.job_index]; m=manifest['config']
    stage_root=Path(args.manifest).resolve().parent; key=stage_root.parent/'.simulation_entropy'
    if digest(key)!=manifest['entropy_sha256']: raise RuntimeError('pairing entropy changed/missing')
    entropy=key.read_text().strip(); folder=stage_root/'runs'/task['id']; folder.mkdir(parents=True,exist_ok=True)
    with exclusive(folder/'run.lock'):
        status_path=folder/'orchestration_status.json'; cp=folder/'checkpoint.pt'
        if (folder/'metrics.json').exists():
            saved=json.loads((folder/'metrics.json').read_text())
            if saved['manifest_signature']!=signature or saved['task']!=task: raise RuntimeError('output signature mismatch')
            if saved['completed_rounds']!=task['rounds']: raise RuntimeError('invalid completed output')
            if not args.resume: raise RuntimeError('existing run: use --resume')
            save(status_path,dict(status='completed',device='cpu',round=task['rounds'],metrics_sha256=digest(folder/'metrics.json')))
            print('Already complete:',task['id']); return
        if cp.exists() and not args.resume: raise RuntimeError('checkpoint exists; use --resume')
        runtime=dict(torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),threads=args.threads)
        save(status_path,dict(status='running',pid=os.getpid(),device='cpu',task=task,slurm_job_id=os.getenv('SLURM_JOB_ID')))
        try:
            data=data_for_task(m,task,args.data_root)
            torch.manual_seed(stream_seed('public-init',task['seed'],0,'init'))
            model=get_model(m['model'],task['dataset']).cpu().eval()
            dim=sum(p.numel() for p in model.parameters() if p.requires_grad); anchor=torch.zeros(dim)
            plan=privacy_plan(epsilon=task['epsilon'],delta=m['delta'],n_local=m['train_per_client'],
                              batch=task['batch'],rounds=task['rounds'],clip=task['clip'] or 1.)
            rows=[]; begin=0; stop=[False]
            signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
            signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True))
            if cp.exists():
                state=torch.load(cp,map_location='cpu',weights_only=False)
                if (state['signature'],state['task_id'],state['data_fingerprint'],state['runtime'])!=(signature,task['id'],data['fingerprint'],runtime):
                    raise RuntimeError('resume sources/data/runtime changed')
                model.load_state_dict(state['model']); anchor=state['anchor']; rows=state['rows']; begin=state['round']
            def checkpoint(round_index):
                save(cp,dict(signature=signature,task_id=task['id'],data_fingerprint=data['fingerprint'],runtime=runtime,
                             model=model.state_dict(),anchor=anchor,rows=rows,round=round_index),tensor=True)
            for t in range(begin,task['rounds']):
                started=time.perf_counter(); messages=[]; hashes=[]
                for cid,ids in enumerate(data['train']):
                    seed=stream_seed(entropy,task['seed'],task['mc'],'batch',t,cid)
                    chosen=ids[torch.randperm(len(ids),generator=generator(seed))[:task['batch']]]
                    hashes.append(hashlib.sha256(chosen.numpy().tobytes()).hexdigest())
                    messages.append(release(model,data['x'][chosen],data['y'][chosen],task['clip'],plan['std'],
                        stream_seed(entropy,task['seed'],task['mc'],'noise',t,cid),m['microbatch']))
                a,r,diag=aggregate(torch.stack(messages),task,m,anchor)
                apply(model,a,task['learning_rate'])
                anchor=(1-m['fcc_anchor_rate'])*anchor+m['fcc_anchor_rate']*r
                row=dict(round=t+1,device='cpu',batch_hashes=hashes,**diag)
                if plan['enabled']:
                    acc=RDPAccountant();acc.add_sampled_without_replacement_gaussian(channel='gradient',
                        sampling_rate=plan['q'],noise_multiplier=plan['z_sensitivity'],steps=t+1)
                    row['epsilon_bound']=acc.epsilon(m['delta'])[0]
                else: row['epsilon_bound']=None
                if (t+1)%m['evaluation_interval']==0 or t==0 or t+1==task['rounds']:
                    row['validation']=evaluate(model,data,'val',m['smoke']['validation_examples_per_client'] if task['stage']=='smoke' else None)
                    if task['stage']=='confirmation': row['test']=evaluate(model,data,'test')
                row['wall_seconds']=time.perf_counter()-started; rows.append(row)
                if (t+1)%m['checkpoint_interval']==0 or t+1==task['rounds'] or stop[0]: checkpoint(t+1)
                save(status_path,dict(status='running',pid=os.getpid(),device='cpu',round=t+1,total=task['rounds']))
                if 'validation' in row: print(f"{task['id']} {t+1}/{task['rounds']} val={row['validation']['accuracy_pct']:.2f}",flush=True)
                if stop[0] and t+1<task['rounds']:
                    save(status_path,dict(status='paused',device='cpu',round=t+1,reason='scheduler_signal'))
                    return
            result=dict(task=task,manifest_signature=signature,runtime=runtime,data_fingerprint=data['fingerprint'],
                split_hashes=data['split_hashes'],device='cpu',privacy=plan,completed_rounds=task['rounds'],rounds=rows,
                diagnostics_are_research_not_private_transcript=True,local_optimizer_steps=0)
            if m['evaluate_train_final'] and task['stage']!='smoke': result['train_final']=evaluate(model,data,'train')
            save(folder/'metrics.json',result)
            save(status_path,dict(status='completed',device='cpu',round=task['rounds'],metrics_sha256=digest(folder/'metrics.json')))
        except Exception as exc:
            save(status_path,dict(status='failed',device='cpu',error=type(exc).__name__,message=str(exc)))
            raise


def status(args):
    p=Path(args.manifest); m=json.loads(p.read_text()); counts={}; invalid=[]
    for t in m['tasks']:
        folder=p.parent/'runs'/t['id']; path=folder/'orchestration_status.json'
        s=json.loads(path.read_text()) if path.exists() else {'status':'not_started'}
        label=s['status']
        if label=='completed':
            try:
                metrics=json.loads((folder/'metrics.json').read_text())
                assert metrics['task']==t and metrics['completed_rounds']==t['rounds'] and metrics['manifest_signature']==m['signature']
                assert len(metrics['rounds'])==t['rounds'] and metrics['device']=='cpu'
                assert digest(folder/'metrics.json')==s['metrics_sha256']
            except Exception:
                label='invalid';invalid.append(t['id'])
        counts[label]=counts.get(label,0)+1
    print(json.dumps(dict(total=len(m['tasks']),counts=counts,invalid=invalid),indent=2))


def summarize(args):
    """Nested mean/SD curves, complete confirmation groups only; no imputation."""
    p=Path(args.manifest);m=json.loads(p.read_text())
    if m['stage']!='confirmation': raise ValueError('hierarchical summary is for the confirmation stage')
    groups={}
    for task in m['tasks']:
        parameters={k:v for k,v in task.items() if k not in {'id','seed','mc'}}
        key=canonical(parameters); group=groups.setdefault(key,dict(parameters=parameters,tasks=[],paths=[]))
        group['tasks'].append(task)
        folder=p.parent/'runs'/task['id']; metric=folder/'metrics.json'; status_path=folder/'orchestration_status.json'
        if not metric.exists() or not status_path.exists(): continue
        s=json.loads(status_path.read_text())
        if s['status']!='completed': continue
        if digest(metric)!=s['metrics_sha256']: raise ValueError('metrics hash mismatch')
        payload=json.loads(metric.read_text())
        if payload['task']!=task or payload['manifest_signature']!=m['signature']: raise ValueError('task mismatch')
        group['paths'].append(metric)
    curves=[];incomplete=[]
    fields=['accuracy_pct','client_accuracy_pct','loss','variance_pp2','worst20_pct','gap_pp','balanced_accuracy_pct']
    for key,g in groups.items():
        if len(g['paths'])!=len(g['tasks']):
            incomplete.append(dict(parameters=g['parameters'],complete=len(g['paths']),expected=len(g['tasks'])))
            continue
        payloads=[json.loads(path.read_text()) for path in g['paths']]
        for split in ['validation','test']:
            times=[r['round'] for r in payloads[0]['rounds'] if split in r]
            for t in times:
                for field in fields:
                    values={s:{} for s in m['config']['outer_seeds']}
                    for payload in payloads:
                        row=payload['rounds'][t-1]
                        values[payload['task']['seed']][payload['task']['mc']]=row[split][field]
                    nested=[[values[s][mc] for mc in range(m['config']['mc_repeats'])] for s in m['config']['outer_seeds']]
                    curves.append(dict(parameters=g['parameters'],split=split,round=t,metric=field,**nested_summary(nested)))
    save(args.output,dict(curves=curves,incomplete_groups=incomplete,
                         band='mean +/- sample SD of 10 per-seed MC means; not CI95'))
    print(json.dumps(dict(complete_groups=len(groups)-len(incomplete),incomplete_groups=len(incomplete),output=args.output)))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--config',default=str(DEFAULT));a.add_argument('--stage',choices=['smoke','calibration','confirmation'],required=True)
    a.add_argument('--output-root',required=True);a.add_argument('--dataset',nargs='+');a.add_argument('--lock');a.set_defaults(func=prepare)
    a=sub.add_parser('run');a.add_argument('--manifest',required=True);a.add_argument('--job-index',type=int,required=True)
    a.add_argument('--data-root',required=True);a.add_argument('--threads',type=int,default=8);a.add_argument('--resume',action='store_true');a.set_defaults(func=run)
    a=sub.add_parser('status');a.add_argument('--manifest',required=True);a.set_defaults(func=status)
    a=sub.add_parser('summarize');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);a.set_defaults(func=summarize)
    a=sub.add_parser('download');a.add_argument('--data-root',required=True)
    a.set_defaults(func=lambda args:[raw_datasets(n,args.data_root,True,True) for n in ['mnist','fashionmnist']])
    args=p.parse_args()
    if getattr(args,'threads',1)<1 or getattr(args,'job_index',0)<0: p.error('threads > 0 and job index >= 0 required')
    args.func(args)


if __name__=='__main__': main()
