"""CPU tests for the explicitly requested Toubkal lane, not historical MPS runs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.run_far_dp_effect_cpu import (
    load_config, make_tasks, generator, release, aggregate, apply, evaluate,
    prepare, stream_seed, ROOT, DEFAULT, run,
)


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def test_stage_counts_mc_seeds_and_lock():
    m=load_config(DEFAULT)
    assert len(make_tasks(m,'smoke'))==18
    assert len(make_tasks(m,'calibration'))==72
    with pytest.raises(ValueError): make_tasks(m,'confirmation')
    lock={'datasets':{d:dict(batch=1200,rounds=300,learning_rate=.1,server_clip=16.,fcc_radius=8.) for d in m['datasets']}}
    tasks=make_tasks(m,'confirmation',lock=lock)
    assert len(tasks)==88920
    assert len({t['id'] for t in tasks})==88920
    assert {t['seed'] for t in tasks}==set(m['outer_seeds'])
    assert {t['mc'] for t in tasks}=={0,1,2}
    assert all(t['reference']=='centered_clipping' for t in tasks if t['alpha']==0)
    assert not any(t['epsilon'] is not None and t['clip'] is None for t in tasks)


def test_prepare_freeze_and_no_overwrite(tmp_path):
    a=SimpleNamespace(config=str(DEFAULT),stage='smoke',lock=None,dataset=['fashionmnist'],output_root=str(tmp_path))
    prepare(a); path=tmp_path/'smoke/manifest.json'; old=path.read_bytes()
    prepare(a); assert path.read_bytes()==old
    assert (tmp_path/'.simulation_entropy').stat().st_mode & 0o077 == 0
    a.dataset=['mnist']
    with pytest.raises(RuntimeError): prepare(a)
    assert path.read_bytes()==old


def test_cpu_release_autograd_clip_noise_microbatch():
    torch.manual_seed(4);model=torch.nn.Linear(3,2).eval()
    x=torch.randn(12,3);y=torch.arange(12)%2
    original=copy.deepcopy(model.state_dict())
    g=torch.autograd.grad(torch.nn.functional.cross_entropy(model(x),y),tuple(model.parameters()))
    expected=torch.cat([a.flatten() for a in g])
    raw=release(model,x,y,None,0,77,4)
    assert torch.allclose(raw,expected,atol=1e-6)
    clean=release(model,x,y,.1,0,77,4)
    noisy=release(model,x,y,.1,.03,77,4)
    assert clean.norm()<=.100001
    assert torch.allclose(noisy,clean+.03*torch.randn(clean.shape,generator=generator(77)),atol=1e-7)
    assert torch.allclose(noisy,release(model,x,y,.1,.03,77,12),atol=1e-6)
    assert all(torch.equal(original[n],v) for n,v in model.state_dict().items())
    with pytest.raises(ValueError): release(model,x,y,None,.03,77,4)


@pytest.mark.parametrize('ref',['rfa','coordinate_median','trimmed_mean','centered_clipping'])
def test_uniform_and_signed_raw_far(ref):
    m=load_config(DEFAULT);x=torch.arange(30,dtype=torch.float32).reshape(10,3)/20
    task=dict(reference=ref,alpha=0,server_clip=None,fcc_radius=8.)
    a,_,_=aggregate(x,task,m,torch.zeros(3));assert torch.allclose(a,x.mean(0),atol=1e-6)
    task['alpha']=-3.4
    a,r,_=aggregate(x,task,m,torch.zeros(3));d=(x-r).norm(dim=1)
    assert torch.allclose(a,(torch.softmax(-3.4*d,0)[:,None]*x).sum(0),atol=1e-6)
    task['server_clip']=.2
    a,_,diag=aggregate(x,task,m,torch.zeros(3));assert a.norm()<=.200001
    assert diag['server_clip_fraction']>0


def test_batches_pair_across_arms_and_change_across_mc():
    seed=stream_seed('test',42,0,'batch',3,2)
    first=torch.randperm(4800,generator=generator(seed))[:300]
    torch.randn(8000) # unrelated draws must not change this batch
    second=torch.randperm(4800,generator=generator(seed))[:300]
    assert torch.equal(first,second) and len(first.unique())==300
    other=torch.randperm(4800,generator=generator(stream_seed('test',42,1,'batch',3,2)))[:300]
    assert not torch.equal(first,other)


def test_evaluation_exact_metrics():
    model=torch.nn.Identity();x=torch.eye(10)*9;y=torch.arange(10)
    data=dict(x=x,y=y,val=[torch.arange(10) for _ in range(10)])
    e=evaluate(model,data,'val')
    assert e['accuracy_pct']==100 and e['worst20_pct']==100
    assert e['variance_pp2']==0 and e['gap_pp']==0
    assert e['balanced_accuracy_pct']==100


def test_signal_and_resume_real_execution_contract(tmp_path,monkeypatch):
    # Synthetic CPU fixture exercises runner, checkpoints, state/anchor and outputs.
    import scripts.run_far_dp_effect_cpu as runner
    m=load_config(DEFAULT);m['smoke']['batch']=4;m['smoke']['rounds']=2
    config=tmp_path/'matrix.yaml';config.write_text(yaml.safe_dump(m))
    args=SimpleNamespace(config=str(config),stage='smoke',lock=None,dataset=['fashionmnist'],output_root=str(tmp_path/'runs'))
    prepare(args)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda _:None)
    def data(*_):
        g=generator(17);x=torch.randn(20,3,generator=g);y=torch.arange(20)%2
        return dict(x=x,y=y,train=[torch.arange(10) for _ in range(10)],val=[torch.arange(10,20) for _ in range(10)],
                    test=[],fingerprint='fixture',split_hashes={})
    monkeypatch.setattr(runner,'data_for_task',data)
    monkeypatch.setattr(runner,'get_model',lambda *_:torch.nn.Linear(3,2))
    manifest=tmp_path/'runs/smoke/manifest.json'
    a=SimpleNamespace(manifest=str(manifest),job_index=0,data_root='unused',threads=1,resume=True)
    run(a)
    folder=manifest.parent/'runs'/json.loads(manifest.read_text())['tasks'][0]['id']
    payload=json.loads((folder/'metrics.json').read_text())
    assert payload['completed_rounds']==2 and payload['device']=='cpu'
    # Simulate crash after final checkpoint before final JSON; replay must match.
    (folder/'metrics.json').unlink()
    run(a)
    replay=json.loads((folder/'metrics.json').read_text())
    assert replay==payload
    run(a) # idempotent --resume completed path
