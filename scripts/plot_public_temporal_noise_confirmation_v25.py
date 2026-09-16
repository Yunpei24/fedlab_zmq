#!/usr/bin/env python3
"""Plot only the completed independent V25 audit, with no parameter selection."""
import json
import os
from pathlib import Path
import statistics as st
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib-v25'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = ROOT/'output/analysis/Public_Temporal_Noise_Confirmation_V25_Analyse.json'
OUT = ROOT/'output/analysis/figures_public_temporal_noise_v25'


def main():
    audit = json.loads(SOURCE.read_text())
    assert audit['audit_passed'] and audit['runs'] == 32 and not audit['snapshot_partial']
    decision = audit['decision']; assert decision is not None
    OUT.mkdir(exist_ok=True)
    rows = audit['final_test_records']
    seeds = sorted({r['job']['seed'] for r in rows})
    methods = ('erm_mean','erm_rfa','risk_mean','risk_rfa')
    labels = ('ERM\nmoyenne','ERM\nRFA','Risque\nmoyenne','Risque\nRFA')
    index = {(r['job']['seed'],r['job']['grid_index'],r['job']['method']):r['test'] for r in rows}
    colors = {0:'#4c78a8',13:'#e17c05'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':160})
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(key,title) in zip(axes.flat,[('accuracy_pct','Accuracy test (%) — plus élevée : mieux'),
        ('worst20_pct','Worst-20 test (%) — plus élevé : mieux'),
        ('gap_best20_worst20_pp','Gap Best-20–Worst-20 (pp) — plus faible : mieux'),
        ('variance_pp2','Variance entre clients (pp²) — plus faible : mieux')]):
        for mode,offset,label in ((0,-.14,'Bruit constant'),(13,.14,'Bruit réalloué k13')):
            x = [i+offset for i in range(4)]
            values = [[index[s,mode,m][key] for s in seeds] for m in methods]
            means,sd = [st.mean(v) for v in values],[st.stdev(v) for v in values]
            ax.errorbar(x,means,yerr=sd,fmt='o',capsize=4,color=colors[mode],label=label,
                        markersize=6,zorder=3)
            for i,v in enumerate(values):
                ax.scatter([x[i]+(j-1.5)*.025 for j in range(4)],v,s=15,alpha=.45,
                           color=colors[mode],zorder=2)
        ax.set_xticks(range(4),labels); ax.set_title(title,fontsize=11); ax.grid(axis='y',alpha=.25)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('V25 — test final fixé au tour 120, quatre nouvelles seeds\n'
                 'Points pleins : moyenne ; barres : écart-type (pas IC95) ; petits points : seeds',fontsize=13)
    fig.savefig(OUT/'test_metrics.png'); plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    contrasts = decision['contrasts']
    names = [('Constant' if c['control_grid_index']==0 else 'Réalloué')+' / '+
             ('ERM moyenne' if c['control_method']=='erm_mean' else 'ERM RFA') for c in contrasts]
    for ax,key,title,threshold in zip(axes,('accuracy_pct','worst20_pct'),
        ('Δ accuracy (pp)','Δ Worst-20 (pp)'),(-1.,0.)):
        for i,c in enumerate(contrasts):
            summary = c['summaries'][key]; mean = summary['mean']; low,high = summary['ci95']
            ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',capsize=5,
                        color='#202020',markersize=6,zorder=3)
            ax.scatter([p['delta'][key] for p in c['pairs']],[i+.13]*4,s=20,
                       color='#e17c05',alpha=.75,zorder=2)
        ax.axvline(threshold,ls='--',color='#b22222',label=f'Seuil de la borne IC : {threshold:g} pp')
        if threshold != 0:
            ax.axvline(0,color='grey',lw=.7)
        else:
            ax.axvline(1,color='grey',ls=':',label='Marge par seed : +1 pp')
        ax.set_yticks(range(4),names); ax.invert_yaxis(); ax.set_xlabel(title)
        ax.grid(axis='x',alpha=.25); ax.legend(fontsize=8,loc='best')
    verdict = 'PASS' if decision['clean_confirmation_passed'] else 'FAIL'
    fig.suptitle('V25 — candidate risque-RFA k13 moins chaque contrôle\n'
        f'IC95 t appariés (4 seeds, df=3) et différences par seed — confirmation propre : {verdict}',fontsize=12)
    fig.savefig(OUT/'paired_primary_intervals.png'); plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
