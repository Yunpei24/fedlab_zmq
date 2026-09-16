"""Execute only dependencies and affected plots, preserving every other cell."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'notebooks/LDP_Gradient_FAR_Positioning_Analysis_Editable.ipynb'
OUT=ROOT/'output/analysis/notebook_observed_connections_20260915'


def check_graphics(nb):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scope=dict(np=np,pd=pd,plt=plt,display=lambda *args:None,Markdown=str)
    cells={c.id:c for c in nb.cells}
    for cid,name in [('180afe8c731ef31f','_positioning_round_coverage_note'),
                     ('180afe8c731ef31f','_mean_sd_by_round'),
                     ('180afe8c731ef31f','_plot_one_trajectory'),
                     ('rcig-review-alpha-accuracy','plot_all_alpha_trajectories')]:
        node=next(x for x in ast.parse(cells[cid].source).body if isinstance(x,ast.FunctionDef) and x.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<plot-test>','exec'),scope)
    observed=list(range(1,20,2))+[20]
    vals=[float(t) if t in observed else np.nan for t in range(1,21)]
    df=pd.DataFrame(dict(reference='F_CC',n_clients=10,C_local=8.,horizon=20,
                        alpha=0.,round=np.arange(1,21),gap_pp=vals,condition='control'))
    original=df.copy(deep=True)
    plots=[]
    scope.update(alpha_all=df,ALPHA_REFERENCES_TO_PLOT=['F_CC'],ALPHA_CLIENTS_TO_PLOT=[10],
                 _save_alpha_figure=lambda fig,name:plots.append(fig))
    scope['plot_all_alpha_trajectories']('gap_pp','Gap')
    scope['_plot_one_trajectory'](df,'gap_pp','Gap','Test')
    plots.append(plt.gcf())
    for fig in plots:
        line=fig.axes[0].lines[0]
        np.testing.assert_array_equal(line.get_xdata(),observed)
        np.testing.assert_array_equal(line.get_ydata(),observed)
        assert '11/20' in fig.axes[0].get_xlabel()
        plt.close(fig)
    pd.testing.assert_frame_equal(df,original)


if __name__=='__main__':
    nb=nbformat.read(PATH,as_version=4)
    check_graphics(nb)
    original=copy.deepcopy(nb)
    source_hash=hashlib.sha256(PATH.read_bytes()).hexdigest()
    # Dependencies + all calls sharing the modified plotting functions.
    indices=[1,3,6,10,11,12,13,16,18,19,20,21]
    small=nbformat.v4.new_notebook(cells=[copy.deepcopy(nb.cells[i]) for i in indices],metadata=nb.metadata)
    for c in small.cells:
        c.outputs=[];c.execution_count=None
    client=NotebookClient(small,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    assert hashlib.sha256(PATH.read_bytes()).hexdigest()==source_hash,'Notebook changed during execution'
    affected=[10,11,12,13,16,18,19,20,21]
    pictures=[]
    import base64
    for i,c in zip(indices,small.cells):
        assert not any(o.output_type=='error' for o in c.get('outputs',[]))
        if i not in affected:continue
        nb.cells[i].outputs=c.outputs
        nb.cells[i].execution_count=c.execution_count
        for j,o in enumerate(c.outputs):
            if 'image/png' in o.get('data',{}):
                path=OUT/f'cell_{i+1}_figure_{j}.png'
                path.write_bytes(base64.b64decode(o.data['image/png']))
                pictures.append(str(path))
    for i,(a,b) in enumerate(zip(original.cells,nb.cells)):
        assert a.source==b.source
        if i not in affected:assert a==b
    nbformat.validate(nb);nbformat.write(nb,PATH)
    report=dict(notebook=str(PATH),executed_source_cells=[i+1 for i in indices],
        changed_output_cells=[i+1 for i in affected],figures=pictures,
        graphic_tests='passed: 11 observed points connected; NaNs in data preserved',
        unmodified_other_cells=True,errors=0)
    (OUT/'execution.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
