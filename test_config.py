import sys,argparse,json,tempfile
from pathlib import Path
R=Path(__file__).resolve().parent;domain=sys.argv[1]
d=R/('code_DVS' if domain=='dvs' else 'code_speech');sys.path.insert(0,str(d))
from config_loader import Config,get_argparser
path=d/'configs/cifar10dvs/16T/cifar10dvs_spikformer.yaml' if domain=='dvs' else d/'configs/T=100/SHD/shd_spikescr.yaml'
rows=[]
for mode in ['full','half','tacos']:
 args=get_argparser().parse_args(['--config',str(path),'--tacos-mode',mode,'--no-make-plot'])
 c=Config(str(path),args)
 assert c.seed_list==[42,43,44,45,46]
 assert c.drop_sequence==(mode=='tacos') and not c.make_plot
 assert c.drop_comp_cap is None and c.drop_ratio==.5
 assert c.drop_windows==([2,4] if domain=='dvs' else [2,4,8])
 if domain=='dvs':assert c.time_steps==(8 if mode=='half' else 16) and c.mlp_ratios==4 and args.lr is None
 else:assert c.duration==(20 if mode=='half' else 10)
 with tempfile.TemporaryDirectory(dir=str(R)) as td:
  f=Path(td)/'resolved.yaml';c.save(str(f));restored=Config(str(f))
  assert restored.tacos_mode==mode and restored.tacos_profile=='paper'
 rows.append({'mode':mode,'time_steps':getattr(c,'time_steps',None),'duration':getattr(c,'duration',None),'lr':c.lr,'epochs':c.epochs})
print(json.dumps({'passed':True,'domain':domain,'configs':rows}))
