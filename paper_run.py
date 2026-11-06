"""Explicit paper entry point. Dry run by default, --execute starts training."""
import argparse,json,sys,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('domain',choices=['dvs','speech']);p.add_argument('model');p.add_argument('mode',choices=['full','half','tacos'])
p.add_argument('--dataset');p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46]);p.add_argument('--execute',action='store_true')
a=p.parse_args();dataset=a.dataset or ('cifar10dvs' if a.domain=='dvs' else 'shd')
d=R/('code_DVS' if a.domain=='dvs' else 'code_speech')
c=d/'configs'/dataset/'16T'/(dataset+'_'+a.model+'.yaml') if a.domain=='dvs' else d/'configs/T=100'/dataset.upper()/(dataset+'_'+a.model+'.yaml')
if not c.exists():raise FileNotFoundError(c)
data=Path(a.data).resolve();output=Path(a.output).resolve()
sys.path.insert(0,str(d));from config_loader import Config
config=Config(str(c),argparse.Namespace(tacos_mode=a.mode,tacos_profile='paper',datasets_path=str(data),seed_list=a.seeds))
cmd=[sys.executable,str(d/'train.py'),'--config',str(c),'--tacos-mode',a.mode,'--tacos-profile','paper','--datasets-path',str(data),'--seed-list']+list(map(str,a.seeds))+['--gpu',str(a.gpu),'--run-tag','paper-r1-'+dataset+'-'+a.model+'-'+a.mode]
print(json.dumps({'command':cmd,'cwd':str(output),'resolved':{k:getattr(config,k,None) for k in ['model_name','time_steps','duration','mlp_ratios','drop_sequence','drop_ratio','drop_windows','drop_comp_window','drop_comp_cap','seed_list','lr','epochs','batch_size']},'execute':a.execute},indent=2),flush=True)
if a.execute:
 if not data.is_dir():raise FileNotFoundError(data)
 if output.exists():raise FileExistsError('Choose a new output directory: '+str(output))
 output.mkdir(parents=True)
 raise SystemExit(subprocess.call(cmd,cwd=str(output)))
