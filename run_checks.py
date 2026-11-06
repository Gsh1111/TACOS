import subprocess,sys,json,platform,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
cases=[['test_core.py']]+[['test_models.py','dvs',m] for m in ['qkformer','spikformer','spikingformer','sdt']]+[['test_models.py','speech',m] for m in ['spikescr','qkformer','spikformer','spikcommander','sdt']]+[['code_DVS/train.py','--help'],['code_speech/train.py','--help']]
cases += [['test_config.py','dvs'],['test_config.py','speech']]
results=[]
for case in cases:
 p=subprocess.run([sys.executable]+[str(R/case[0])]+case[1:],capture_output=True,text=True,timeout=180)
 row={'case':case,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr};results.append(row)
 print(json.dumps(row),flush=True)
report={'python':sys.version,'host':platform.node(),'results':results,'passed':all(x['exit']==0 for x in results)}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2))
sys.exit(0 if report['passed'] else 1)
