# Run with upstream's uv environment; registers modality before upstream evaluator CLI.
import argparse,runpy,sys,importlib.util
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--modality',required=True)
a,rest=p.parse_known_args();repo=Path(a.repo).resolve();sys.path.insert(0,str(repo))
config=Path(a.modality).resolve();spec=importlib.util.spec_from_file_location('custom_modality',config)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.argv=['open_loop_eval.py']+rest
runpy.run_path(str(repo/'gr00t/eval/open_loop_eval.py'),run_name='__main__')
