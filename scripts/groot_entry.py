"""Small version-scoped adapter: pinned Cosmos path and optional official profiler."""
import os,runpy,sys
from pathlib import Path
# cwd is the commit-pinned official GR00T repository.
sys.path.insert(0,str(Path.cwd()))
import gr00t.experiment.experiment as experiment
original=experiment.run
def pinned_run(config):
    if os.environ.get('COSMOS_LOCAL_PATH'):config.model.model_name=os.environ['COSMOS_LOCAL_PATH']
    config.training.logging_steps=1
    if os.environ.get('GROOT_PROFILE')=='1':config.training.enable_profiling=True
    return original(config)
experiment.run=pinned_run
runpy.run_path(str(Path.cwd()/'gr00t/experiment/launch_finetune.py'),run_name='__main__')
