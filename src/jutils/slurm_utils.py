import numpy as np
import subprocess
import os
import time
from typing import Any
import submitit
from argparse import ArgumentParser
import functools
import hydra.utils as hydra_utils
from pathlib import Path
import logging as log


def update_pythonpath_relative_hydra():
    """Update PYTHONPATH to only have absolute paths."""
    # NOTE: We do not change sys.path: we want to update paths for future instantiations
    # of python using the current environment (namely, when submitit loads the job
    # pickle).
    try:
        original_cwd = Path(hydra_utils.get_original_cwd()).resolve()
    except (AttributeError, ValueError):
        # Assume hydra is not initialized, we don't need to do anything.
        # In hydra 0.11, this returns AttributeError; later it will return ValueError
        # https://github.com/facebookresearch/hydra/issues/496
        # I don't know how else to reliably check whether Hydra is initialized.
        return
    paths = []
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = "."
    for orig_path in os.environ["PYTHONPATH"].split(":"):
        path = Path(orig_path)
        if not path.is_absolute():
            path = original_cwd / path
        paths.append(path.resolve())
    os.environ["PYTHONPATH"] = ":".join([str(x) for x in paths])
    log.info('PYTHONPATH: {}'.format(os.environ["PYTHONPATH"]))


def wrap_cmd(cmd):
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    try:
        print('wait')
        p.wait()
    except KeyboardInterrupt:
        try:
            print('Detect Ctrl+C, terminating....')
            p.terminate()
        except OSError:
            pass
        p.wait()



#### config ####
# cluster: slurm

# slurm_signal_delay_s: 120
# timeout_min: 2880 # 48 hours
# slurm_partition: learnlab

# slurm_mem_per_gpu: 40 # ${eval:'40*${ngpu}'}
# gpus_per_node: ${ngpu}
# tasks_per_node: ${ngpu}
# cpus_per_task: 10
# nodes: 1
# slurm_max_num_timeout: 100


# folder: ${exp_dir}/submitit_train_logs/
# slurm_job_name: ${expname}
# constraint: volta|volta32gb

# exclude:

def slurm_engine():
    def main_decorator(task_function):
        @functools.wraps(task_function)
        def decorated_main(cfg=None, *args, **kwargs):
            update_pythonpath_relative_hydra()
            local_flag = False
            if cfg is None:
                local_flag = True
            engine = cfg.get('engine', None)
            if engine is not None and engine.cluster == 'slurm':
                pass
            else:
                local_flag = True

            if local_flag:
                task_function(cfg, *args, **kwargs)
            else:
                # asks SLURM to send USR1 signal 30 seconds before the time limit
                additional_parameters = {}
                additional_parameters["signal"] = 'SIGUSR1@120'
                params = {
                    'timeout_min': cfg.engine.timeout_min,
                    'slurm_partition': cfg.engine.slurm_partition,
                    'cpus_per_task': cfg.engine.cpus_per_task,
                    'gpus_per_node': cfg.engine.gpus_per_node,
                    'nodes': cfg.engine.nodes,
                    'tasks_per_node': cfg.engine.tasks_per_node,
                    'slurm_job_name': cfg.engine.slurm_job_name,
                    'slurm_signal_delay_s': cfg.engine.slurm_signal_delay_s,
                    'slurm_mem_per_gpu': cfg.engine.slurm_mem_per_gpu,
                }
                init_params = {
                    "cluster": cfg.engine.cluster,
                    "folder": cfg.engine.folder,
                    "slurm_max_num_timeout": cfg.engine.slurm_max_num_timeout,
                    }
                executor = submitit.AutoExecutor(
                    **init_params)
                for k, v in cfg.engine.items():
                    if k not in params and k not in init_params:
                        additional_parameters[k] = v
                if additional_parameters['exclude'] is None:
                    additional_parameters.pop('exclude')

                executor.update_parameters(**params)
                executor.update_parameters(slurm_additional_parameters=additional_parameters)
                job = executor.submit(Worker(), task_function, cfg, *args, **kwargs)
                print(job.job_id, cfg.engine.folder)         
        return decorated_main
    
    return main_decorator


def slurm_wrapper_hydra(args, callable_fn, debug=False):
    """name matching for hydra environment"""
    if args.environment.slurm:
        executor = submitit.AutoExecutor(
            folder=args.environment.submitit_dir,
            slurm_max_num_timeout=100,
            cluster=None if args.environment.slurm else "debug",
        )
        # asks SLURM to send USR1 signal 30 seconds before the time limit
        additional_parameters = {}
        # additional_parameters = {"signal": 'SIGUSR2@30'}
        additional_parameters["signal"] = 'USR1@120'
        additional_parameters["signal"] = 'SIGUSR1@120'

        
        # additional_parameters = {"signal": 'SIGUSR1@90'}
        if args.environment.nodelist != "":
            additional_parameters = {"nodelist": args.environment.nodelist}
        if args.environment.exclude_nodes != "":
            additional_parameters.update(
                {"exclude": args.environment.exclude_nodes.replace('+', ',')})
        if args.environment.gpu_mem_gb is not None:
            additional_parameters['mem_per_gpu'] = args.environment.gpu_mem_gb
        args.environment.workers = 10 * args.environment.ngpu
        executor.update_parameters(
            timeout_min=args.environment.slurm_timeout,
            slurm_partition=args.environment.slurm_partition,
            cpus_per_task=args.environment.workers,
            gpus_per_node=args.environment.ngpu,
            nodes=args.environment.world_size,
            tasks_per_node=1,
            mem_gb=args.environment.mem_gb,
            slurm_additional_parameters=additional_parameters,
            signal_delay_s=120)
        executor.update_parameters(name=args.expname)
        job = executor.submit(callable_fn, args)
        if debug:
            print('submit job')
            time.sleep(10)
            int_type = 0 # np.random.randint(0, 2)
            print(f'interrupt job {int_type}')
            job._interrupt(timeout=False)        
            # job._interrupt()
            print('interrupt')
        if not args.environment.slurm:
            job.result()
    else:
        callable_fn(args)
    return 



class Worker:
    def checkpoint(self, *args: Any, **kwargs: Any):
        # for resubmit
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)  # submits to requeuing    

    # def __call__(self, func, **kwargs):
    #     print('args, kwargs', kwargs)
    #     print('func', func, )
    #     func(**kwargs)
    def __call__(self, func, *args, **kwargs):
        print('func', func, 'args', args, 'kwargs', kwargs)
        func(*args, **kwargs)
        


def slurm_wrapper(args, save_dir, func, func_kwargs, resubmit=True, func_args=()):
    """name matching for my own slurm args"""
    job_name = args.sl_name if args.sl_name is not None else '/'.join(save_dir.split('/')[-2:])
    save_dir = save_dir  + '/submitit_cache'
    if args.slurm:
        # asks SLURM to send USR1 signal 30 seconds before the time limit
        additional_parameters = {"signal": 'SIGUSR1@90', }
        executor = submitit.AutoExecutor(folder=save_dir)
        executor.update_parameters(
            tasks_per_node=args.sl_ntask_pnode,
            timeout_min=args.sl_time*60,
            slurm_partition=args.sl_part,
            gpus_per_node=args.sl_ngpu,
            cpus_per_task=max(args.sl_ngpu * 10, 10),          
            nodes=args.sl_node,
            slurm_job_name=job_name,
            slurm_additional_parameters=additional_parameters,
            slurm_signal_delay_s=120
        )
        print('slurm cache in ', save_dir)
        if args.sl_mem is not None:
            executor.update_parameters(mem_gb=args.sl_mem)
        if args.sl_nodelist is not None:
            additional_parameters['nodelist'] = args.sl_nodelist
        if args.sl_gpu_mem_gb is not None:
            additional_parameters['mem_per_gpu'] = args.sl_gpu_mem_gb
        print(additional_parameters)

        executor.update_parameters(slurm_additional_parameters=additional_parameters)
        with executor.batch():
            for _ in range(args.sl_node):
                if resubmit:
                    # job = executor.submit(Worker(), **{'func': func, 'args': func_args, 'kwargs': func_kwargs})
                    job = executor.submit(Worker(), **{'func': func, 'args': func_args, 'kwargs': func_kwargs})
                else:
                    job = executor.submit(func, *func_args, **func_kwargs)
                # print('run job in ', save_dir, job.job_id)
    else:
        print(func_args, )
        func(*func_args, **func_kwargs)
        job = None
    return job

    
def add_slurm_args(arg_parser):
    arg_parser.add_argument(
        "--slurm",
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument("--sl_time",default=48, type=int, help='timeout in hours')  # in ours hrs
    arg_parser.add_argument("--sl_name", default='dev', type=str)  
    arg_parser.add_argument("--sl_gpu_mem_gb", default=None, type=str)  
    arg_parser.add_argument("--sl_dir", default='/private/home/yufeiy2/slurm_cache_shot', type=str)  
    arg_parser.add_argument("--sl_work",default=10, type=int)
    arg_parser.add_argument("--sl_node",default=1, type=int)  # 16 hrs
    arg_parser.add_argument("--sl_nodelist",default=None, type=str)  # 16 hrs
    arg_parser.add_argument("--sl_mem",default=None, type=int)  # 16 hrs
    arg_parser.add_argument("--sl_ngpu",default=1, type=int)
    arg_parser.add_argument("--sl_ntask_pnode",default=1, type=int)
    arg_parser.add_argument("--sl_part",default='devaccel,learnaccel,learnfair', type=str)
    return arg_parser


def interactive_node():
    parser = ArgumentParser()
    parser = add_slurm_args(parser)
    args = parser.parse_args()
    func = time.sleep
    slurm_wrapper(args, args.sl_dir, func, {}, True, (args.sl_time*3600, ))
    return 

if __name__ == '__main__':
    interactive_node()
