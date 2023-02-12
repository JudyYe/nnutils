import subprocess
import os
import time
from typing import Any
import submitit
from argparse import ArgumentParser

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
        additional_parameters = {"signal": 'SIGUSR2@30'}
        
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
            job._interrupt(timeout=True)        
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
    def __call__(self, func, args, kwargs):
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
    arg_parser.add_argument("--sl_dir", default='/home/yufeiy2/slurm_cache_shot', type=str)  
    arg_parser.add_argument("--sl_work",default=10, type=int)
    arg_parser.add_argument("--sl_node",default=1, type=int)  # 16 hrs
    arg_parser.add_argument("--sl_nodelist",default=None, type=str)  # 16 hrs
    arg_parser.add_argument("--sl_mem",default=None, type=int)  # 16 hrs
    arg_parser.add_argument("--sl_ngpu",default=1, type=int)
    arg_parser.add_argument("--sl_ntask_pnode",default=1, type=int)
    arg_parser.add_argument("--sl_part",default='abhinavlong,shubhamlong,all', type=str)
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
