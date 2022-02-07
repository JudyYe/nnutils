import os

def slurm_wrapper(args, save_dir, func, func_kwargs):
    if args.slurm:
        import submitit
        executor = submitit.AutoExecutor(folder=save_dir)
        executor.update_parameters(
            tasks_per_node=args.sl_ntask_pnode,
            timeout_min=args.sl_time,
            slurm_partition=args.sl_part,
            gpus_per_node=args.sl_ngpu,
            cpus_per_task=args.sl_ngpu * 10,
            slurm_job_name=os.path.basename(save_dir),
        )
        print('slurm cache in ', save_dir)
        with executor.batch():
            for _ in range(args.sl_node):
                job = executor.submit(func, **func_kwargs)
    else:
        func(**func_kwargs)

    
def add_slurm_args(arg_parser):
    arg_parser.add_argument(
        "--slurm",
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument("--sl_time",default=1080, type=int)  # 16 hrs
    arg_parser.add_argument("--sl_work",default=10, type=int)
    arg_parser.add_argument("--sl_node",default=1, type=int)  # 16 hrs
    arg_parser.add_argument("--sl_ngpu",default=1, type=int)
    arg_parser.add_argument("--sl_ntask_pnode",default=1, type=int)
    arg_parser.add_argument("--sl_part",default='devlab', type=str)
    return arg_parser


