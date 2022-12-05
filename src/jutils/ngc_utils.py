import subprocess
import re
import os
from argparse import ArgumentParser

# Template
# ngc batch run  \
# 	--preempt RUNONCE \
# 	--ace nv-us-west-2 \
#   --instance dgx1v.16g.1.norm \
# 	--image "nvidian/lpr/dexafford:conda_libgl" \
# 	--result /result \
# 	--org nvidian --team lpr  \
#   --workspace ws-judyye:ws-judyye:RW \
# 	--datasetid 102373:ho3d \
#   --total-runtime 3600s \
# 	--name "ml-model.res_dog" \
# 	--commandline ". /ws-judyye/conda/remote.sh; python -m jutils.ngc.res_dog" 


def add_ngc_args(arg_parser: ArgumentParser):
    arg_parser.add_argument("--slurm",action="store_true",)
    arg_parser.add_argument("--sl_dry",action="store_true",)
    arg_parser.add_argument("--sl_time",default=24, type=float, help='in hour')  # 16 hrs
    arg_parser.add_argument("--sl_ngpu",default=1, type=int)
    arg_parser.add_argument("--sl_mem",default=16, type=int)

    arg_parser.add_argument("--sl_name",default=None, type=str)
    arg_parser.add_argument("--sl_org",default='nvidian', type=str)
    arg_parser.add_argument("--sl_team",default='lpr', type=str)
    arg_parser.add_argument("--sl_ace",default='nv-us-west-2', type=str)
    arg_parser.add_argument("--sl_preempt", default='RUNONCE')

    arg_parser.add_argument("--sl_ws",default='ws-judyye', type=str)
    arg_parser.add_argument("--sl_ws_src",default='/ws/', type=str)
    arg_parser.add_argument("--sl_data",default='102373:ho3d+109663:obman', type=str)

    arg_parser.add_argument('--sl_pip', default='/ws-judyye/dex_afford/src/docs/jit_pip.sh')
    arg_parser.add_argument("--sl_image",default='nvidian/lpr/dexafford:all_in_docker', type=str)
    arg_parser.add_argument("--sl_result", default='/result/', type=str)  

    return arg_parser


def ngc_wrapper(args, name, core_cmd):
    cmd = 'ngc batch run '
    cmd += ' --preempt %s' % args.sl_preempt
    cmd += ' --ace %s' % args.sl_ace
    if args.sl_ngpu == 0:
        cmd += ' --instance cpu.x86.tiny'
    else:
        cmd += ' --instance dgx1v.%dg.%d.norm' % (args.sl_mem, args.sl_ngpu)
    cmd += ' --image %s' % args.sl_image
    cmd += ' --result %s ' % args.sl_result
    cmd += ' --org %s --team %s' % (args.sl_org, args.sl_team)
    cmd += ' --workspace %s:/%s:RW' % (args.sl_ws, args.sl_ws)
    for data in args.sl_data.split('+'):
        cmd += ' --datasetid %s' % data
    cmd += ' --total-runtime %ds' % int(args.sl_time * 3600)
    cmd += ' --name ml-model.%s' % name
    
    print(core_cmd)
    ws_dst = os.path.join('/', args.sl_ws, os.getcwd().split(args.sl_ws_src)[1])
    setup_cmd = '. %s; cd %s; ' % (args.sl_pip, ws_dst)
    cmd += r' --commandline "%s  %s"' % (setup_cmd, core_cmd)

    if not args.sl_dry:
        wrap_cmd(cmd)
    else:
        print(cmd)
    return


def ngc_engine():
    parser = ArgumentParser()
    add_ngc_args(parser)

    sl_args, unknown = parser.parse_known_args()

    print(unknown)
    # \$ -> \\\$
    unknown = [e.replace('$', '\\\\\\$') for e in unknown]  # because reading in will take 
    # print(unknown)
    if sl_args.sl_name is None:
        sl_args.sl_name = re.sub(r'[^A-Za-z0-9\.]+', '', '.'.join(unknown))[0:100]
    ngc_wrapper(
        sl_args, 
        sl_args.sl_name,  
        ' '.join(unknown))


class Worker():
    def __init__(self) -> None:
        pass

    def __call__(self, args):
        import importlib
        mod = importlib.import_module(args.worker)
        main_worker = getattr(mod, 'main_worker')
        main_worker(args)
        

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



class Executor:
    def __init__(self, folder, local_str='', remote_str='', cmd='') -> None:
        self.params = {}
        os.makedirs(folder, exist_ok=True)
        self.submit_dir = folder
        self.local_str = local_str
        self.remote_str = remote_str
        self.cmd = cmd

    def update_parameters(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'total_runtime': 
                self.params['total-runtime'] = v                    
                print(v)
            else:
                self.params[k] = v
    
    def submit(self, func, func_args):
        import hydra.utils as hydra_utils
        print(self.params)
        cmd = self._param_to_str()
        MAIN_PID = os.getpid()
        fname = os.path.join(self.submit_dir, '%d_submit.pkl' % MAIN_PID)

        ws_dst = hydra_utils.get_original_cwd().replace(self.local_str, self.remote_str)

        import pickle
        with open(fname , 'wb') as fp:
            pickle.dump({'func': func, 'args': func_args}, fp) 

        # write batch file
        setup_cmd = ' %s; cd %s; ' % (self.cmd, ws_dst)
        core_cmd = 'python -m jutils.ngc.exec_submit %s' % fname.replace(self.local_str, self.remote_str)
        cmd += ' --commandline "%s %s"' % (setup_cmd, core_cmd)
    
        os.makedirs(self.submit_dir, exist_ok=True)
        batch_file = self.submit_dir + '/%d_submit.sh' % MAIN_PID
        with open(batch_file, 'w') as wr_fp:
            wr_fp.write('echo HELLO\n')
            wr_fp.write('%s\n' % cmd)
            wr_fp.write('echo world\n')
        print('submitit! cache to %s' % batch_file)

        inner_cmd = '. %s' % batch_file
        wrap_cmd(inner_cmd)
    
    def _param_to_str(self, ):
        cmd = 'ngc batch run '        
        for k, v in self.params.items():
            print(k)
            if k == 'datasetid':
                for e in v.split('+'):
                    cmd += ' --datasetid %s ' % e
            else:
                cmd += ' --%s %s ' % (k, v)            
        print('ngc cmd', cmd)
        return cmd


if __name__ == '__main__':
    ngc_engine()
