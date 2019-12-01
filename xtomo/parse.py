import argparse
import json
#import textwrap

#print('\033[1m' + 'Hello parse')

bold='\033[1m'
endb= '\033[0m'
blue='\033[94m'

ap = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=bold+'+'+'-'*61+"+\n|"+
            " option precedence (highest on the left):                    |\n|"+blue+
            " individual options, 'opts' dictionary, 'fopts' file options"+endb+bold+" |\n"+'+'+'-'*61+'+')


ap.add_argument("-f", "--file_in",  type = str, help="h5 file in")
ap.add_argument("-o", "--file_out", default='0',  type = str, help="file out, default 0, 0: autogenerate name, -1: skip saving")
ap.add_argument("-rot_center", "--rot_center",  type = int, help="rotation center, int ")
ap.add_argument("-a", "--algo",  type = str, help="algorithm: 'iradon' (default), 'sirt', 'cgls', 'tv', 'tvrings' 'tomopy-gridrec'")
ap.add_argument("-G", "--GPU",  type = int, help="turn on GPU, bool")
ap.add_argument("-S", "--shmem",  type = int, help="turn on shared memory MPI, bool")
ap.add_argument("-maxiter", "--maxiter", type = int, help="maxiter, default 10")
ap.add_argument("-tol", "--tol", type = float, help="tolerance, default 5e-3")
ap.add_argument("-max_chunk", "--max_chunk_slice",  type = int, help="max chunks per mpi rank")
ap.add_argument("-chunks", "--chunks",  type = int, nargs='+', help="chunks to reconstruct")

ap.add_argument("-reg", "--reg",  type = float, help="regularization parameter")
ap.add_argument("-tau", "--tau", type = float, help="soft thresholding parameter")
ap.add_argument("-v", "--verbose",   type = float, help="verbose float between (0-1), default 1")
ap.add_argument("-sim", "--simulate",  type = bool, help="use simulated data, bool")
ap.add_argument("-sim_shape", "--sim_shape",  type = int,nargs='+', help="simulate shape nslices,nangles,nrays")
ap.add_argument("-sim_width", "--sim_width",  type = int, help="object width between 0-1")
ap.add_argument('-opts', '--options', type=json.loads, help="e.g. \'{\"algo\":\"iradon\", \"maxiter\":10, \"tol\":1e-2, \"reg\":1, \"tau\":.05} \' ")
ap.add_argument('-fopts', '--foptions', type=str, help="file with json options  ")
ap.add_argument("-ncore", "--ncore", type=int, help="ncore for tomopy reconstruction algorithms")

ap.add_argument("-rb", "--ring_buffer", type=int, default=0, help="input ring buffer  1=true, 0=false")

 
Dopts={ 'algo':'iradon', 'maxiter':10, 'shmem':True, 'GPU':True, 
       'max_chunk_slice':16, 'verbose':True, 'tol':5e-3}
Dopts['sim_shape']=[128, 181, 256] 
Dopts['sim_width']=.95


#sim_shape=[256, 181, 256]
#sim_width=0.95
global args

args = vars(ap.parse_args())
#print("hello parser", args)

opts = args['options']

if args['foptions']!=None:
    fopts = json.load(open(args['foptions'],'r'))
    Dopts.update(fopts)

if opts!=None:
    Dopts.update(opts)

for key in Dopts:
    if args[key]==None: args[key]=Dopts[key]
if args['file_in']==None: args['simulate']=True


def main():
    global args
    return args

#
#if __name__ == '__main__':
#    main()
