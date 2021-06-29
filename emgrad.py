import numpy as np
import pandas as pd
import os
import json
import time
import argparse

#import MDAnalysis as mda
#from MDAnalysis.analysis import align
#from MDAnalysis.analysis.rms import rmsd

from multiprocessing import Pool

import grad_lib as GL


# parser.add_argument("--n_proc", help="number of processors used. Default=1", default=1, type=int)
# parser.add_argument("--n_img", help="number of images to be compared", required=True, type=int)
# parser.add_argument("--ref_pdb", help="name of the input pdb (do not include path)", required=True)

def set_params(parser):

    parser.add_argument("-np", "--n_pixels", help="Number of pixels for projection", type=int)
    parser.add_argument("-ps", "--pixel_size", help="Pixel size for projection", type=float)
    parser.add_argument("-s", "--sigma", help="sigma used in gaussians for projection", type=float)
    parser.add_argument("-nc", "--neigh_cutoff", help="Neighbor cutoff for projection", type=int)

    args = parser.parse_args()

    myparam_device = GL.param_device()
    myparam_device.load_params(args)
    myparam_device.write_param_file(True)

def gen_db(parser):

    parser.add_argument("--n_proc", help="number of processors used. Default=1", default=1, type=int)
    parser.add_argument("--n_img", help="number of images to be compared", required=True, type=int)
    parser.add_argument("--ref_pdb", help="name of the input pdb (do not include path)", required=True)

    args = parser.parse_args()

    myimg_gen = GL.img_generator()
    myimg_gen.load_args(args)
    myimg_gen.prep_system()
    myimg_gen.gen_db()

def calc_grad(parser):

    parser.add_argument("--n_proc", help="number of processors used. Default=1", default=1, type=int)
    parser.add_argument("--n_img", help="number of images to be compared", required=True, type=int)
    parser.add_argument("--ref_pdb", help="name of the reference (for alignment) pdb (do not include path)", required=True)
    parser.add_argument("--system_pdb", help="name of the system pdb (do not include path)", required=True)

    args = parser.parse_args()

    mygrad_calc = GL.gradient_calculator()
    mygrad_calc.load_args(args)
    mygrad_calc.calc_gradient()


def grad_desc(parser):

    parser.add_argument("--n_proc", help="number of processors used. Default=1", default=1, type=int)
    parser.add_argument("--n_img", help="number of images to be compared", required=True, type=int)
    parser.add_argument("--ref_pdb", help="name of the reference (for alignment) pdb (do not include path)", required=True)
    parser.add_argument("--system_pdb", help="name of the system pdb (do not include path)", required=True)
    
    #Gradient descent
    parser.add_argument("--n_steps", help="max steps for gradient descent", default=100, type=int)
    parser.add_argument("--learn_rate", help="learning rate for gradient descent", default=100, type=float)
    parser.add_argument("--stride", help="stride for writing into traj file", default=1, type=int)
    parser.add_argument("--tol", help="stop the descent if the change in the cv is less than tol", default=1e-3, type=float)

    args = parser.parse_args()

    mygrad_desc = GL.gradient_descent()
    mygrad_desc.load_args(args)
    mygrad_desc.grad_descent()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("program", type=str, help="Program you want to use, e.g, gen_img")

    args, _ = parser.parse_known_args()

    if args.program == "set_params":
        set_params(parser)
    
    elif args.program == "gen_db":
        gen_db(parser)

    elif args.program == "calc_grad":
        calc_grad(parser)

    elif args.program == "grad_desc":
        grad_desc(parser)

    else:
        raise NameError(args.program + " has not been defined. Please check the documentation.")

if __name__ == '__main__':
    main()