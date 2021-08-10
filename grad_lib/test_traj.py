import numpy as np
import pandas as pd
import os
import json
import time
import argparse

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd

from multiprocessing import Pool


def read_grads(index):
    # Retrieve gradients of each training step fron the text output files
    # from the c++ program

    with open(f"data/output/grad_{index}.json") as j_file:
        data = json.load(j_file)[0]
        s = data["s"]
        x_grad = np.array(data['sgrad_x']) 
        y_grad = np.array(data['sgrad_y'])
        z_grad = np.array(data['sgrad_z'])

    return s, np.array([x_grad, y_grad, z_grad])

def run_cpp(index):

        os.system(f"./gradcv.out {index} -grad > /dev/null 2>&1")
        return 1

def rmsd(r1, r2, N):
    R = np.sqrt(np.sum(np.sum((r1 - r2)**2, axis=0)) / N)

    return R

class test_traj:

    def __init__(self):
        pass

    def load_args(self, args):

        self.n_proc = args.n_proc
        self.n_img = args.n_img
        self.ref_pdb = "data/input/" + args.ref_pdb
        
        self.ref_universe = mda.Universe(self.ref_pdb)

        self.system_pdb = "data/input/" + args.system_pdb
        self.system_traj = "data/input/" + args.system_traj
        self.system_universe = mda.Universe(self.system_pdb, self.system_traj)
        #self.align_system()
        self.system_atoms = self.system_universe.select_atoms('name CA').positions.T.copy()
        self.n_atoms = self.system_atoms.shape[1]

        self.n_steps = len(self.system_universe.trajectory)
        
    def align_system(self):
        
        '''
        Returns the rotation matrix that minimizes the rmsd between two molecules
        
        It uses only the CA atoms
        '''

        # system_ca = self.system_universe.select_atoms('name CA').positions - \
        #             self.system_universe.atoms.center_of_mass()

        # reference_ca = self.ref_universe.select_atoms('name CA').positions - \
        #                self.ref_universe.atoms.center_of_mass()

        # rot_mat, _ = align.rotation_matrix(system_ca, reference_ca)

        # # Extract atomic positions
        # system_atoms = self.system_universe.select_atoms('name CA').positions.copy()

        # # Define the origin as the center of mass
        # system_atoms -= self.system_universe.atoms.center_of_mass()

        # Rotate the coordinates
        #self.system_atoms = np.dot(rot_mat, system_atoms.T)

        align.AlignTraj(self.system_universe,  # trajectory to align
                self.ref_universe,  # reference
                select='name CA',  # selection of atoms to align
                filename='data/input/traj_aligned.xtc',  # file to write the trajectory to
                match_atoms=False,  # whether to match atoms based on mass
               ).run()

        self.system_universe = mda.Universe(self.system_pdb, "data/input/traj_aligned.xtc")
        self.system_atoms = self.system_universe.select_atoms('name CA').positions.copy()
        self.n_atoms = self.system_atoms.shape[1]
        

    def accumulate_gradient(self):

        self.grad = np.zeros((3, self.n_atoms))
        self.s_cv = 0

        for i in range(self.n_img):

            s, g = read_grads(i)
            self.s_cv += s
            self.grad += g


    def write_results(self):

        res_dict = {
            "s" : self.s_cv,
            "sgrad_x": self.grad[0].tolist(),
            "sgrad_y": self.grad[1].tolist(),
            "sgrad_z": self.grad[2].tolist(),
        }

        #print(self.grad[0])
        with open("data/output/grad_all.json", "w") as json_file:
            json.dump(res_dict, json_file, indent=4)


    def calc_gradient(self):

        # Save the coordinates

        if os.path.exists("data/input/coord.txt"):
            os.system("rm data/input/coord.txt")

        with open("data/input/coord.txt", "a") as f:
            f.write("{cols}\n".format(cols=self.n_atoms))
            np.savetxt(f, self.system_atoms/10, fmt='%.4f')

        
        ### RUNNING C++
        # indexes = np.array(range(0, self.n_img))
        # p = Pool(self.n_proc)

        #print(f"Calculating gradient for {len(indexes)} images...")
        #start = time.time()
        # _ = p.map(run_cpp, indexes)

        run_cpp(0)
        #print("... done. Run time: {}".format(time.time() - start))
        # p.close()

        self.accumulate_gradient()

        if self.n_img > 1:
            self.write_results()

    

    ## \brief Perform gradient descent
    # Longer description (TODO)
    # \param n_steps How many steps of gradient descent will be done
    # \param alpha Learning rate
    # \param img_stride An image will be printed every img_stride steps
    def run_traj(self):
        
        with open("data/output/COLVAR", "w") as COLVAR:
                
            for i in range(0, self.n_steps, 10):
                
                #Update the coordinates
                self.system_universe.trajectory[i]
                self.system_atoms = self.system_universe.select_atoms('name CA').positions.T.copy()
                #self.align_system()

                #Calculate the gradient
                self.calc_gradient()
                self.accumulate_gradient()
                #self.write_results()
                COLVAR.write(f"{self.s_cv:.6f}\n")

                

            COLVAR.close()

 
def main():


    parser = argparse.ArgumentParser()

    parser.add_argument("--n_proc", help="number of processors used. Default=1", default=1, type=int)
    parser.add_argument("--n_img", help="number of images to be compared", required=True, type=int)
    parser.add_argument("--ref_pdb", help="name of the input pdb (do not include path)", required=True)
    parser.add_argument("--system_pdb", 
                        help="name of the system pdb in the current md_step pdb (do not include path)", 
                        required=True)

    args = parser.parse_args()
    
    n_proc = args.n_proc
    n_img = args.n_img
    ref_pdb = "data/input/" + args.ref_pdb

    if args.system_pdb != "random": system_pdb = "data/input/" + args.system_pdb    
    else: system_pdb = "random"


if __name__ == '__main__':
    main()