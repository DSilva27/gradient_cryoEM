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


class gradient_calculator:

    def __init__(self):
        pass

    def load_args(self, args):

        self.n_proc = args.n_proc
        self.n_img = args.n_img
        self.ref_pdb = "data/input/" + args.ref_pdb
        
        self.ref_universe = mda.Universe(self.ref_pdb)

        if args.system_pdb == "random":
            self.gen_random_system()
        
        else:
            self.system_pdb = "data/input/" + args.system_pdb
            self.system_universe = mda.Universe(self.system_pdb)
            self.align_system()
    
        
    def gen_random_system(self):

        ref_coords = self.ref_universe.select_atoms('name CA').positions - \
                     self.ref_universe.atoms.center_of_mass()

        random_coords = ref_coords + np.random.normal(0, 1, ref_coords.shape)

        self.n_atoms = ref_coords.shape[0]
        self.system_atoms = random_coords.T
        self.ref_coords = ref_coords
        
        
    def align_system(self):
        
        '''
        Returns the rotation matrix that minimizes the rmsd between two molecules
        
        It uses only the CA atoms
        '''

        system_ca = self.system_universe.select_atoms('name CA').positions - \
                    self.system_universe.atoms.center_of_mass()

        reference_ca = self.ref_universe.select_atoms('name CA').positions - \
                       self.ref_universe.atoms.center_of_mass()

        rot_mat, _ = align.rotation_matrix(system_ca, reference_ca)

        # Extract atomic positions
        system_atoms = self.system_universe.select_atoms('name CA').positions

        # Define the origin as the center of mass
        system_atoms -= self.system_universe.atoms.center_of_mass()

        # Rotate the coordinates
        self.system_atoms = np.dot(rot_mat, system_atoms.T)
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
            np.savetxt(f, self.system_atoms, fmt='%.4f')

        
        ### RUNNING C++
        indexes = np.array(range(0, self.n_img))
        p = Pool(self.n_proc)

        print(f"Calculating gradient for {len(indexes)} images...")
        start = time.time()
        _ = p.map(run_cpp, indexes)
        print("... done. Run time: {}".format(time.time() - start))
        p.close()

        self.accumulate_gradient()

        if self.n_img > 1:
            self.write_results()
