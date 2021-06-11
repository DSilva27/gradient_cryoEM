from Bio import PDB
import numpy as np
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

class optimizer:

    def __init__(self, n_proc, n_img, ref_pdb, system_pdb):

        self.n_proc = n_proc
        self.n_img = n_img
        self.ref_pdb = ref_pdb
        
        self.ref_universe = mda.Universe(self.ref_pdb)

        if system_pdb == "random":
            self.gen_random_system()
        
        else:
            self.system_pdb = system_pdb
            self.system_universe = mda.Universe(self.system_pdb)
            self.align_system()
        
        
    def gen_random_system(self):

        ref_coords = self.ref_universe.select_atoms('name CA').positions - \
                     self.ref_universe.atoms.center_of_mass()

        random_coords = 2 * np.random.random(ref_coords.shape) - 1

        random_coords[:,0] *= np.max(ref_coords[:,0])
        random_coords[:,1] *= np.max(ref_coords[:,1])
        random_coords[:,2] *= np.max(ref_coords[:,2])

        self.n_atoms = ref_coords.shape[0]
        self.system_atoms = random_coords.T
        print(self.system_atoms.shape)
        
        
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
        print(self.system_atoms.shape)

    def accumulate_gradient(self):

        self.grad = np.zeros((3, self.n_atoms))
        self.s_cv = 0

        for i in range(self.n_img):

            s, g = read_grads(i)
            self.s_cv += s
            self.grad += g


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

        #print(f"Calculating gradient for {len(indexes)} images...")
        #start = time.time()
        _ = p.map(run_cpp, indexes)
        #print("... done. Run time: {}".format(time.time() - start))
        p.close()

        #self.accumulate_gradient()

        
    ## \brief Perform gradient descent
    # Longer description (TODO)
    # \param n_steps How many steps of gradient descent will be done
    # \param alpha Learning rate
    # \param img_stride An image will be printed every img_stride steps
    def grad_descent(self, n_steps, alpha, img_stride):
        
        img_counter = 0
        for i in range(n_steps):
            
            #Calculate the gradient
            self.calc_gradient()
            self.accumulate_gradient()

            #Update the coordinates
            self.system_atoms -= alpha * self.grad

            #Print images
            if i%img_stride == 0:
                os.system(f"./gradcv.out {self.n_img + img_counter} -gen -no > /dev/null 2>&1")
                img_counter +=1

            
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
    #system_pdb = "data/input/" + args.system_pdb
    system_pdb = "random"

    opt = optimizer(n_proc, n_img, ref_pdb, system_pdb)
    opt.grad_descent(100, 1000, 10)


if __name__ == '__main__':
    main()