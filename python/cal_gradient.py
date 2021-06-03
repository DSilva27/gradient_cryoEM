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


def aligment_rotation_matrix(reference, system):
    
    '''
    Returns the rotation matrix that minimizes the rmsd between two molecules
    
    It uses only the CA atoms
    '''

    system_ca = system.select_atoms('name CA').positions - system.atoms.center_of_mass()
    reference_ca = reference.select_atoms('name CA').positions - reference.atoms.center_of_mass()
    Ra, rmsd = align.rotation_matrix(system_ca, reference_ca)

    return (Ra)



### MAIN CODE ###

# Parse flags
parser = argparse.ArgumentParser()

parser.add_argument("--n_proc", help="number of processors used. Default=1", default=1, type=int)
parser.add_argument("--n_img", help="number of images to be compared", required=True, type=int)
parser.add_argument("--ref_pdb", help="name of the input pdb (do not include path)", required=True)
parser.add_argument("--system_pdb", 
                    help="name of the system pdb in the current md_step pdb (do not include path)", 
                    required=True)

args = parser.parse_args()

# Set pdb file names
ref_pdb = "data/input/" + args.ref_pdb
system_pdb = "data/input/" + args.system_pdb

# Create MD Universe objects
ref_universe = mda.Universe(ref_pdb)
system_universe = mda.Universe(system_pdb)

# Calculate the rotation matrix
rot_matrix = aligment_rotation_matrix(ref_universe, system_universe)

# Extract atomic positions
system_atoms = system_universe.select_atoms('name CA').positions

# Define the origin as the center of mass
system_atoms -= system_universe.atoms.center_of_mass()

# Rotate the coordinates
system_atoms_aligned = np.dot(rot_matrix, system_atoms.T)

# Save the coordinates
n_atoms = system_atoms_aligned.shape[1]

if os.path.exists("data/input/coord.txt"):
    os.system("rm data/input/coord.txt")

with open("data/input/coord.txt", "a") as f:
    f.write("{cols}\n".format(cols=n_atoms))
    np.savetxt(f, system_atoms_aligned, fmt='%.4f')

### RUNNING C++

indexes = np.array(range(0, args.n_img))

def gen_img(index):

    os.system(f"./gradcv.out {index} -grad > /dev/null 2>&1")
    return 1

p = Pool(args.n_proc)

print(f"Calculating gradient for {len(indexes)} images...")
start = time.time()
_ = p.map(gen_img, indexes)
print("... done. Run time: {}".format(time.time() - start))


print(f"You should be set to run:\n python python/process_gradient.py --n_img {args.n_img} --n_atoms {n_atoms}")