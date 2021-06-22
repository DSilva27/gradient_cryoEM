import numpy as np
import MDAnalysis as mda
from multiprocessing import Pool
import os
import time
import argparse

### MAIN CODE ###

# Parse flags
parser = argparse.ArgumentParser()
parser.add_argument("--n_proc", help="number of processors used. Default=1", default=1, type=int)
parser.add_argument("--n_img", help="number of images to be created", required=True, type=int)
parser.add_argument("--ref_pdb", help="name of the input pdb (do not include path)", required=True)

args = parser.parse_args()

# Read reference pdb
ref_pdb = "data/input/" + args.ref_pdb

# Create MD Universe
ref_universe = mda.Universe(ref_pdb)

# Extract the positions
system_atoms = ref_universe.select_atoms('all').positions

# Define the origin as the center of mass
system_atoms -= ref_universe.atoms.center_of_mass()

system_atoms = system_atoms.T
n_atoms = system_atoms.shape[1]

# Save the coordinates
if os.path.exists("data/input/coord.txt"):
    os.system("rm data/input/coord.txt")

with open("data/input/coord.txt", "a") as f:
    f.write("{cols}\n".format(cols=n_atoms)) 
    np.savetxt(f, system_atoms, fmt='%.4f')

### RUNNING C++

# Used to run Pool
indexes = np.array(range(0, args.n_img))

# Small function to generate an image
def gen_img(index):

    os.system(f"./gradcv.out {index} -gen > /dev/null 2>&1")
    return 1

p = Pool(args.n_proc)

print(f"Generating {len(indexes)} images...")
start = time.time()
_ = p.map(gen_img, indexes)
print("... done. Run time: {}".format(time.time() - start))