import numpy as np
import MDAnalysis as mda
from multiprocessing import Pool
import os
import time
import argparse


def gen_img(index):

        os.system(f"./gradcv.out {index} -gen -nqt > /dev/null 2>&1")
        return 1

class img_generator:

    def __init__(self):
        pass

    def load_args(self, args):

        self.ref_pdb = "data/input/" + args.ref_pdb
        self.ref_universe = mda.Universe(self.ref_pdb)
        
        self.n_proc = args.n_proc
        self.n_img = args.n_img
        
        
    ##\brief Perform rmsd alignment using a reference    
    def prep_system(self):
        

        self.system_atoms = self.ref_universe.select_atoms('name CA').positions - \
                      self.ref_universe.atoms.center_of_mass()

        # Rotate the coordinates
        self.system_atoms = self.system_atoms.T
        self.n_atoms = self.system_atoms.shape[1]

        # Save the coordinates
        if os.path.exists("data/input/coord.txt"):
            os.system("rm data/input/coord.txt")

        with open("data/input/coord.txt", "w") as f:
            f.write("{cols}\n".format(cols=self.n_atoms)) 
            np.savetxt(f, self.system_atoms/10, fmt='%.4f')
        
        
    def gen_db(self):

        # Used to run Pool
        indexes = np.array(range(0, self.n_img))

        p = Pool(self.n_proc)

        print(f"Generating {len(indexes)} images...")
        start = time.time()
        _ = p.map(gen_img, indexes)
        print("... done. Run time: {}".format(time.time() - start))

