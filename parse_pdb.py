import numpy as np
import argparse
import MDAnalysis as mda
from MDAnalysis.analysis import align
import os

def write_coords(fname, coords):

    if os.path.exists(fname):
        os.system(f"rm {fname}")

    n_atoms = coords.shape[1]

    with open(fname, "a") as f:
        f.write("{cols}\n".format(cols=n_atoms))
        np.savetxt(f, coords/10, fmt='%.4f')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, help="Ref pdb (for alignment/image gen)")
    parser.add_argument("--sys", type=str, help="System pdb (for simulations)", default=None)
    args = parser.parse_args()

    ref_pdb = "data/input/" + args.ref
    sys_pdb = args.sys

    ref_uni = mda.Universe(ref_pdb)

    if (sys_pdb is not None): 
        
        sys_pdb = "data/input/" + sys_pdb
        fname = sys_pdb.replace(".pdb", ".txt")
        sys_uni = mda.Universe(sys_pdb)

        # Center both universes
        ref_uni.atoms.translate(-ref_uni.select_atoms("all").center_of_mass())
        sys_uni.atoms.translate(-sys_uni.select_atoms("all").center_of_mass())

        # Align universes
        align.alignto(sys_uni, ref_uni, select="name CA", match_atoms=True)

        # Write coordinates
        write_coords(fname, sys_uni.select_atoms("name CA").positions.T)
        

    else: 
        fname = ref_pdb.replace(".pdb", ".txt")

        # Center ref universe
        ref_uni.atoms.translate(-ref_uni.select_atoms("all").center_of_mass())

        # Write coordinates
        write_coords(fname, ref_uni.select_atoms("name CA").positions.T)

        # ref_uni.select_atoms("name CA").write("cent_holo_groel_CA.pdb")
        


if __name__ == "__main__":

    main()