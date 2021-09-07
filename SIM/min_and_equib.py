import argparse, sys
parser = argparse.ArgumentParser()
parser.add_argument("--input_pdb", help="full path to the input pdb for simulation",required=True)
parser.add_argument("--n_proc", help="number of threads to use",required=True,type=int)
parser.add_argument("--n_rep", help="number of replicas to use",required=True,type=int)
parser.add_argument("-p", "--system_prefix", help="Prefix for .gro files", default="conf")

args = parser.parse_args()
pdb = args.input_pdb
n_proc = int(args.n_proc)
n_rep = int(args.n_rep)
prefix = args.system_prefix

import os
# Load PDB
os.system(f"printf \"6\n1\" | gmx_mpi pdb2gmx -f {pdb} -o {prefix}.gro -ignh")

# Set-up simulation box and solvent
os.system(f"gmx_mpi editconf -f {prefix}.gro -bt dodecahedron -d 1.0 -o {prefix}_box.gro")
os.system(f"printf \"0\n0\" | gmx_mpi trjconv -f {prefix}_box.gro -s {prefix}.gro -fit translation -o {prefix}_box_oriented.gro")
os.system(f"gmx_mpi solvate -cp {prefix}_box_oriented.gro -cs spc216.gro -p topol.top -o {prefix}_water.gro")

# Run Equilibration MD
os.system(f"gmx_mpi grompp -f em_2016.mdp -c {prefix}_water.gro -o em.tpr -maxwarn 3")
os.system(f"mpirun -n {n_proc} gmx_mpi mdrun -c {prefix}_em.gro -deffnm em -ntomp 1")

# Run NPT MD
os.system(f"gmx_mpi grompp -f npt_2016.mdp -c {prefix}_em.gro -o npt.tpr -maxwarn 3")
os.system(f"mpirun -n {n_proc} gmx_mpi mdrun -c {prefix}_npt.gro -nsteps 250000 -deffnm npt -ntomp 1")

# Run equil NVT MD
os.system(f"gmx_mpi grompp -f nvt_2016_EQUIL.mdp -c {prefix}_npt.gro -o nvt.tpr -maxwarn 3")
os.system(f"mpirun -n {n_proc} gmx_mpi mdrun -c {prefix}_nvt.gro -nsteps 1000000 -deffnm nvt -ntomp 1")

# Extract a random configuration during the NVT simulation
os.system(f"echo 0 | gmx_mpi trjconv -f traj.trr -s topol.tpr -dump 0 -o {prefix}_MD.gro")
os.system(f"echo 0 | gmx_mpi trjconv -f {prefix}_box_oriented.gro -s {prefix}_box_oriented.gro -o structure.pdb")
