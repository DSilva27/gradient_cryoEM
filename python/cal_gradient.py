from Bio import PDB
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import os
import json
import time
import argparse

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd


#Define the functions that will be used

def center_atomic_coord (x,y,z):
    
    x, y, z = x-np.mean(x), y-np.mean(y), z-np.mean(z)
    return(x, y, z) 

def quaternion_rotation(q, coord):
    
    '''
    Performs a rotation using quaternions.
    
    If it is based from a quaternion q1 = w + xi + yj + zk, then the 
    quaternion array q should be q = [x, y, z, w]
    '''
    
    Q = R.from_quat(q).as_matrix()
    rot_coord = np.dot(Q, coord)
    
    return rot_coord

def read_grads():
    # Retrieve gradients of each training step fron the text output files
    # from the c++ program

    with open("data/output/grad.json") as j_file:
        data = json.load(j_file)[0]
        s = data["s"]
        x_grad = np.array(data['sgrad_x']) 
        y_grad = np.array(data['sgrad_y'])
        z_grad = np.array(data['sgrad_z'])

        return s, np.array([x_grad, y_grad, z_grad])
    
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

##ADVICE THESE SHOULD BE INPUTS THAT THE USER CAN MODIFY

parser = argparse.ArgumentParser()

parser.add_argument("--ref_pdb", help="name of the input pdb (do not include path)", required=True)
parser.add_argument("--system_pdb", help="name of the system pdb in the current md_step pdb (do not include path)", required=True)
args = parser.parse_args()

#Set the pdb's with their paths
ref_pdb = '../data/input/' + args.ref_pdb
system_pdb = '../data/input/' + args.system_pdb


#Importat 1xck's PDB to extract XYZ atomic coordinates
ref_universe = mda.Universe(ref_pdb)
system_universe = mda.Universe(system_pdb)

#calculate the rotation matrix
rot_matrix = aligment_rotation_matrix(ref_universe, system_universe)

#extract the positions
system_atoms = system_universe.select_atoms('all').positions

#Define the origin as the center of mass
system_atoms -= system_universe.atoms.center_of_mass()

#Rotate the coordinates
system_atoms_aligned = np.dot(rot_matrix, system_atoms.T)

##ADVICE THE QUATERNIONS SHOULD BE INPUTS THAT THE USER CAN MODIFY
#Quaternion parameters
'''
If it is based from a quaternion q1 = w + xi + yj + zk, then the 
quaternion array q should be q = [x, y, z, w]
'''

q = np.loadtxt("data/input/quaternions.txt")

#Rotate them with quaternions
system_atoms_aligned = quaternion_rotation(q, system_atoms_aligned)

### ADVICE: WE HAVE TO BE CAREFUL BECAUSE IF THERE ARE MULTPLE IMAGES WE
### WOULD OVER-WRITE THE FILES, MAYBE BETTER TO BE ABLE TO CHANGE ITS NAME 
### HERE AND IN THE C++?  

# Save the coordinates
if os.path.exits("data/input/coord.txt"):
    os.system("rm ../data/input/coord.txt")

with open("data/input/coord.txt", "a") as f:
    f.write("{cols}\n".format(cols=system_atoms_aligned.shape[1]))
    np.savetxt(f, system_atoms_aligned, fmt='%.4f')

### RUNNING C++
start = time.time()
os.system("cd .. && ./gradcv.out");
print("Projection time: {}".format(time.time() - start))

# Created files

#Inoctf.txt -> projected image without ctf and gaussian noise
#Ictf_noise.txt -> projected image with ctf and gaussian noise
#grad.json -> .json file where the cv and its gradient are stored

### ADVICE: PUT HERE THE FUNCTION THAT WRITES OUT THE ROTATED GRADIENTS. TELL THE USER WHERE THEY ARE
### THE CALCULATED IMAGE LEAVE IT AS AN INPUT OPTION DO NOT PRINT IT OUT ALWAYS

s, grad = read_grads()

grad_rot = np.dot(rot_matrix.T, grad)