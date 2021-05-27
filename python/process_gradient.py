import json
import numpy as np
import argparse

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


def acc_grad_and_cv(n_atoms, n_img):

    grad = np.zeros((3, n_atoms))
    s_cv = 0
    for i in range(n_img):

        s, g = read_grads(i)[1]
        s_cv += s
        grad += g

    return s_cv, grad

    
    
parser = argparse.ArgumentParser()
parser.add_argument("--n_img", help="number of images to be compared", required=True, type=int)
parser.add_argument("--n_atoms", 
                    help="number of atoms in the system. Check the first line of data/input/coords.txt", 
                    required=True, type=int)