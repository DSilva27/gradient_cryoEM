import numpy as np
import argparse

'''
This .py is used to create the parameters used, which will be in the parameters.txt file and
in the quaternions.txt file.
Here you can create the files with the parameters you'd like.
'''


#Create the parameters file
'''
Explanation of the parameters
Grid Parameters
* NUMBER_PIXELS: quantity of pixels for the grid were the system will be projected
* PIXEL_SIZE: relation between pixels and distance units, e.g: a pixel is equivalent to 1.77 nm
CTF parameters
* CTF_ENV: b parameter for the envelope function defined as env(s) = exp(- b * s^2 / 2)
* CTF_DEFOCUS: the synthetic image will have a random defocus, this parameters defines the lower and
               upper limits for said random defocus.
* CTF_AMPLITUDE: 
Modeling with gaussians parameters
* SIGMA: standard deviation for the gaussians with which we model the atoms
* SIGMA_REACH: cut-off for neighbors, only atoms within a n*sigma are considered neighbors
'''


class param_device:

    def __init__(self):
        pass


    def load_params(self, args):

        self.n_pixels = args.n_pixels
        self.pixel_size = args.pixel_size
        self.sigma = args.sigma
        self.neigh_cutoff = args.neigh_cutoff

        #For the moment these aren't needed
        self.ctf_env = 1.0 #args.ctf_env
        self.ctf_defocus_min = 1.0 #args.def_min
        self.ctf_defocus_max = 3.0 #args.def_max
        self.ctf_amp = 0.1 #args.ctf_amp


    def write_param_file(self, print_p=False):

        #LOAD PARAMETERS INTO FILE
        params_file = open("data/input/parameters.txt","w")
            
        string = (
            f"NUMBER_PIXELS {self.n_pixels}\n"
            f"PIXEL_SIZE {self.pixel_size}\n"
            f"CTF_ENV {self.ctf_env}\n"
            f"CTF_DEFOCUS {self.ctf_defocus_min} {self.ctf_defocus_max}\n"
            f"CTF_AMPLITUDE {self.ctf_amp}\n"
            f"SIGMA {self.sigma}\n"
            f"SIGMA_REACH {self.neigh_cutoff}\n")
        
        params_file.write(string)
        params_file.close()

        if print_p:
            print("######## SETTING PARAMETERS AS ##############")
            print(string)


    #For annealing
    def update_sigma(self, new_sigma): 

        self.sigma = new_sigma
        self.write_param_file()




