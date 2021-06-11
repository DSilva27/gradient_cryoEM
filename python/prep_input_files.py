import numpy as np

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

### Change this parameters as you see fit

number_pixels = 124
pixel_size = 1.

ctf_env = 1.0

#These will only be used if you are generating images
ctf_defocus_min = 1.0
ctf_defocus_max = 3.0
#********

ctf_amplitude = 0.1

sigma = 1
sigma_reach = 3


#LOAD PARAMETERS INTO FILE
params_file = open("data/input/parameters.txt","w")
params_file.write(f"""NUMBER_PIXELS {number_pixels}
PIXEL_SIZE {pixel_size}
CTF_ENV {ctf_env}
CTF_DEFOCUS {ctf_defocus_min} {ctf_defocus_max}
CTF_AMPLITUDE {ctf_amplitude}
SIGMA {sigma}
SIGMA_REACH {sigma_reach}
""")
params_file.close()


