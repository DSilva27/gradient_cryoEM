import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os


def center_system(sys):

    sys.atoms.translate(-sys.select_atoms('all').center_of_mass())

    return sys

def add_noise(img, n_pixels, pixel_size, snr):

    rad_sq = (pixel_size * (n_pixels + 1)*0.5)**2

    grid_min = -pixel_size * (n_pixels - 1)*0.5
    grid_max = pixel_size * (n_pixels - 1)*0.5 + pixel_size

    grid = np.arange(grid_min, grid_max, pixel_size)

    mask = grid[None,:]**2 + grid[:,None]**2 < rad_sq

    var = np.std(img[mask])
    noise = np.random.normal(loc=0.0, scale = var, size=img.shape)

    img_noise = img + noise*snr

    var = np.std(img_noise)

    return img_noise/var

def calc_img(coord, n_pixels, pixel_size, sigma):
    
    n_atoms = coord.shape[1]
    norm =  (2 * np.pi * sigma**2 * n_atoms)

    grid_min = -pixel_size * (n_pixels - 1)*0.5
    grid_max = pixel_size * (n_pixels - 1)*0.5 + pixel_size

    x_grid = np.arange(grid_min, grid_max, pixel_size)
    y_grid = np.arange(grid_min, grid_max, pixel_size)

    gauss = np.exp( -0.5 * ( ((x_grid[:,None] - coord[0,:])/sigma)**2) )[:,None] * \
            np.exp( -0.5 * ( ((y_grid[:,None] - coord[1,:])/sigma)**2) )

    Icalc = gauss.sum(axis=2)

    return Icalc/norm

def print_image(fname, img, quat):

    if os.path.exists(fname): os.system(f"rm {fname}")

    with open(fname, "w") as f:
        f.write(f"0.0\n{quat[0]}\n{quat[1]}\n{quat[2]}\n{quat[3]}\n")
        np.savetxt(f, img)

    return

def print_coords(fname, coords):

    if os.path.exists(fname): os.system(f"rm {fname}")

    with open(fname, "w") as f:
        f.write(f"{coords.shape[1]}\n")
        np.savetxt(f, coords)

    return

def create_images(r_coord, quats, *args, prefix="img_", print_imgs=True):

    N_PIXELS, PIXEL_SIZE, SIGMA, SNR = args
    dataset = []

    for i, q in enumerate(quats):

        rot = R.from_quat(q)
        rot_mat = rot.as_matrix()
        rot_mat_inv = rot.inv().as_matrix()

        r_rot = np.matmul(rot_mat, r_coord)
        Iexp = calc_img(r_rot, N_PIXELS, PIXEL_SIZE, SIGMA)

        if (print_imgs): print_image(f"{prefix}{i}.txt", Iexp, q)
        img = {"Q": rot_mat, "Q_inv":rot_mat_inv, "I": Iexp}
        dataset.append(img)

    return dataset


############################################# LOAD YOUR COORDINATES HERE ####################################################
'''
Here are a few examples on how to load your coordinates

################################################## FROM PDB FILES ############################################################
ref_system_mda = mda.Universe("ref_filename.pdb") # The system used to create the experimental images (always needed!)
ref_system_mda = center_system(ref_system_mda) # Center your coordinates if they are not centered

select_filter = "not name *H*" # Change if needed (defines which atoms are selected from the pdb)
ref_sys = ref_system_mda.select_atoms(select_filter).positions.T # This returns a numpy array of shape (3, # atoms)


################################################## FROM TXT FILES ############################################################
# Since these txt files usually come from the function print_coords() used in  this code, I will assume your .txt file is 
# formatted as such. Be careful if using different .txt files, remember that the shape of the coordinate array must be (3, # atoms)

ref_sys = np.loadtxt("ref_filename.txt", skiprows=1)

YOU NEED A ref_sys VARIABLE BEFORE YOU EXIT THIS PART OF THE CODE
'''
##############################################################################################################################

# np.random.seed(0) # Uncomment if doing debugging

# Print coordinates for c++
# Will be needed in the future
# COORD_FILE = "ref_sys.txt" # Change if needed
# print_coords(COORD_FILE, ref_sys)

# TODO create a function that loads a set of quaternions and creates a dataset
# TODO integrate wrapper with c++ function to create images 
# Create quaternions
quat = np.zeros((3, 4))
quat[0] = R.from_euler("x", 0, degrees=True).as_quat()
quat[1] = R.from_euler("x", 90, degrees=True).as_quat()
quat[2] = R.from_euler("y", 90, degrees=True).as_quat()

# Set image parameters
N_PIXELS = 32
PIXEL_SIZE = 0.5
SIGMA = 0.5
SNR = 0.0 # Unused at the moment
IMG_PFX = "pfx_img_" #images will be called pfx_img_0.txt, pfx_img_1.txt, ...
N_IMGS = 3 # Only used when creating images with random quaternions (to be implemented)

# Create/Load images
# TODO Replace this function with the c++ code
dataset = create_images(ref_sys, quat, N_PIXELS, PIXEL_SIZE, SIGMA, SNR, IMG_PFX, True) # images created = # quats

# Print initial images 
# fig, ax_imgs = plt.subplots(1, 3, figsize=(16,9))
# ax_imgs[0].imshow(dataset[0]["I"])
# ax_imgs[1].imshow(dataset[1]["I"])
# ax_imgs[2].imshow(dataset[2]["I"])
# plt.show()