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

def grad_lu2(coord, Icalc, I_exp, n_pixels, pixel_size, sigma):
    
    n_atoms = coord.shape[1]
    norm =  (2 * np.pi * sigma**2 * n_atoms)

    grid_min = -pixel_size * (n_pixels - 1)*0.5
    grid_max = pixel_size * (n_pixels - 1)*0.5 + pixel_size

    x_grid = np.arange(grid_min, grid_max, pixel_size)
    y_grid = np.arange(grid_min, grid_max, pixel_size)

    grad = np.zeros_like(coord)

    f2 = Icalc - I_exp
    f1 = np.exp( -0.5 * ( ((x_grid[:,None] - coord[0,:])/sigma)**2) )[:,None] * \
         np.exp( -0.5 * ( ((y_grid[:,None] - coord[1,:])/sigma)**2) )

    fx = (x_grid[:,None] - coord[0,:])[:, None] / sigma**2
    fy = (y_grid[:,None] - coord[1,:])[None, :] / sigma**2

    grad[0,:] = 2*np.sum( (f2.T * (fx*f1).T).T, axis=((0), (1))) / norm
    grad[1,:] = 2*np.sum( (f2.T * (fy*f1).T).T, axis=((0), (1))) / norm

    s = np.sum((Icalc - I_exp)**2)
    return s, grad

def grad_harmonic(coord, k, d0):
    
    grad = np.zeros_like(coord) # shape (3, N_ATOMS)
    
    V_h = np.sqrt(np.sum( (coord[:,1:] - coord[:, :-1])**2, axis=0 ))

    grad[:, 0] = (1 - d0[0]/V_h[0]) * (coord[:,1] - coord[:,0])

    grad[:,1:-1] = (1 - d0[1:]/V_h[1:]) * (coord[:,2:] - coord[:,1:-1] ) -\
                   (1 - d0[:-1]/V_h[:-1]) * (coord[:,1:-1] - coord[:,:-2])

    grad[:,-1] = (1 - d0[-1]/V_h[-1]) * (coord[:,-1] - coord[:,-2])

    V_h = 0.5*k * np.sum((V_h - d0)**2)
    grad *= -k

    return V_h, grad

def grad_descent(init_coord, I_exp, N_imgs, *args):

    N_steps, stride, learn_rate, tol, n_pixels, pixel_size, sigma, fact, k_h, d0 = args

    coord = init_coord.copy()
    s_cv = []
    v_hm = []

    cv_old = 0.0

    grad_hm = np.zeros_like(init_coord)
    vh = 0.0

    for i in range(N_steps):
            
        grad_l2 = np.zeros_like(init_coord)
        grad_hm = np.zeros_like(init_coord)
        acc_cv = 0.0

        if (fact != 0):
            for j in range(N_imgs):
                coord_rot = np.matmul(I_exp[j]["Q"], coord)

                I_calc = calc_img(coord_rot, n_pixels, pixel_size, sigma)
                cv, grad_l2_rot = grad_lu2(coord_rot, I_calc, I_exp[j]["I"], n_pixels, pixel_size, sigma)

                grad_l2 += np.matmul(I_exp[j]["Q_inv"], grad_l2_rot)

                acc_cv += cv
                acc_cv *= fact
                
        if (k_h != 0):
            vh, grad_hm = grad_harmonic(coord, k_h, d0)

        coord = coord - learn_rate * (fact*grad_l2 + grad_hm)
        
        cv_diff = np.abs(acc_cv - cv_old)
        if (i%stride == 0): 
            print(f"CV at step {i}: {acc_cv:.7e}, {vh:.7e}")
            s_cv.append(acc_cv)
            v_hm.append(vh)

        if (cv_diff <= tol or np.isnan(acc_cv)): 
            print(f"Stopping simulation at step {i} with diff: {cv_diff}")
            break

        cv_old = acc_cv

    return s_cv, v_hm, coord

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

def print_ref_d(fname, d0):

    if os.path.exists(fname): os.system(f"rm {fname}")

    with open(fname, "w") as f:
        f.write(f"{d0.shape[0]}\n")
        np.savetxt(f, d0)

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

def load_dataset(n_imgs, prefix):

    dataset = []
    for i in range(n_imgs):

        q = np.loadtxt(f"{prefix}{i}.txt", skiprows=1, max_rows=4)
        Iexp = np.loadtxt(f"{prefix}{i}.txt", skiprows=5)

        rot = R.from_quat(q)
        rot_mat = rot.as_matrix()
        rot_mat_inv = rot.inv().as_matrix()

        img = {"Q": rot_mat, "Q_inv":rot_mat_inv, "I": Iexp}
        dataset.append(img)

    return dataset

def write_param_file(fname, *args, print_p=True):

    n_pixels, pixel_size, sigma, cutoff, learn_rate, l2_weight, hm_weight, tol = args

    #LOAD PARAMETERS INTO FILE
    params_file = open(fname,"w")
        
    string = (
        f"NUMBER_PIXELS {n_pixels}\n"
        f"PIXEL_SIZE {pixel_size}\n"
        f"SIGMA {sigma}\n"
        f"CUTOFF {cutoff}\n"
        f"LEARN_RATE {learn_rate}\n"
        f"L2_WEIGHT {l2_weight}\n"
        f"HM_WEIGHT {hm_weight}\n"
        f"TOLERANCE {tol}\n"
    )
    
    params_file.write(string)
    params_file.close()

    if print_p:
        print("######## SETTING PARAMETERS AS ##############")
        print(string)

def grad_descent_cpp(*args):

    nranks, coord_file, param_file, out_pfx, n_imgs, img_pfx, nsteps, stride, ntomp, d0 = args

    if (nranks > 1):
        string = (
            f"mpirun -n {nranks} "
            f"./gradcv.out grad_descent "
            f"-f {coord_file} -p {param_file} -out_pfx {out_pfx} "
            f"-n_imgs {n_imgs} -img_pfx {img_pfx} "
            f"-nsteps {nsteps} -stride {stride} "
            f"-ntomp {ntomp} "
            f"-d0 {d0}"
        )

    else:
        string = (
            f"./gradcv.out grad_descent "
            f"-f {coord_file} -p {param_file} -out_pfx {out_pfx} "
            f"-n_imgs {n_imgs} -img_pfx {img_pfx} "
            f"-nsteps {nsteps} -stride {stride} "
            f"-ntomp {ntomp} "
            f"-d0 {d0}"
        )
        
    os.system(string)

np.random.seed(0)
############################################# LOAD YOUR COORDINATES HERE ####################################################

ref_sys_mda = mda.Universe("centered_ALA_final.pdb") # The system used to create the experimental images (always needed!)
ref_sys_mda = center_system(ref_sys_mda) # Center your coordinates if they are not centered

# *********************************************** From random noise ********************************************************
sigma = 1
select_filter = "not name *H*" # Change if needed (defines which atoms are selected from the pdb)
ref_sys = ref_sys_mda.select_atoms(select_filter).positions.T # This returns a numpy array of shape (3, # atoms)
sim_sys_mda = mda.Universe("centered_ALA_initial.pdb")
sim_sys = sim_sys_mda.select_atoms(select_filter).positions.T
#sim_sys = ref_sys + np.random.randn(*ref_sys.shape) * sigma # sigma = standard deviation of the noise

#sim_sys_mda = ref_sys_mda.copy()
#sim_sys_mda.select_atoms("not name *H*").positions = sim_sys.T
align.alignto(sim_sys_mda, ref_sys_mda, select="not name H*")
sim_sys_mda.select_atoms("not name *H*").write("initial_ala.pdb")


# *********************************************** From a pdb file **********************************************************
# select_filter = "not name *H*" # Change if needed (defines which atoms are selected from the pdb)

# sim_sys_mda = mda.Universe("ref_filename.pdb")
# align.alignto(sim_sys_mda, ref_sys_mda, select=select_filter) # To align initial model to reference
# ref_sys = ref_sys_mda.select_atoms(select_filter).positions.T # This returns a numpy array of shape (3, # atoms)
# sim_sys = sim_sys_mda.select_atoms(select_filter).positions.T # This returns a numpy array of shape (3, # atoms)

##############################################################################################################################


# Print coordinates for c++
COORD_FILE = "ala_initial_coords.txt" # Change if needed
print_coords(COORD_FILE, sim_sys)

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
IMG_PFX = "ala_img_"
N_IMGS = 3

# Create/Load images
# dataset = create_images(ref_sys, quat, N_PIXELS, PIXEL_SIZE, SIGMA, SNR, IMG_PFX, True) # images created = # quats
dataset = load_dataset(N_IMGS, IMG_PFX)

# Prepare gradient Descent

# Reference distances for harmonic potential
ref_d = np.sqrt(np.sum( (ref_sys[:,1:] - ref_sys[:, :-1])**2, axis=0 ))
        
print_ref_d("ref_ala.txt", ref_d)

CUTOFF = 10 #Cutoff for neighboring pixels
N_STEPS = 300000; STRIDE = 10000; LEARN_RATE = 0.001; TOL = 1e-8 # Parameters for gradient descent
L2_WEIGHT = 1.0; HM_WEIGHT = 0.00 # Weights for the forces

# Create parameters file
PARAM_FNAME = "parameters.txt"
write_param_file(PARAM_FNAME, N_PIXELS, PIXEL_SIZE, SIGMA, CUTOFF, LEARN_RATE, L2_WEIGHT, HM_WEIGHT, TOL)

# Create extra parameters for the descent
MPI_RANKS = 3
OUT_PFX = "ala_final_" # final coordinates saved as OUT_PFX + coords.txt
NTOMP = 2

# Run descent
#grad_descent_cpp(MPI_RANKS, COORD_FILE, PARAM_FNAME, OUT_PFX, N_IMGS, IMG_PFX, N_STEPS, STRIDE, NTOMP, "ref_ala.txt")

print("End of gradient descent")
print(f"The final coordinates have been uploaded to {OUT_PFX}coords.txt")
# Load final coordinates
final_sys = np.loadtxt(f"{OUT_PFX}coords.txt", skiprows=1)
colvar = np.loadtxt(f"{OUT_PFX}colvar.txt", skiprows=1)


# TODO Load value of the CV and the Harmonic Potential too (need to set up printing in c++)

#######################################################
# Uncomment if doing grad_descent with python
# gd_args = (N_STEPS, STRIDE, LEARN_RATE, TOL)
# l2_args = (N_PIXELS, PIXEL_SIZE, SIGMA, L2_WEIGHT)
# hm_args = (HM_WEIGHT, ref_d)
# L2, VH, final_sys = grad_descent(sim_sys, dataset, N_IMGS, *gd_args, *l2_args, *hm_args) 
#######################################################

# Here are a few ways you can visualize the resulting data, I need to keep working on this
sim_sys_mda.select_atoms("not name *H*").positions = final_sys.T
align.alignto(sim_sys_mda, ref_sys_mda, select="not name H*")
final_sys = sim_sys_mda.select_atoms("not name *H*").positions.T

sim_sys_mda.select_atoms("not name *H*").write("final_ala.pdb")
ref_sys_mda.select_atoms("not name *H*").write("ref_ala_noH.pdb")

final_imgs = create_images(final_sys, quat, N_PIXELS, PIXEL_SIZE, SIGMA, SNR, print_imgs=False)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(ref_sys[0], ref_sys[1], ref_sys[2], c="blue", label="ref", marker="*", s=75)
# ax.scatter(final_sys[0], final_sys[1], final_sys[2], c="red", label="final", s=50)
# # ax.scatter(sim_sys[0], sim_sys[1], sim_sys[2], color="blue", label="init", s=50)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()

# Print initial and final images
fig1, ax1 = plt.subplots(2, 3, figsize=(16,9))
ax1[0,0].imshow(dataset[0]["I"])
ax1[0,1].imshow(dataset[1]["I"])
ax1[0,2].imshow(dataset[2]["I"])

ax1[1,0].imshow(final_imgs[0]["I"])
ax1[1,1].imshow(final_imgs[1]["I"])
ax1[1,2].imshow(final_imgs[2]["I"])


fig2, ax2 = plt.subplots()
twin1 = ax2.twinx()

ax2.plot(colvar[:,0], colvar[:,1], c="red", label="L2")
twin1.plot(colvar[:,0], colvar[:,2], c="blue", label="V_Harm")
ax2.legend()
twin1.legend()

plt.show()