# Gradient Cryo-EM
# January, 2021
## Contributors
David Silva Sánchez, Arley Flórez López, and Pilar Cossio

## References
* [1] [Cossio, P and Hummer, G. J Struct Biol. 2013 Dec;184(3):427-37. doi: 10.1016/j.jsb.2013.10.006.](http://www.ncbi.nlm.nih.gov/pubmed/24161733)
* [2] [K. Shoemake. Uniform random rotations. In D. Kirk, editor, Graphics Gems III, pages 124-132. Academic, New York, 1992](https://www.sciencedirect.com/book/9780124096738/graphics-gems-iii-ibm-version)

## Description

The objective of this code is to calculate the gradient of the correlation of a projection of a structural model (e.g. from MD) with a set of 2D cryo-EM raw images (experimental images). We assume that the optimal projection direction of each image to a reference model are known.

The main steps for a single image are:
1) Rotation alignment: align model to reference and extract rotation matrix (python)
2) Rotation for projection: rotate align model using optimal quaterions for the projection direction (c++)
3) Projection: calculate an ideal image from rotated model (c++)
4) Cross-correlation and gradient: extract these by comparing the ideal image to the experimental image (see notes) (c++)

Most of these methods are based on the ones already implemented in BioEM [1]. For multiple images, the code is parallelized using the multiprocessing python tool. 

## Dependencies and software requirements:

* FFTW: a serial but fully thread-safe fftw3 installation or equivalent (tested with fftw 3.3)
     -> point environment variable $FFTW_ROOT to a FFTW3 installation or use ccmake to specify

* conda (optional): a package and virtual environment manager for python. https://docs.conda.io/en/latest/miniconda.html
* MDAnalysis: a Python library for the analysis of computer simulations of many-body systems at the molecular scale.
* At least g++ 9.3.0 or a compiler that let's you use C++17 (change Makefile if not using g++, Cmake has not been implemented yet)

### Main steps for cloning and installation (only once)

```
#clone the repository
git clone ...
#install the python dependencies
#if you have conda
./setup_env.sh
#if not, then install using pip (replace the x with your python version)
pythonx -m pip install matplotlib numpy MDAnalysis

# go into the develop branch
git checkout develop

#build the c++ code (FFTW needs to be installed)
make
```
#### Possible errors
If you get the error `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.` just run `conda init <shell_name>` and then `conda activate em2d_end`.

### Main steps for generating images from reference (only once)

* Place the reference pdb (the one you're going to use to create the images) in data/input/
* The images are generated using all the atoms with Gaussians with the same width centered at their atomic positions.

```
#Activate the previous conda env
conda activate em2d_env

#Create input parameters for images (e.g., number of pixels, pixel size etc).  You should edit python/prep_parameters.py to set your own parameters
python emgrad.py set_params -np NUMBER_PIXELS -ps PIXEL_SIZE -s SIGMA -nc NEIGHBOR_CUTOFF 

#The parameters file is saved in data/input/parameters.txt, if you want new parameters you can change it directly or re-run the command

#Generate images in parallel (reference pdb MUST be in the data/input, otherwise it wont work)
# Example REF_PDB=apo.pdb that is in data/input (no need to include directory address)
python gen_db --n_proc N_PROC --n_img N_IMG --ref_pdb REF_PDB
```

### Main steps for calculating the gradient (same develop branch)

* Place the structural pdb (from MD) in data/input/ (e.g. system.pdb)
* The default here is to calculate the gradient just with respect to the C-alpha atoms (but this can be changed)

```
#Activate the previous conda env (if not activated)
conda activate em2d_env

# Run main code to calculate the gradient 
# REF_PDB and SYSTEM_PDB should be in data/input
python emgrad.py calc_grad --n_proc N_PROC --n_img N_IMG --ref_pdb REF_PDB --system_pdb SYSTEM_PDB


```

### Main steps for doing a gradient descent run (same develop branch)

* Place the structural pdb (from MD) in data/input/ (e.g. system.pdb)
* The default here is to calculate the gradient just with respect to the C-alpha atoms (but this can be changed)

```
#Activate the previous conda env (if not activated)
conda activate em2d_env

# Run main code to calculate the gradient 
# REF_PDB and SYSTEM_PDB should be in data/input
python emgrad.py grad_desc --n_proc N_PROC --n_img N_IMG --ref_pdb REF_PDB --system_pdb SYSTEM_PDB

```

#### Additional flags
* n_steps: maximum number of steps for the descent (default=100)
* learn_rate: learning rate for the descent (default=100)
* stride: stride for writing into trajectory file (default=1)
* tol: stop the descent if the change in the cv is less than this value



### Output Files
When generating images: 
* data/input/parameters.txt: Parameters used to generate the synthetic images.
* data/images/Ical_??.txt : Synthetic images: first item: defocus; second - fifth items: quaternions for projection; following items: image.

When calculating the gradient:
* data/input/coord.txt : aligned coordinates of system with reference structure.
* data/output/grad_??.json: jason files with gradients
* data/output/grad_all.json: file that has the accumulation of all the gradient (if multiple images are used)

When doing gradient descent (TODO: change folder name): 
* data/gd_images/traj.xyz : trajectory of the descent 
* data/gd_image/apo.xyz : reference structure (TODO: change name)
