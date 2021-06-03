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
```
#### Possible errors
If you get the error `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.` just run `conda init <shell_name>` and then `conda activate em2d_end`.

### Main steps for generating images from reference (only once)

* Place the reference pdb (the one you're going to use to create the images) in data/input/
```
#Activate the previous conda env
conda activate em2d_env

# go into the develop branch
git checkout develop

#build the c++ code (FFTW needs to be installed)
make

#Create input parameters for images (e.g., number of pixels, pixel size etc).  You should edit python/prep_parameters.py to set your own parameters
python python/prep_input_files.py

#Generate images in parallel (reference pdb MUST be in the data/input, otherwise it wont work)
python python/gen_images.py -n_proc [processors to be used] --n_img [images to be generated] --ref_pdb [ref_pdb_without_path] 
```

### Main steps for calculating the gradient 

* Place the structural pdb (from MD) in data/input/

```
#Activate the previous conda env (if not done so)
conda activate em2d_env

# go into the develop branch
git checkout develop

#build the c++ code 
make

#Create input parameters for images (e.g., number of pixels, pixel size etc)
python python/prep_parameters.py

# Run main code to calculate the gradient 
python python/cal_gradient.py N_PROC --n_img N_IMG --ref_pdb REF_PDB --system_pdb SYSTEM_PDB
```
