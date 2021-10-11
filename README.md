# Gradient Cryo-EM

## January, 2021

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

## Dependencies and software requirements

* FFTW: a serial but fully thread-safe fftw3 installation or equivalent (tested with fftw 3.3)
     -> point environment variable $FFTW_ROOT to a FFTW3 installation or use ccmake to specify (not needed for now)

* conda (optional): a package and virtual environment manager for python. <https://docs.conda.io/en/latest/miniconda.html>
* MDAnalysis: a Python library for the analysis of computer simulations of many-body systems at the molecular scale.
* At least g++ 9.3.0 or a compiler that let's you use C++17 (change Makefile if not using g++, Cmake has not been implemented yet)
* MPI/OpenMP, usually installed by default (check that you the compiler mpi++)

### Main steps for cloning and installation (only once)

```bash
#clone the repository
git clone ...
#install the python dependencies
#if you have conda
./setup_env.sh
#if not, then install using pip (replace the x with your python version, e.g, python3)
pythonx -m pip install matplotlib numpy MDAnalysis

# go into the develop branch
git checkout develop

#build the c++ code (FFTW does not needs to be installed)
make
```

#### Possible errors

If you get the error `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.` just run `conda init <shell_name>` and then `conda activate em2d_end`.

### Before running the code

* Copy the compiled code in your working directory (where you will run stuff)
* Copy the wrapper you need in your working directory (more on this later)
* Copy the PDB/TXT file that contains the model's coordinates in your working directory

### Generate Cryo-EM Images

* Copy wrappers/gen_imgs.py in your working directory
* Follow the instructions inside gen_imgs.py to generate your images

### Run a gradient descent simulation

* Copy wrappers/grad_desc.py in your working directory
* Follow the instructions inside grad_desc.py in your working directory

### Examples

There are three examples available in /example. There you can find:

* Coordinate files (PDB or TXT)
* Wrappers already configured for running the code (REMEMBER TO COPY THE COMPILED C++ CODE!)
