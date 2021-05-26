# Gradient Cryo-EM
# January, 2021
## Contributors
Arley Flórez López, David Silva Sánchez and Pilar Cossio

## References
* [1] [Cossio, P and Hummer, G. J Struct Biol. 2013 Dec;184(3):427-37. doi: 10.1016/j.jsb.2013.10.006.](http://www.ncbi.nlm.nih.gov/pubmed/24161733)
* [2] [K. Shoemake. Uniform random rotations. In D. Kirk, editor, Graphics Gems III, pages 124-132. Academic, New York, 1992](https://www.sciencedirect.com/book/9780124096738/graphics-gems-iii-ibm-version)

## Description

The Gradient Cryo-EM code calculates the correlation of the projection of an structural model with a 2d cryo-em raw image (experimental image). The projection is done by rotating the system, modeling the C-alpha atoms as gaussians and integrating in the z-direction. We then apply CTF effects to the image by Fourier convolution, we call that the calculated image. Then we calculate the cross-correlation between the calculated and experimental image and the respective gradient. Most of these methods are based on the ones already implemented in BioEM [1].

With this branch you can create your own database. The only ctf parameter that is varied is the defocus. The reference system is rotated using random quaternions as described in [2].

### C++ code

```
#clone the repository
git clone ...
git checkout db_gen
#build the c++ code (FFTW needs to be installed)
make
#install the python dependencies
#if you have conda
./setup_env.sh
#if not, then install using pip (replace the x with your python version)
pythonx -m pip install matplotlib numpy MDAnalysis
```

Dependencies and software requirements:

* FFTW: a serial but fully thread-safe fftw3 installation or equivalent (tested with fftw 3.3)
     -> point environment variable $FFTW_ROOT to a FFTW3 installation or use ccmake to specify

* conda (optional): a package and virtual environment manager for python. https://docs.conda.io/en/latest/miniconda.html

## Running the program

* You should edit python/prep_parameters.py to set your own parameters
* Place your pdb's in data/input/
* Make sure you're using the venv em2d_env created by running setup_env.sh (or that you installed the necessary libraries)

```
#creates data/input/parameters.txt and data/input/quaternions.txt
python python/prep_parameters.py

#Run the main code using the python wrapper 
python python/cal_gradient.py --ref_pdb [ref_pdb_without_path] --system_pdb [system_pdb_without_path]
```
