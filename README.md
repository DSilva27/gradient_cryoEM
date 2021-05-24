# Gradient Cryo-EM
# January, 2021
## Contributors
Arley Flórez López, David Silva Sánchez and Pilar Cossio

## Description

The Gradient Cryo-EM code calculates the correlation of the projection of an structural model with a 2d cryo-em raw image (experimental image). The projection is done by rotating the system, modeling the C-alpha atoms as gaussians and integrating in the z-direction. We then apply CTF effects to the image by Fourier convolution, we call that the calculated image. Then we calculate the cross-correlation between the calculated and experimental image and the respective gradient. 

### C++ code

```
#clone the repository
git clone ...
cd BioEM
#build the c++ code
make
#install the python dependencies
#if you have conda
./setup_env.sh
#if not the install using pip (replace the x with your python version)
pythonx -m pip install matplotlib numpy MDAnalysis
```

Dependencies and software requirements:

* FFTW: a serial but fully thread-safe fftw3 installation or equivalent (tested with fftw 3.3)
     -> point environment variable $FFTW_ROOT to a FFTW3 installation or use ccmake to specify

* conda (optional): a package and virtual environment manager for python. https://docs.conda.io/en/latest/miniconda.html

## References
* ...
## Description
...
