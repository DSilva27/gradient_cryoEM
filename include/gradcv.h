#ifndef GRADCV 
#define GRADCV 

#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <string> 
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <filesystem>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>


// **************** DEFINITIONS ***********************
typedef float myfloat_t;
typedef myfloat_t mycomplex_t[2];
typedef std::vector <myfloat_t> myvector_t;
typedef std::vector <myvector_t> mymatrix_t;

#define myfftw_malloc fftwf_malloc
#define myfftw_free fftwf_free
#define myfftw_destroy_plan fftwf_destroy_plan
#define myfftw_execute_dft_r2c fftwf_execute_dft_r2c
#define myfftw_execute_dft_c2r fftwf_execute_dft_c2r
#define myfftw_plan_dft_r2c_2d fftwf_plan_dft_r2c_2d
#define myfftw_plan_dft_c2r_2d fftwf_plan_dft_c2r_2d
#define myfftw_plan fftwf_plan
#define myfftw_cleanup fftwf_cleanup

#define myError(error, ...)                                                    \
  {                                                                            \
    printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");                                      \
    printf("Error - ");                                                        \
    printf((error), ##__VA_ARGS__);                                            \
    printf(" (%s: %d)\n", __FILE__, __LINE__);                                 \
    printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");                                      \
    exit(1);                                                                   \
  }


class Grad_cv {
    
 private:

  //***************** Declare variables
  int number_pixels, number_pixels_fft_1d, sigma_reach;
  myfloat_t pixel_size, sigma_cv;
  myfloat_t b_factor, min_defocus, max_defocus, CTF_amp, phase;
  myfloat_t defocus;

  myvector_t quaternions;

  //***************** Control for parameters
  bool yesPixSi = false;
  bool yesNumPix = false;
  bool yesBFact = false;
  bool yesDefocus = false;
  bool yesAMP = false;
  bool yesSigmaCV = false;
  bool yesSigmaReach = false;
  //***************** Default VALUES
  myfloat_t elecwavel = 0.019866;
  myfloat_t SNR = 1.0;


  //Name of files used
  std::string params_file;
  std::string coords_file;
  std::string image_file;
  

  //coordinates 
  myvector_t x_coord;
  myvector_t y_coord;
  myvector_t z_coord;

  //grid
  myvector_t x;
  myvector_t y;


  //projections and gradients
  myfloat_t s_cv;

  myvector_t grad_x;
  myvector_t grad_y;
  myvector_t grad_z;

  mymatrix_t Icalc;
  mymatrix_t Iexp;

  int fft_plans_created = 0;
  myfftw_plan fft_plan_r2c_forward, fft_plan_c2r_backward;

  void release_FFT_plans();

public:

  Grad_cv(std::string, std::string, std::string);
  //~Grad_cv();

  void init_variables();
  void prepare_FFTs();

  void read_coord();
  void center_coord(myvector_t &, myvector_t &, myvector_t &);
  void quaternion_rotation(myvector_t &);

  void I_calculated();
  void calc_ctf(mycomplex_t*);
  void conv_proj_ctf();
  
  void I_with_noise(mymatrix_t &, mymatrix_t &, myfloat_t);
  myfloat_t calc_I_variance();
  myfloat_t collective_variable();
  void gradient(myvector_t &, myvector_t &, myvector_t &, const char *);
  void run();

  //Utilities
  void arange(myvector_t &, myfloat_t, myfloat_t, myfloat_t);

  void where(myvector_t &, std::vector<size_t> &, myfloat_t, myfloat_t);
  void where(myvector_t &, myvector_t &, std::vector<int> &, myfloat_t);

  void matmul(mymatrix_t &, mymatrix_t &, mymatrix_t &);
  void transpose(mymatrix_t &, mymatrix_t &);
  void print_image(mymatrix_t &, std::string);
  int read_parameters(const char *);
  void load_quaternions(myvector_t &);
  void results_to_json(myfloat_t, myvector_t &, myvector_t &, myvector_t &);
};
#endif