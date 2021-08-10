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
// typedef float myfloat_t;
// typedef myfloat_t mycomplex_t[2];
// typedef std::vector <myfloat_t> myvector_t;
// typedef std::vector <myvector_t> mymatrix_t;

// #define myfftw_malloc fftwf_malloc
// #define myfftw_free fftwf_free
// #define myfftw_destroy_plan fftwf_destroy_plan
// #define myfftw_execute_dft_r2c fftwf_execute_dft_r2c
// #define myfftw_execute_dft_c2r fftwf_execute_dft_c2r
// #define myfftw_plan_dft_r2c_2d fftwf_plan_dft_r2c_2d
// #define myfftw_plan_dft_c2r_2d fftwf_plan_dft_c2r_2d
// #define myfftw_plan fftwf_plan
// #define myfftw_cleanup fftwf_cleanup
// #define myfftw_import_wisdom_from_filename fftwf_import_wisdom_from_filename
// #define myfftw_export_wisdom_to_filename fftwf_export_wisdom_to_filename

typedef double myfloat_t;
typedef myfloat_t mycomplex_t[2];
typedef std::vector <myfloat_t> myvector_t;
typedef std::vector <myvector_t> mymatrix_t;

#define myfftw_malloc fftw_malloc
#define myfftw_free fftw_free
#define myfftw_destroy_plan fftw_destroy_plan
#define myfftw_execute_dft_r2c fftw_execute_dft_r2c
#define myfftw_execute_dft_c2r fftw_execute_dft_c2r
#define myfftw_plan_dft_r2c_2d fftw_plan_dft_r2c_2d
#define myfftw_plan_dft_c2r_2d fftw_plan_dft_c2r_2d
#define myfftw_plan fftw_plan
#define myfftw_cleanup fftw_cleanup
#define myfftw_import_wisdom_from_filename fftw_import_wisdom_from_filename
#define myfftw_export_wisdom_to_filename fftw_export_wisdom_to_filename


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

  //Defines if the program shall create a database or calculate a gradient
  const char *p_type;

  //***************** Declare variables
  int number_pixels, number_pixels_fft_1d, sigma_reach, n_atoms;
  myfloat_t pixel_size, sigma_cv;
  myfloat_t b_factor, defocus, CTF_amp, phase, min_defocus, max_defocus;
  myfloat_t norm;
  
  myvector_t quat, quat_inv;

  //***************** Control for parameters
  bool yesPixSi = false;
  bool yesNumPix = false;
  bool yesBFact = false;
  bool yesAMP = false;
  bool yesSigmaCV = false;
  bool yesSigmaReach = false;
  bool yesDefocus = false;
  //***************** Default VALUES
  myfloat_t elecwavel = 0.019866;
  myfloat_t SNR = 1.0;
  myfloat_t sqrt_2pi = std::sqrt(2. * M_PI);
  


  //Name of files used
  std::string params_file;
  std::string coords_file;
  std::string image_file;
  std::string json_file;
  

  //coordinates 
  myvector_t x_coord;
  myvector_t y_coord;
  myvector_t z_coord;

  //grid
  myvector_t x;
  myvector_t y;


  //projections and gradients
  myfloat_t s_cv;

  myfloat_t* grad_x;
  myfloat_t* grad_y;
  myfloat_t* grad_z;

  mymatrix_t Icalc;
  mymatrix_t Iexp;

  int fft_plans_created = 0;
  myfftw_plan fft_plan_r2c_forward, fft_plan_c2r_backward;

  void release_FFT_plans();

public:

  Grad_cv();
  //~Grad_cv();

  void init_variables(std::string, std::string, 
                      std::string, std::string,
                      const char *);
  void prepare_FFTs();

  void read_coord();
  void center_coord(myvector_t &, myvector_t &, myvector_t &);
  void quaternion_rotation(myvector_t &, myvector_t &, myvector_t &, myvector_t &);
  void quaternion_rotation(myvector_t &, myfloat_t*, myfloat_t*, myfloat_t*);

  void correlation(myvector_t &, myvector_t &, myvector_t &,
                   mymatrix_t &, myfloat_t *, myfloat_t *, myfloat_t &);

  void l2_norm(myvector_t &, myvector_t &, myvector_t &,
               mymatrix_t &, myfloat_t *, myfloat_t *, myfloat_t &);

  void calc_I();

  void calc_ctf(mycomplex_t*);
  void conv_proj_ctf();
  
  void I_with_noise(mymatrix_t &, myfloat_t);
  void gaussian_normalization();

  void grad_run();
  void gen_run(bool);

  //Utilities
  void arange(myvector_t &, myfloat_t, myfloat_t, myfloat_t);

  void where(myvector_t &, std::vector<size_t> &, myfloat_t, myfloat_t);
  void where(myvector_t &, myvector_t &, std::vector<int> &, myfloat_t);

  void print_image(mymatrix_t &, std::string);
  void read_exp_img(std::string);
  int read_parameters(std::string);
  void results_to_json(myfloat_t, myfloat_t*, myfloat_t*, myfloat_t*);
};
#endif
