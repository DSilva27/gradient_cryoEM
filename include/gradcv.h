#ifndef GRADCV 
#define GRADCV 

#define _USE_MATH_DEFINES

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <typeinfo>
#include <thread>
#include <random>

typedef double myfloat_t;
typedef myfloat_t mycomplex_t[2];
typedef std::vector<myfloat_t> myvector_t;
typedef std::vector<myvector_t> mymatrix_t;

typedef struct image{

    myfloat_t defocus = 0;
    myvector_t q = myvector_t(4, 0.0);
    myvector_t q_inv = myvector_t(4, 0);
    myvector_t I;

    std::string fname;
} myimage_t;

typedef struct param_device{

  std::string mode;
  int n_pixels, n_neigh, n_imgs, n_atoms;
  myfloat_t pixel_size, sigma, cutoff, norm;
  myfloat_t learn_rate, l2_weight, hm_weight;
  
  myvector_t grid;    

  void calc_neigh(){
    n_neigh = (int) std::ceil(sigma * cutoff / pixel_size);
  }

  void calc_norm(){
    norm = 1. / (2*M_PI * sigma*sigma * n_atoms);
  }

  void gen_grid(){
    grid.resize(n_pixels);
    myfloat_t grid_min = -pixel_size * (n_pixels - 1)*0.5;
    for (int i=0; i<n_pixels; i++) grid[i] = grid_min + pixel_size*i;
  }

} myparam_t;


typedef std::vector<myimage_t> mydataset_t;

#pragma omp declare reduction(vec_float_plus : std::vector<myfloat_t> : \
                              std::transform(omp_out.begin(), omp_out.end(), \
                              omp_in.begin(), omp_out.begin(), std::plus<myfloat_t>())) \
                              initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#define myError(error, ...)                                                    \
  {                                                                            \
    printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");                                      \
    printf("Error - ");                                                        \
    printf((error), ##__VA_ARGS__);                                            \
    printf(" (%s: %d)\n", __FILE__, __LINE__);                                 \
    printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");                                      \
    exit(1);                                                                   \
  }

void parse_args(int, char**, int, int, std::string &, std::string &, 
                std::string &, std::string &, std::string &, int &, 
                int &, int &, int &);
void read_coord(std::string, myvector_t &, int);
void read_parameters(std::string, myparam_t *, int);
void load_dataset(std::string, int, int, mydataset_t &, int, int);
void read_exp_img(std::string, myimage_t *);
void print_image(myimage_t *, int);
void print_image(std::string, myvector_t &, int);
void print_coords(std::string, myvector_t &, int);
void where(myvector_t &, myvector_t &, std::vector<int> &, myfloat_t);                       

void quaternion_rotation(myvector_t &, myvector_t &, myvector_t &);
void quaternion_rotation(myvector_t &, myvector_t &);
void gaussian_normalization(myvector_t &, param_device *, myfloat_t);

void calc_img(myvector_t &, myvector_t &, myparam_t *);
void calc_img_omp(myvector_t &, myvector_t &, myparam_t *, int);

void L2_grad(myvector_t &, myvector_t &, myvector_t &, myvector_t &, myfloat_t &, myparam_t *);
void L2_grad_omp(myvector_t &, myvector_t &, myvector_t &, myvector_t &, myfloat_t &, myparam_t *, int);
void harm_pot(myvector_t &, myfloat_t, myfloat_t, myfloat_t &, myvector_t &, int);

void run_emgrad(std::string, std::string, int, int, int, int);
void run_gen(std::string, std::string, int, int, int, int);
void run_grad_descent(std::string, std::string, std::string, std::string, int, int, int, int, int, int);

// Tests
void run_num_test(std::string, std::string, int, int, int, int);
void run_num_test_omp(std::string, std::string, int, int, int, int);

void run_time_test(std::string, std::string, int, int, int, int);
void run_time_test_omp(std::string, std::string, int, int, int, int);

#endif