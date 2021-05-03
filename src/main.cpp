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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef std::vector <double> vec;
typedef std::vector <vec> mat;

void read_data(vec &, vec &, vec &);
void center_coord(vec &, vec &, vec &);
void quaternion_rotation(vec &, vec &, vec &, vec &, vec &, vec &, vec &);
void I_calculated(vec &, vec &, double, int, int, mat &, vec &, vec &);
void I_with_noise(mat &, mat &, double);
double collective_variable(mat &, mat &);
void gradient(mat &, mat &, vec &, vec &, vec &, double);

//Utilities
void linspace(vec &, double, double, int);
void where(vec &, std::vector<size_t> &, double, double);
void matmul(mat &, mat &, mat &);
void transpose(mat &, mat &);
void print_image(mat &, std::string);
void load_parameters(int &, double &, int &);
void load_quaternions(vec &);
void results_to_json(double, vec &, vec &, vec &);

int main(){

  //Define all the variables that are going to be used  
  vec x_coord; vec y_coord; vec z_coord;

  int n;
  double sigma;
  int res;
  
  //Load data in data/input/parameters.txt
  load_parameters(n, sigma, res);

  //Read coordinates of the atoms
  read_data(x_coord, y_coord, z_coord);
  int N = x_coord.size(); //number of atoms

  /*
  //Move the center of mass to the origin
  center_coord(x_coord, y_coord, z_coord);

  std::vector <double> q(4, 0);
  //Fill vector q with the parameters of the quaternion rotation matrix
  load_quaternions(q);
  //create vector for the rotated coordinates
  vec x_rot(N); vec y_rot(N); vec z_rot(N);

  quaternion_rotation(q, x_coord, y_coord, z_coord, x_rot, y_rot, z_rot);
  */

  mat Ixy; vec x; vec y;

  //The coordinates are given already centered and rotated
  I_calculated(x_coord, y_coord, sigma, n, res, Ixy, x, y);

  std::vector <std::vector <double>> Iexp(Ixy.size(), std::vector<double> (Ixy.size(), 0));

  I_with_noise(Ixy, Iexp, 8);
  print_image(Ixy, "data/output/Iclean.txt");
  print_image(Iexp, "data/output/Inoise.txt");

  double S = collective_variable(Ixy, Iexp);

  std::vector <double> Sgrad_x(N, 0);
  std::vector <double> Sgrad_y(N, 0);
  std::vector <double> Sgrad_z(N, 0);

  gradient(Ixy, Iexp, x, x_coord, Sgrad_x, sigma);
  gradient(Ixy, Iexp, y, y_coord, Sgrad_y, sigma);

  results_to_json(S, Sgrad_x, Sgrad_y, Sgrad_z);
  
  return 0;
}

void read_data(vec &x_a,  vec &y_a,  vec &z_a){

  /**
   * @brief reads data from a pdb and stores the coordinates in vectors
   * 
   * @param y_a stores the y coordinates of the atoms
   * @param z_a stores the z coordinates of the atoms
   * @param x_a stores the x coordinates of the atoms
   * 
   * @return void
   */

  // ! I'm not going to go into much detail here, because this will be replaced soon.
  // ! It's just for testing

  std::ifstream xfile, yfile, zfile;
  xfile.open("data/input/xcoord.txt");
  yfile.open("data/input/ycoord.txt");
  zfile.open("data/input/zcoord.txt");
  
  std::string data;
  
  while(std::getline(xfile, data)){

    if(data.size() > 0){
      x_a.push_back(std::stod(data));
    }
  }

  while(std::getline(yfile, data)){

    if(data.size() > 0){
      y_a.push_back(std::stod(data));
    }
  }

  while(std::getline(zfile, data)){

    if(data.size() > 0){
      z_a.push_back(std::stod(data));
    }
  }

  xfile.close();
  yfile.close();
  zfile.close();
}

void center_coord(vec &x_a,  vec &y_a,  vec &z_a){

  /**
   * @brief Centers the coordinates of the biomolecule around its center of mass 
   * 
   * @param x_a stores the values of x
   * @param y_a stores the values of y
   * @param z_a stores the values of z
   * 
   * @return void
   */

  double x_mean, y_mean, z_mean;
  int n = x_a.size();

  for (unsigned int i=0; i<n; i++){

    x_mean += x_a[i];
    y_mean += y_a[i];
    z_mean += z_a[i];
  }

  x_mean /= n;
  y_mean /= n;
  z_mean /= n;

  for (unsigned int i=0; i<n; i++){

    x_a[i] -= x_mean;
    y_a[i] -= y_mean;
    z_a[i] -= z_mean;
  }
}

void quaternion_rotation(vec &q, vec &x_a,  vec &y_a,  vec &z_a,  vec &x_r, vec &y_r, vec &z_r){

/**
 * @brief Rotates a biomolecule using the quaternions rotation matrix
 *        according to (https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion)
 * 
 * @param q vector that stores the parameters for the rotation std::vector <double> (4)
 * @param x_a original coordinates x
 * @param y_a original coordinates y
 * @param z_a original coordinates z
 * @param x_r stores the rotated values x
 * @param y_r stores the rotated values x
 * @param z_r stores the rotated values x
 * 
 * @return void
 * 
 */

//Definition of the quaternion rotation matrix 

  double q00 = 1 - 2*std::pow(q[2],2) - 2*std::pow(q[3],2);
  double q01 = 2*q[1]*q[2] - 2*q[3]*q[0];
  double q02 = 2*q[1]*q[3] + 2*q[2]*q[0];
  vec q0{ q00, q01, q02 };
  
  double q10 = 2*q[1]*q[2] + 2*q[3]*q[0];
  double q11 = 1 - 2*std::pow(q[1],2) - 2*std::pow(q[3],2);
  double q12 = 2*q[2]*q[3] - 2*q[1]*q[0];
  vec q1{ q10, q11, q12 };

  double q20 = 2*q[1]*q[3] - 2*q[2]*q[0];
  double q21 = 2*q[2]*q[3] + 2*q[1]*q[0];
  double q22 = 1 - 2*std::pow(q[1],2) - 2*std::pow(q[2],2);
  vec q2{ q20, q21, q22};

  std::vector <vec> Q{ q0, q1, q2 };
  
  int n = x_a.size();
  
  for (unsigned int i=0; i<n; i++){

    x_r[i] = x_a[i]*Q[0][0] + y_a[i]*Q[0][1] + z_a[i]*Q[0][2];
    y_r[i] = x_a[i]*Q[1][0] + y_a[i]*Q[1][1] + z_a[i]*Q[1][2];
    z_r[i] = x_a[i]*Q[2][0] + y_a[i]*Q[2][1] + z_a[i]*Q[2][2];

  }
}

void I_calculated(vec &x_a,  vec &y_a,  double sigma, int n, int res, mat &Ixy, vec &x, vec &y){

  /**
   * @brief Calculates the image from the 3D model by representing the atoms as gaussians and using a 2d Grid
   * 
   * @param x_a original coordinates x
   * @param y_a original coordinates y
   * @param z_a original coordinates z
   * @param sigma standard deviation for the gaussians, equal for all the atoms  (double)
   * @param n number of sigmas used for cutoff (int)
   * @param res Resolution of the calculated image
   * @param Ixy Matrix used to store the calculated image
   * @param x values in x for the grid
   * @param y values in y for the grid
   * 
   * @return void
   * 
   */

  //Calculate minimum and maximum values for the linspace-like vectors x and y
  auto xmin = *std::min_element(x_a.begin(), x_a.end());
  auto xmax = *std::max_element(x_a.begin(), x_a.end());
    
  auto ymin = *std::min_element(y_a.begin(), y_a.end());
  auto ymax = *std::max_element(y_a.begin(), y_a.end());

  //Assign memory space required to fill the vectors
  x.resize(res); y.resize(res);

  //Generate them
  linspace(x, xmin-1, xmax+1, res);
  linspace(y, ymin-1, ymax+1, res);

  //Turn Ixy into a res x res matrix
  Ixy.resize(res);
  for (int i = 0; i < res; i++) { Ixy[i].resize(res); }

  //Fill Ixy with zeros
  for (int i=0; i<res; i++){ for (int j=0; j<res; j++){ Ixy[i][j] = 0; }}

  //Vectors used for masked selection of coordinates
  std::vector <size_t> x_sel;
  std::vector <size_t> y_sel;

  //Vectors to store the values of the gaussians
  std::vector <double> g_x(res, 0.0);
  std::vector <double> g_y(res, 0.0);

  for (int atom=0; atom<x_a.size(); atom++){

    //calculates the indices that satisfy |x - x_atom| <= n*sigma
    where(x, x_sel, x_a[atom], n*sigma);
    where(y, y_sel, y_a[atom], n*sigma);

    //calculate the gaussians
    for (int i=0; i<x_sel.size(); i++){

      g_x[x_sel[i]] = std::exp( -0.5 * (std::pow( (x[x_sel[i]] - x_a[atom])/sigma, 2 )) );
    }

    for (int i=0; i<y_sel.size(); i++){

      g_y[y_sel[i]] = std::exp( -0.5 * (std::pow( (y[y_sel[i]] - y_a[atom])/sigma, 2 )) );
    }

    //Calculate the image
    for (int i=0; i<Ixy.size(); i++){ 
      for (int j=0; j<Ixy[0].size(); j++){ 
        
        Ixy[i][j] += g_x[i] * g_y[j];     
      }
    }

    //Reset the vectors for the gaussians and selection
    x_sel.clear(); y_sel.clear();

    g_x.clear(); g_y.clear();
    g_x.resize(res); g_y.resize(res);
    std::fill(g_x.begin(), g_x.end(), 0);
    std::fill(g_y.begin(), g_y.end(), 0);
  }

  for (int i=0; i<Ixy.size(); i++){ 
    for (int j=0; j<Ixy[0].size(); j++){ 
        
      Ixy[i][j] *= std::sqrt(2 * M_PI) * sigma;     
    }
  }
}

void I_with_noise(mat &I_i, mat &I_f, double std=0.1){

  /**
   * @brief Blurs an image using gaussian noise
   * 
   * @param I_i reference image (matrix)
   * @param I_f reference image + noise (matrix)
   * 
   */

  // Define random generator with Gaussian distribution
  const double mean = 0.0;
  std::default_random_engine generator;
  std::normal_distribution<double> dist(mean, std);

  // Add Gaussian noise
  for (int i=0; i<I_i.size(); i++){
    for (int j=0; j< I_i[i].size(); j++){

      I_f[i][j] = dist(generator) + I_i[i][j];
    }
  }
}

double collective_variable(mat &Ical, mat &Iexp){

  /**
   * @brief Calculates the collective variable (s). Which is the cross-correlation between the calculated image
   *        and an experimental or synthetic image.
   * 
   *        s = - sum_w sum_{x, y} Icalc(x, y, phi_w) * Iw(x, y)
   * 
   *      ! phi_w is a rotation (to be developed)
   * 
   * @param Ical calculated image (matrix of doubles)
   * @param Iexp experimental or synthetic image (matrix of doubles)
   * 
   * @return s the value of the collective variable
   */

  double s=0; //to store the collective variable

  //TODO: improve the raising of this warning
  if (Ical.size() != Iexp.size()){ 

    std::cout << "Intensity matrices must have the same dimension" << std::endl;
    return 0;
  }

  int N = Ical.size();
  int M = Ical[0].size();

  for (int i=0; i<N; i++){
    for (int j=0; j<M; j++){

      s += Ical[i][j] * Iexp[i][j];
    }
  }

  return -s;
}

void gradient(mat &Ical, mat &Iexp, vec &r, vec &r_a, vec &sgrad, double sigma){

  /**
   * @brief Calculates the gradient of the colective variable (s) for image w along r_a 
   *        An example respect to coordinate x:
   *        
   *        ds_w/dx_i = - sum_{x, y} (x - x_i)/sigma^2 * Icalc(x, y, phi_w)Iw(x, y)
   * 
   *        ! phi_w is a rotation (to be developed)
   * 
   * @param Icalc calculated image (matrix of doubles)
   * @param Iexp experimental or synthetic image (matrix of doubles)
   * @param r coordinates for the grid (e.g x or y)
   * @param r_a coordinates for the atoms (e.g x_a or y_a)
   * @param sigma standard deviation of the gaussians used for the atoms
   * 
   * @return void
   * 
   */

  int N = Ical.size();
  int M = Ical[0].size();

  //Creates matrices of dimension NxM filled with zeros
  std::vector <std::vector <double>> Sxy(N, std::vector<double> (M, 0));
  std::vector <std::vector <double>> Iexp_T(N, std::vector<double> (M, 0));

  transpose(Iexp, Iexp_T);
  matmul(Ical, Iexp_T, Sxy);

  for (int i=0; i<sgrad.size(); i++){
    for (int j=0; j<r.size(); j++){

      sgrad[i] += (r_a[i] - r[j]) * Sxy[j][j];
    }
    sgrad[i] /= std::pow(sigma, 2);
  }
}

// Utilities
void where(vec &inp_vec, std::vector<size_t> &out_vec, double x_res, double limit){  

    /**
     * @brief Finds the indexes of the elements of a vector (x) that satisfy |x - x_res| <= limit
     * 
     * @param inp_vec vector of the elements to be evalutated
     * @param out_vec vector to store the indices
     * @param x_res value used to evaluate the condition (| x - x_res|)
     * @param limit value used to compare the condition ( <= limit)
     * 
     * @return void
     */

    //std::vector::iterator points to the position that satisfies the condition
    auto it = std::find_if( std::begin(inp_vec), std::end(inp_vec ), 
              [&](double i){ return std::abs(i - x_res) <= limit; });
    
    //while the pointer doesn't point to the end of the vector
    while (it != std::end(inp_vec)) {

        //save the value of the last index found
        out_vec.emplace_back(std::distance(std::begin(inp_vec), it));

        //calculate the next item that satisfies the condition
        it = std::find_if(std::next(it), std::end(inp_vec), 
             [&](double i){return std::abs(i - x_res) <= limit;});
    }
}

void linspace(vec &out_vec, double xo, double xf, int n){

    /**
     * @brief Creates a linearly spaced vector from xo to xf with dimension n
     * 
     * @param out_vec vector where the values will be stored std::vector <double>
     * @param xo initial value (double)
     * @param xf final value (double)
     * 
     * @return void
     */

    //spacing for the values
    double a_x = (xf - xo)/(n - 1.0);

    //fills a vector with values separated by a_x starting from xo [xo, xo + a, xo + 2a, ...]
    std::generate(out_vec.begin(), out_vec.end(), [n=0, &xo, &a_x]() mutable { return n++ * a_x + xo; });   
}

void transpose(mat &A, mat &A_T){

  /**
   * @brief Computes the transpose of a matrix
   * 
   * @param A: original matrix (std::vector <std::vector <double>>)
   * @param A_T: matrix where the transpose of A will be stored (std::vector <std::vector <double>>)
   * 
   * @return void
   */

  for (int i=0; i<A.size(); i++){
    for (int j=0; j<A[0].size(); j++){

      A_T[i][j] = A[j][i]; //turns Aij to Aji
    }
  }
}

void matmul(mat &A, mat &B, mat &C){

  /**
   * @brief Performs matrix multiplication of 2D matrices
   * 
   * @param A matrix to be multiplied by the left (std::vector <std::vector <double>>)
   * @param B matrix to be multiplied by the right (std::vector <std::vector <double>>)
   * @param C matrix used to store the multiplication (std::vector <std::vector <double>>)
   * 
   * @return void
   */

  for (int i=0; i<C.size(); i++){
    for (int j=0; j<C[0].size(); j++){

      for (int k=0; k<A.size(); k++){

          C[i][j] += A[i][k] * B[k][j]; //Typical matrix multiplication cij = sum_k aik bkj
      }
    }
  }
}

void print_image(mat &Im, std::string fname){

  std::ofstream matrix_file;
  matrix_file.open (fname);
  int N = Im.size();
  int M = Im[0].size();

  std::cout.precision(3);

  for (int i=0; i<N; i++){
    for (int j=0; j<M; j++){

      matrix_file << std::scientific << std::showpos << Im[i][j] << " " << " \n"[j==M-1];
    }
  }

  matrix_file.close();
}

void load_parameters(int &n, double &sigma, int &res) {

  char letter, eq;
  double value;

  std::ifstream file;
  file.open("data/input/parameters.txt");

  if (file.fail()){
    std::cout << "Warning: file parameters.txt doesn't exist" << std::endl;

    return;
  }

  while (file >> letter >> eq >> value) {

    switch (letter) {

    case 'N':
      n = (int)value;
      break;

    case 'S':
      sigma = value;
      break;

    case 'R':
      res = (int)value;
      break;

    default:

      std::cout << "InputError: undefined parameter in parameters.txt" << std::endl;
      break;
    }
  }

  file.close();
}

void load_quaternions(vec &q){
  
  char letter, eq;
  double value;

  std::ifstream file;
  file.open("data/input/quaternions.txt");

  if (file.fail()){
    std::cout << "Warning: file quaternions.txt doesn't exist" << std::endl;

    return;
  }

  while (file >> letter >> eq >> value) {

    switch (letter) {

    case 'A':
      q[0] = value;
      break;

    case 'B':
      q[1] = value;
      break;

    case 'C':
      q[2] = value;
      break;
      
    case 'D':
      q[3] = value;
      break;

    default:

      std::cout << "InputError: undefined parameter in quaternions.txt" << std::endl;
      break;
    }
  }

  file.close();
}

void results_to_json(double s, vec &sgrad_x, vec &sgrad_y, vec &sgrad_z) {

  std::ofstream gradfile;

  gradfile.open("data/output/grad.json");

  //begin json file
  gradfile << "[" << std::endl;
  gradfile << std::setw(4) << "{" << std::endl;

  //save colvar
  gradfile << std::setw(10) << "\"s\": "
           << s << "," << std::endl;

  //begin sgrad_x
  gradfile << std::setw(17) << "\"sgrad_x\":"
           << "[" << std::endl;

  for (int j=0; j<sgrad_x.size(); j++) {

    if (j == sgrad_x.size() - 1) {

      gradfile << std::setw(17) << std::left << " " << sgrad_x[j]
               << std::endl;
    }

    else {
      gradfile << std::setw(17) << std::left << " " << sgrad_x[j] << ","
               << std::endl;
    }
  }

  gradfile << std::setw(6) << " "
           << "]," << std::endl;
  //end sgrad_x

  //begin sgrad_y
  gradfile << std::setw(17) << "\"sgrad_y\":"
           << "[" << std::endl;

  for (int j=0; j<sgrad_y.size(); j++) {

    if (j == sgrad_y.size() - 1) {

      gradfile << std::setw(17) << std::left << " " << sgrad_y[j]
               << std::endl;
    }

    else {
      gradfile << std::setw(17) << std::left << " " << sgrad_y[j] << ","
               << std::endl;
    }
  }

  gradfile << std::setw(6) << " "
           << "]," << std::endl;

  //end sgrad_y

  //begin sgrad_z
  gradfile << std::setw(17) << "\"sgrad_z\":"
           << "[" << std::endl;

  for (int j=0; j<sgrad_z.size(); j++) {

    if (j == sgrad_z.size() - 1) {

      gradfile << std::setw(17) << std::left << " " << sgrad_z[j]
               << std::endl;
    }

    else {
      gradfile << std::setw(17) << std::left << " " << sgrad_z[j] << ","
               << std::endl;
    }
  }

  gradfile << std::setw(6) << " "
           << "]" << std::endl;
  //end sgrad_z

  //end json file
  gradfile << std::setw(4) << " "
           << "}" << std::endl;

  gradfile << "]" << std::endl;
  gradfile.close();
}