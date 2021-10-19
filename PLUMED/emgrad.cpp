#include "/Data/Packages/Research/plumed2/src/isdb/colvar/Colvar.h"
#include "/Data/Packages/Research/plumed2/src/isdb/colvar/ActionRegister.h"
#include "/Data/Packages/Research/plumed2/src/isdb/core/PlumedMain.h"
#include "/Data/Packages/Research/plumed2/src/isdb/tools/Pbc.h"
#include "/Data/Packages/Research/plumed2/src/isdb/tools/OpenMP.h"
#include "/Data/Packages/Research/plumed2/src/isdb/tools/Communicator.h"

#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>

typedef double myfloat_t;
typedef myfloat_t mycomplex_t[2];
typedef std::vector <myfloat_t> myvector_t;
typedef std::vector <myvector_t> mymatrix_t;
typedef std::vector <PLMD::Vector> mycoord_t;

typedef double myfloat_t;
typedef myfloat_t mycomplex_t[2];
typedef std::vector <myfloat_t> myvector_t;
typedef std::vector <myvector_t> mymatrix_t;

typedef struct image {
  myfloat_t defocus;
  myvector_t q = myvector_t(4, 0.0);
  myvector_t q_inv = myvector_t(4, 0);
  myvector_t I;
  std::string fname;
} myimage_t;

//typedef struct image myimage_t;
typedef std::vector<myimage_t> mydataset_t;

#pragma omp declare reduction(vec_float_plus : std::vector<myfloat_t> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<myfloat_t>())) \
                              initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp declare reduction(vec_plmd_plus : std::vector<PLMD::Vector> : \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<PLMD::Vector>())) \
                              initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))


namespace PLMD {
namespace isdb {

class EmGrad : public Colvar {

 private:
  bool pbc, serial;
  myfloat_t sigma, cutoff, pixel_size, defocus, norm, fact;
  int n_pixels, n_atoms, n_neigh, n_imgs, rank, world_size, ntomp;

  std::string img_fname;
  mydataset_t exp_imgs;
  
  myvector_t grid;
  
  myfloat_t s_cv;

  Tensor virial;
  
  void read_coord(myvector_t &, myvector_t &, myvector_t &, myfloat_t &);
  void where(myvector_t &, std::vector<size_t> &, myfloat_t, myfloat_t);
  void read_exp_img(myimage_t *); 
  void create_rot_matrix(myvector_t &, mymatrix_t &);

  void quaternion_rotation(myvector_t &, mycoord_t &);
  void quaternion_rotation(myvector_t &, mycoord_t &, mycoord_t &);


  void calc_I(mycoord_t &, myvector_t &);
  void calc_I_omp(mycoord_t &, myvector_t &, int);

  void L2_grad(mycoord_t &, myvector_t &, myvector_t &,
               mycoord_t &, myfloat_t &);
  void L2_grad_omp(mycoord_t &, myvector_t &, myvector_t &,
               mycoord_t &, myfloat_t &, int ntomp);
  

 public:
  static void registerKeywords( Keywords& keys );
  explicit EmGrad(const ActionOptions&);
// active methods:
  void calculate() override;
};

PLUMED_REGISTER_ACTION(EmGrad,"EMGRAD")

void EmGrad::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords( keys );
  keys.add("atoms", "ATOMS","The atoms for which we will calculate the neighbors");
  keys.add("compulsory", "SIGMA","Standard deviation of the gaussians.");
  keys.add("compulsory", "CUTOFF","Neighbor cutoff.");
  keys.add("compulsory", "N_PIXELS","Number of pixels for projection.");
  keys.add("compulsory", "PIXEL_SIZE","Size of each pixel.");
  keys.add("compulsory", "IMG_PREFIX", "Prefix for the file with experimental images");
  keys.add("compulsory", "N_IMAGES","Number of experimental images.");
  keys.add("compulsory", "FACTOR", "Scaling factor for EMGRAD.");
}

EmGrad::EmGrad(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true)
{
  std::vector<AtomNumber> atoms;
  parseAtomList("ATOMS",atoms);
  if(atoms.size()<1)
    error("You should define at least one atom");
  
  parse("SIGMA",sigma);
  parse("CUTOFF",cutoff);
  parse("N_PIXELS",n_pixels);
  parse("PIXEL_SIZE",pixel_size);
  parse("FACTOR", fact);

  parse("IMG_PREFIX", img_fname);
  parse("N_IMAGES", n_imgs);

  serial = (n_imgs == 1 ? true : false);

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;
  checkRead();

  log.printf("  atoms involved : ");
  for(unsigned i=0; i<atoms.size(); ++i) log.printf("%d ",atoms[i].serial());
  log.printf("\n");
  log.printf("  standard deviation for gaussians : %1f\n", sigma);
  log.printf("  neighbor list cutoff : %lf\n", cutoff);
  log.printf("  Image file prefix : %s\n", img_fname.c_str());
  log.printf("  Number of images : %d\n", n_imgs);
  
  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");

  // log << " Bibliography" << plumed.cite("Bonomi, Camilloni, Bioinformatics, 33, 3999 (2017)") << "\n";

  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms(atoms);

  n_atoms = getNumberOfAtoms();

  // MPI stuff
  if (serial){
    exp_imgs.resize(n_imgs);

    exp_imgs[0].I = myvector_t(n_pixels*n_pixels, 0.0);
    exp_imgs[0].fname = img_fname + std::to_string(0) + ".txt";
    read_exp_img(&exp_imgs[0]);
  }

  else {

    world_size = comm.Get_size();
    rank = comm.Get_rank();

    int imgs_per_process = n_imgs/world_size;
    exp_imgs.resize(imgs_per_process);
    int start_img = rank*imgs_per_process;
    

    for (int i = 0; i < exp_imgs.size(); i++) {

      exp_imgs[i].I = myvector_t(n_pixels*n_pixels, 0.0);
      exp_imgs[i].fname = img_fname + std::to_string(i+start_img) + ".txt";
      read_exp_img(&exp_imgs[i]);
    }
  }
  
  norm = 1. / (2*M_PI * sigma*sigma * n_atoms);

  // Create grid
  myfloat_t min = -pixel_size * (n_pixels - 1)*0.5;
  grid.resize(n_pixels);

  for (int i=0; i<n_pixels; i++) grid[i] = min + pixel_size*i;

  n_neigh = (int) std::ceil(sigma * cutoff / pixel_size);

  ntomp = 2; //OpenMP::getNumThreads();
}

void EmGrad::create_rot_matrix(myvector_t &q, mymatrix_t &Q){

  //Definition of the quaternion rotation matrix 

  myfloat_t q00 = 1 - 2*std::pow(q[1],2) - 2*std::pow(q[2],2);
  myfloat_t q01 = 2*q[0]*q[1] - 2*q[2]*q[3];
  myfloat_t q02 = 2*q[0]*q[2] + 2*q[1]*q[3];
  myvector_t q0{ q00, q01, q02 };
  
  myfloat_t q10 = 2*q[0]*q[1] + 2*q[2]*q[3];
  myfloat_t q11 = 1 - 2*std::pow(q[0],2) - 2*std::pow(q[2],2);
  myfloat_t q12 = 2*q[1]*q[2] - 2*q[0]*q[3];
  myvector_t q1{ q10, q11, q12 };

  myfloat_t q20 = 2*q[0]*q[2] - 2*q[1]*q[3];
  myfloat_t q21 = 2*q[1]*q[2] + 2*q[0]*q[3];
  myfloat_t q22 = 1 - 2*std::pow(q[0],2) - 2*std::pow(q[1],2);
  myvector_t q2{ q20, q21, q22};

  Q = mymatrix_t{ q0, q1, q2 };
}

void EmGrad::quaternion_rotation(myvector_t &q, mycoord_t &r_ref){

/**
 * @brief Rotates a biomolecule using the quaternions rotation matrix
 *        according to (https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion)
 * 
 * @param q vector that stores the parameters for the rotation myvector_t (4)
 * @param x_data original coordinates x
 * @param y_data original coordinates y
 * @param z_data original coordinates z
 * @param x_r stores the rotated values x
 * @param y_r stores the rotated values x
 * @param z_r stores the rotated values x
 * 
 * @return void
 * 
 */

  //Definition of the quaternion rotation matrix 

  myfloat_t q00 = 1 - 2*std::pow(q[1],2) - 2*std::pow(q[2],2);
  myfloat_t q01 = 2*q[0]*q[1] - 2*q[2]*q[3];
  myfloat_t q02 = 2*q[0]*q[2] + 2*q[1]*q[3];
  myvector_t q0{ q00, q01, q02 };
  
  myfloat_t q10 = 2*q[0]*q[1] + 2*q[2]*q[3];
  myfloat_t q11 = 1 - 2*std::pow(q[0],2) - 2*std::pow(q[2],2);
  myfloat_t q12 = 2*q[1]*q[2] - 2*q[0]*q[3];
  myvector_t q1{ q10, q11, q12 };

  myfloat_t q20 = 2*q[0]*q[2] - 2*q[1]*q[3];
  myfloat_t q21 = 2*q[1]*q[2] + 2*q[0]*q[3];
  myfloat_t q22 = 1 - 2*std::pow(q[0],2) - 2*std::pow(q[1],2);
  myvector_t q2{ q20, q21, q22};

  mymatrix_t Q{ q0, q1, q2 };

  myfloat_t x_tmp, y_tmp, z_tmp;
  for (int i=0; i<n_atoms; i++){

    x_tmp = Q[0][0]*r_ref[i][0] + Q[0][1]*r_ref[i][1] + Q[0][2]*r_ref[i][2];
    y_tmp = Q[1][0]*r_ref[i][0] + Q[1][1]*r_ref[i][1] + Q[1][2]*r_ref[i][2];
    z_tmp = Q[2][0]*r_ref[i][0] + Q[2][1]*r_ref[i][1] + Q[2][2]*r_ref[i][2];

    r_ref[i][0] = x_tmp;
    r_ref[i][1] = y_tmp;
    r_ref[i][2] = z_tmp;
  }
}

void EmGrad::quaternion_rotation(myvector_t &q, mycoord_t &r_ref, mycoord_t &r_rot){

/**
 * @brief Rotates a biomolecule using the quaternions rotation matrix
 *        according to (https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion)
 * 
 * @param q vector that stores the parameters for the rotation myvector_t (4)
 * @param x_data original coordinates x
 * @param y_data original coordinates y
 * @param z_data original coordinates z
 * @param x_r stores the rotated values x
 * @param y_r stores the rotated values x
 * @param z_r stores the rotated values x
 * 
 * @return void
 * 
 */

  //Definition of the quaternion rotation matrix 

  myfloat_t q00 = 1 - 2*std::pow(q[1],2) - 2*std::pow(q[2],2);
  myfloat_t q01 = 2*q[0]*q[1] - 2*q[2]*q[3];
  myfloat_t q02 = 2*q[0]*q[2] + 2*q[1]*q[3];
  myvector_t q0{ q00, q01, q02 };
  
  myfloat_t q10 = 2*q[0]*q[1] + 2*q[2]*q[3];
  myfloat_t q11 = 1 - 2*std::pow(q[0],2) - 2*std::pow(q[2],2);
  myfloat_t q12 = 2*q[1]*q[2] - 2*q[0]*q[3];
  myvector_t q1{ q10, q11, q12 };

  myfloat_t q20 = 2*q[0]*q[2] - 2*q[1]*q[3];
  myfloat_t q21 = 2*q[1]*q[2] + 2*q[0]*q[3];
  myfloat_t q22 = 1 - 2*std::pow(q[0],2) - 2*std::pow(q[1],2);
  myvector_t q2{ q20, q21, q22};

  mymatrix_t Q{ q0, q1, q2 };

  for (int i=0; i<n_atoms; i++){

    r_rot[i][0] = Q[0][0]*r_ref[i][0] + Q[0][1]*r_ref[i][1] + Q[0][2]*r_ref[i][2];
    r_rot[i][1] = Q[1][0]*r_ref[i][0] + Q[1][1]*r_ref[i][1] + Q[1][2]*r_ref[i][2];
    r_rot[i][2] = Q[2][0]*r_ref[i][0] + Q[2][1]*r_ref[i][1] + Q[2][2]*r_ref[i][2];
  }
}

void EmGrad::where(myvector_t &inp_vec, std::vector<size_t> &out_vec, 
                    myfloat_t x_res, myfloat_t limit){  

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
              [&](myfloat_t i){ return std::abs(i - x_res) <= limit; });
    
    //while the pointer doesn't point to the end of the vector
    while (it != std::end(inp_vec)) {

        //save the value of the last index found
        out_vec.emplace_back(std::distance(std::begin(inp_vec), it));

        //calculate the next item that satisfies the condition
        it = std::find_if(std::next(it), std::end(inp_vec), 
             [&](myfloat_t i){return std::abs(i - x_res) <= limit;});
    }
}

void EmGrad::read_exp_img(myimage_t *IMG){

  std::ifstream file;
  file.open(IMG->fname);

  if (!file.good()){
    error("The file " + IMG->fname + " does not exist");
  }

  //Read defocus
  file >> IMG->defocus;
  
  // Read quaternions
  for (int i=0; i<4; i++) file >> IMG->q[i];

  // Fill inverse quat
  myfloat_t quat_abs = IMG->q[0] * IMG->q[0] + 
                       IMG->q[1] * IMG->q[1] +
                       IMG->q[2] * IMG->q[2] +
                       IMG->q[3] * IMG->q[3];

  if (quat_abs > 0.0){

    IMG->q_inv[0] = -IMG->q[0] / quat_abs;
    IMG->q_inv[1] = -IMG->q[1] / quat_abs;
    IMG->q_inv[2] = -IMG->q[2] / quat_abs;
    IMG->q_inv[3] =  IMG->q[3] / quat_abs;
  }

  //Read image
  for (int i=0; i<n_pixels*n_pixels; i++){

    file >> IMG->I[i];
  }
  file.close();
}

void EmGrad::calc_I(mycoord_t &r_a, myvector_t &I_c){

  /**
   * @brief Calculates the image from the 3D model by representing the atoms as gaussians and using a 2d Grid
   * 
   * @param x_coord original coordinates x
   * @param y_coord original coordinates y
   * @param z_coord original coordinates z
   * @param sigma standard deviation for the gaussians, equal for all the atoms  (myfloat_t)
   * @paramsigma_reachnumber of sigmas used for cutoff (int)
   * @param n_pixels Resolution of the calculated image
   * @param Icalc Matrix used to store the calculated image
   * @param x values in x for the grid
   * @param y values in y for the grid
   * 
   * @return void
   * 
   */


  int m_x, m_y;
  int ind_i, ind_j;

  // std::vector<size_t> x_sel, y_sel;
  myvector_t gauss_x(2*n_neigh+3, 0.0);
  myvector_t gauss_y(2*n_neigh+3, 0.0);

  for (int atom=0; atom<n_atoms; atom++){

    m_x = (int) std::round(abs(r_a[atom][0] - grid[0])/pixel_size);
    m_y = (int) std::round(abs(r_a[atom][1] - grid[0])/pixel_size);

    for (int i=0; i<=2*n_neigh+2; i++){
      
      ind_i = m_x - n_neigh - 1 + i;
      ind_j = m_y - n_neigh - 1 + i;

      if (ind_i<0 || ind_i>=n_pixels) gauss_x[i] = 0;
      else {

        myfloat_t expon_x = (grid[ind_i] - r_a[atom][0])/sigma;
        gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
      }
            
      if (ind_j<0 || ind_j>=n_pixels) gauss_y[i] = 0;
      else{

        myfloat_t expon_y = (grid[ind_j] - r_a[atom][1])/sigma;
        gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
      }
    }

    //Calculate the image and the gradient
    for (int i=0; i<=2*n_neigh+2; i++){
      
      ind_i = m_x - n_neigh - 1 + i;
      if (ind_i<0 || ind_i>=n_pixels) continue;
      
      for (int j=0; j<=2*n_neigh+2; j++){

        ind_j = m_y - n_neigh - 1 + j;
        if (ind_j<0 || ind_j>=n_pixels) continue;

        I_c[ind_i*n_pixels + ind_j] += gauss_x[i]*gauss_y[j];
      }
    }
  }

  for (int i=0; i<n_pixels*n_pixels; i++){ 
  
    I_c[i] *= norm;
  }
}

void EmGrad::L2_grad(mycoord_t &r_a, myvector_t &I_c, myvector_t &I_e,
                      mycoord_t &gr_r, myfloat_t &s){

  /**
   * @brief Calculates the image from the 3D model by representing the atoms as gaussians and using a 2d Grid
   * 
   * @param x_a original coordinates x 
    * @param I_c Matrix used to store the calculated image
   * @param gr_x vectors for the gradient in x
   * @param gr_y vectors for the gradient in y
   * 
   * @return void
   * 
   */

  int m_x, m_y;
  int ind_i, ind_j;
  
  myvector_t gauss_x(2*n_neigh+3, 0.0);
  myvector_t gauss_y(2*n_neigh+3, 0.0);

  for (int atom=0; atom<n_atoms; atom++){

    m_x = (int) std::round(abs(r_a[atom][0] - grid[0])/pixel_size);
    m_y = (int) std::round(abs(r_a[atom][1] - grid[0])/pixel_size);

    for (int i=0; i<=2*n_neigh+2; i++){
      
      ind_i = m_x - n_neigh - 1 + i;
      ind_j = m_y - n_neigh - 1 + i;

      if (ind_i<0 || ind_i>=n_pixels) gauss_x[i] = 0;
      else {

        myfloat_t expon_x = (grid[ind_i] - r_a[atom][0])/sigma;
        gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
      }
            
      if (ind_j<0 || ind_j>=n_pixels) gauss_y[i] = 0;
      else{

        myfloat_t expon_y = (grid[ind_j] - r_a[atom][1])/sigma;
        gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
      }
    }

    myfloat_t s1=0, s2=0;

    //Calculate the image and the gradient
    for (int i=0; i<=2*n_neigh+2; i++){
      
      ind_i = m_x - n_neigh - 1 + i;
      if (ind_i<0 || ind_i>=n_pixels) continue;
      
      for (int j=0; j<=2*n_neigh+2; j++){

        ind_j = m_y - n_neigh - 1 + j;
        if (ind_j<0 || ind_j>=n_pixels) continue;

        s1 += (I_c[ind_i*n_pixels + ind_j] - I_e[ind_i*n_pixels + ind_j]) * (grid[ind_i] - r_a[atom][0]) * gauss_x[i] * gauss_y[j];
        s2 += (I_c[ind_i*n_pixels + ind_j] - I_e[ind_i*n_pixels + ind_j]) * (grid[ind_j] - r_a[atom][1]) * gauss_x[i] * gauss_y[j];
      }
    }

    gr_r[atom][0] = s1 * 2*norm / (sigma * sigma);
    gr_r[atom][1] = s2 * 2*norm / (sigma * sigma);
    gr_r[atom][2] = 0;
  }
  
  s = 0;
  for (int i=0; i<n_pixels*n_pixels; i++) s += (I_c[i] - I_e[i]) * (I_c[i] - I_e[i]);
}

void EmGrad::calc_I_omp(mycoord_t &r_a, myvector_t &I_c, int ntomp){

  /**
   * @brief Calculates the image from the 3D model by representing the atoms as gaussians and using a 2d Grid
   * 
   * @param x_coord original coordinates x
   * @param y_coord original coordinates y
   * @param z_coord original coordinates z
   * @param sigma standard deviation for the gaussians, equal for all the atoms  (myfloat_t)
   * @paramsigma_reachnumber of sigmas used for cutoff (int)
   * @param n_pixels Resolution of the calculated image
   * @param Icalc Matrix used to store the calculated image
   * @param x values in x for the grid
   * @param y values in y for the grid
   * 
   * @return void
   * 
   */

  #pragma omp parallel num_threads(ntomp)
  {
    int m_x, m_y;
    int ind_i, ind_j;

    // std::vector<size_t> x_sel, y_sel;
    myvector_t gauss_x(2*n_neigh+3, 0.0);
    myvector_t gauss_y(2*n_neigh+3, 0.0);

    #pragma omp for reduction(vec_float_plus: I_c)
    for (int atom=0; atom<n_atoms; atom++){

      m_x = (int) std::round(abs(r_a[atom][0] - grid[0])/pixel_size);
      m_y = (int) std::round(abs(r_a[atom][1] - grid[0])/pixel_size);

      for (int i=0; i<=2*n_neigh+2; i++){
        
        ind_i = m_x - n_neigh - 1 + i;
        ind_j = m_y - n_neigh - 1 + i;

        if (ind_i<0 || ind_i>=n_pixels) gauss_x[i] = 0;
        else {

          myfloat_t expon_x = (grid[ind_i] - r_a[atom][0])/sigma;
          gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
        }
              
        if (ind_j<0 || ind_j>=n_pixels) gauss_y[i] = 0;
        else{

          myfloat_t expon_y = (grid[ind_j] - r_a[atom][1])/sigma;
          gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
        }
      }

      //Calculate the image and the gradient
      for (int i=0; i<=2*n_neigh+2; i++){
        
        ind_i = m_x - n_neigh - 1 + i;
        if (ind_i<0 || ind_i>=n_pixels) continue;
        
        for (int j=0; j<=2*n_neigh+2; j++){

          ind_j = m_y - n_neigh - 1 + j;
          if (ind_j<0 || ind_j>=n_pixels) continue;

          I_c[ind_i*n_pixels + ind_j] += gauss_x[i]*gauss_y[j];
        }
      }
    }

    #pragma omp for
    for (int i=0; i<n_pixels*n_pixels; i++){ 
    
      I_c[i] *= norm;
    }
  }
}

void EmGrad::L2_grad_omp(mycoord_t &r_a, myvector_t &I_c, myvector_t &I_e,
                      mycoord_t &gr_r, myfloat_t &s, int ntomp){

  /**
   * @brief Calculates the image from the 3D model by representing the atoms as gaussians and using a 2d Grid
   * 
   * @param x_a original coordinates x 
    * @param I_c Matrix used to store the calculated image
   * @param gr_x vectors for the gradient in x
   * @param gr_y vectors for the gradient in y
   * 
   * @return void
   * 
   */

  s = 0;

  myfloat_t Ccc = 0.0, Coc = 0.0;

  for (size_t i=0; i<I_c.size(); i++){

    Ccc += I_c[i] * I_c[i];
    Coc += I_e[i] * I_c[i];
  }

  myfloat_t N = Coc/Ccc;
  myfloat_t NORM = 2*N * norm / (sigma * sigma);

  #pragma omp parallel num_threads(ntomp)
  {

    int m_x, m_y;
    int ind_i, ind_j;
    
    myvector_t gauss_x(2*n_neigh+3, 0.0);
    myvector_t gauss_y(2*n_neigh+3, 0.0);

    #pragma omp for reduction(vec_float_plus: I_c)
    for (int atom=0; atom<n_atoms; atom++){

      m_x = (int) std::round(abs(r_a[atom][0] - grid[0])/pixel_size);
      m_y = (int) std::round(abs(r_a[atom][1] - grid[0])/pixel_size);

      for (int i=0; i<=2*n_neigh+2; i++){
        
        ind_i = m_x - n_neigh - 1 + i;
        ind_j = m_y - n_neigh - 1 + i;

        if (ind_i<0 || ind_i>=n_pixels) gauss_x[i] = 0;
        else {

          myfloat_t expon_x = (grid[ind_i] - r_a[atom][0])/sigma;
          gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
        }
              
        if (ind_j<0 || ind_j>=n_pixels) gauss_y[i] = 0;
        else{

          myfloat_t expon_y = (grid[ind_j] - r_a[atom][1])/sigma;
          gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
        }
      }

      myfloat_t s1=0, s2=0;

      //Calculate the image and the gradient
      for (int i=0; i<=2*n_neigh+2; i++){
        
        ind_i = m_x - n_neigh - 1 + i;
        if (ind_i<0 || ind_i>=n_pixels) continue;
        
        for (int j=0; j<=2*n_neigh+2; j++){

          ind_j = m_y - n_neigh - 1 + j;
          if (ind_j<0 || ind_j>=n_pixels) continue;

          s1 += (N*I_c[ind_i*n_pixels + ind_j] - I_e[ind_i*n_pixels + ind_j]) * \
                (grid[ind_i] - r_a[atom][0]) * gauss_x[i] * gauss_y[j];

          s2 += (N*I_c[ind_i*n_pixels + ind_j] - I_e[ind_i*n_pixels + ind_j]) * \
                (grid[ind_j] - r_a[atom][1]) * gauss_x[i] * gauss_y[j];
        }
      }

      gr_r[atom][0] = s1 * NORM;
      gr_r[atom][1] = s2 * NORM;
      gr_r[atom][2] = 0;
    }
    
    s = 0;
    for (int i=0; i<n_pixels*n_pixels; i++) s += (N*I_c[i] - I_e[i]) * (N*I_c[i] - I_e[i]);
  }
}

// calculator
void EmGrad::calculate() {

  if(pbc) makeWhole();

  mycoord_t pos = getPositions();
  mycoord_t emgrad_der(n_atoms);

  s_cv = 0;

  if (ntomp > 1 && !serial){

    // #pragma omp parallel num_threads(ntomp)
    // {
    //   mycoord_t grad_tmp(n_atoms);
    //   mycoord_t r_rot(n_atoms);
    //   myfloat_t s_tmp;
    //   myvector_t Icalc(n_pixels*n_pixels, 0.0);

    //   #pragma omp for reduction(vec_plmd_plus : emgrad_der) reduction(+ : s_cv)
    //   for (int i=0; i<exp_imgs.size(); i++){

    //     quaternion_rotation(exp_imgs[i].q, pos, r_rot);

    //     calc_I(r_rot, Icalc);
    //     L2_grad(r_rot, Icalc, exp_imgs[i].I, grad_tmp, s_tmp);
    //     quaternion_rotation(exp_imgs[i].q_inv, grad_tmp);

    //     s_cv += s_tmp;

    //     for (int j=0; j<n_atoms; j++){

    //       emgrad_der[j] += grad_tmp[j];
    //     }

    //     Icalc = myvector_t(n_pixels*n_pixels, 0.0);
    //   }
    // }

    mycoord_t grad_tmp(n_atoms);
    mycoord_t r_rot(n_atoms);
    myfloat_t s_tmp;
    myvector_t Icalc(n_pixels*n_pixels, 0.0);

    for (int i=0; i<exp_imgs.size(); i++){

      quaternion_rotation(exp_imgs[i].q, pos, r_rot);

      calc_I_omp(r_rot, Icalc, ntomp);
      L2_grad_omp(r_rot, Icalc, exp_imgs[i].I, grad_tmp, s_tmp, ntomp);
      quaternion_rotation(exp_imgs[i].q_inv, grad_tmp);

      s_cv += s_tmp;

      for (int j=0; j<n_atoms; j++){

        emgrad_der[j] += grad_tmp[j];
      }

      Icalc = myvector_t(n_pixels*n_pixels, 0.0);
    }
  }
  
  else {

    myvector_t Icalc(n_pixels*n_pixels, 0.0);


    quaternion_rotation(exp_imgs[0].q, pos);
    calc_I(pos, Icalc);
    L2_grad(pos, Icalc, exp_imgs[0].I, emgrad_der, s_cv);
    quaternion_rotation(exp_imgs[0].q_inv, emgrad_der);
  }

  if (!serial){

    comm.Sum(s_cv);
    comm.Sum(&emgrad_der[0][0], 3*emgrad_der.size());
  }
  
  for(unsigned i=0; i<emgrad_der.size(); ++i) setAtomsDerivatives(i, fact*emgrad_der[i]);
  setValue(fact*s_cv);
  // setBoxDerivatives(virial);
  setBoxDerivativesNoPbc();
}
}
}
