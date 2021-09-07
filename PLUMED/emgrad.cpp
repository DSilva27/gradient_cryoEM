#include "/Data/Packages/Research/plumed2/src/isdb/colvar/Colvar.h"
#include "/Data/Packages/Research/plumed2/src/isdb/colvar/ActionRegister.h"
#include "/Data/Packages/Research/plumed2/src/isdb/core/PlumedMain.h"
#include "/Data/Packages/Research/plumed2/src/isdb/tools/Pbc.h"
#include "/Data/Packages/Research/plumed2/src/isdb/tools/OpenMP.h"

#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>

typedef double myfloat_t;
typedef myfloat_t mycomplex_t[2];
typedef std::vector <myfloat_t> myvector_t;
typedef std::vector <myvector_t> mymatrix_t;

#pragma omp declare reduction(vec_float_plus : std::vector<myfloat_t> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<myfloat_t>())) \
                              initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

namespace PLMD {
namespace isdb {

class EmGrad : public Colvar {

 private:
  bool pbc;
  myfloat_t sigma, cutoff, pixel_size, defocus, norm;
  int n_pixels, n_atoms, n_neigh;

  myvector_t quat, quat_inv;

  myvector_t x_grid, y_grid;
  //mymatrix_t Icalc;
  //mymatrix_t Iexp;
  myvector_t Icalc;
  myvector_t Iexp;

  mymatrix_t Q_rot;

  std::vector<Vector> pos, emgrad_der;
  myvector_t gauss_x;
  myvector_t gauss_y;
  myfloat_t s_cv;

  Tensor virial;
  
  void read_coord(myvector_t &, myvector_t &, myvector_t &, myfloat_t &);
  void arange(myvector_t &, myfloat_t, myfloat_t, myfloat_t);
  void where(myvector_t &, std::vector<size_t> &, myfloat_t, myfloat_t);
  void read_exp_img(std::string); 
  void create_rot_matrix(myvector_t &, mymatrix_t &);
  void quaternion_rotation(mymatrix_t &, std::vector<Vector> &);


  void calc_I(std::vector<Vector> &, mymatrix_t &);
  void L2_grad(std::vector<Vector> &, mymatrix_t &, mymatrix_t &,
               std::vector<Vector> &, myfloat_t &);
  

 public:
  static void registerKeywords( Keywords& keys );
  explicit EmGrad(const ActionOptions&);
// active methods:
  void calculate() override;
};

PLUMED_REGISTER_ACTION(EmGrad,"EMGRAD")

void EmGrad::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords( keys );
  keys.add("atoms","ATOMS","The atoms for which we will calculate the neighbors");
  keys.add("compulsory","SIGMA","Standard deviation of the gaussians.");
  keys.add("compulsory","CUTOFF","Neighbor cutoff.");
  keys.add("compulsory","N_PIXELS","Number of pixels for projection.");
  keys.add("compulsory","PIXEL_SIZE","Size of each pixel.");
  keys.add("compulsory", "IMG_FILE", "File with experimental images");
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

  std::string IMG_file;
  parse("IMG_FILE", IMG_file);

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;
  checkRead();

  log.printf("  atoms involved : ");
  for(unsigned i=0; i<atoms.size(); ++i) log.printf("%d ",atoms[i].serial());
  log.printf("\n");
  log.printf("  standard deviation for gaussians : %1f\n", sigma);
  log.printf("  neighbor list cutoff : %lf\n", cutoff);
  log.printf("  Image file : %s\n", IMG_file.c_str());
  
  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");

  // log << " Bibliography" << plumed.cite("Bonomi, Camilloni, Bioinformatics, 33, 3999 (2017)") << "\n";

  

  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms(atoms);

  n_atoms = getNumberOfAtoms();
  emgrad_der.resize(n_atoms);
  // Icalc = mymatrix_t(n_pixels, myvector_t(n_pixels, 0));
  // Iexp = mymatrix_t(n_pixels, myvector_t(n_pixels, 0));
  Icalc = myvector_t(n_pixels*n_pixels, 0.0);
  Iexp = myvector_t(n_pixels*n_pixels, 0.0);

  quat = myvector_t(n_pixels, 0);

  read_exp_img(IMG_file);
  norm = 1. / (2*M_PI * sigma*sigma * n_atoms);

  //Calculate minimum and maximum values for the linspace-like vectors x and y
  myfloat_t min = -pixel_size * (n_pixels - 1)*0.5;
  myfloat_t max = pixel_size * (n_pixels - 1)*0.5 + pixel_size;

  //Assign memory space required to fill the vectors
  x_grid.resize(n_pixels); y_grid.resize(n_pixels);

  //Generate them
  arange(x_grid, min, max, pixel_size);
  arange(y_grid, min, max, pixel_size);

  n_neigh = (int) std::ceil(sigma * cutoff / pixel_size);

  gauss_x = myvector_t(2*n_neigh+3, 0.0);
  gauss_y = myvector_t(2*n_neigh+3, 0.0);
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

void EmGrad::quaternion_rotation(mymatrix_t &Q, std::vector<Vector> &P){

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

  myfloat_t x_tmp, y_tmp, z_tmp;

  for (unsigned int i=0; i<n_atoms; i++){

    x_tmp = P[i][0]*Q[0][0] + P[i][1]*Q[0][1] + P[i][2]*Q[0][2];
    y_tmp = P[i][0]*Q[1][0] + P[i][1]*Q[1][1] + P[i][2]*Q[1][2];
    z_tmp = P[i][0]*Q[2][0] + P[i][1]*Q[2][1] + P[i][2]*Q[2][2];

    P[i][0] = x_tmp;
    P[i][1] = y_tmp;
    P[i][2] = z_tmp;
  }
}

void EmGrad::arange(myvector_t &out_vec, myfloat_t xo, myfloat_t xf, myfloat_t dx){

    /**
     * @brief Creates a linearly spaced vector from xo to xf with dimension n
     * 
     * @param out_vec vector where the values will be stored myvector_t
     * @param xo initial value (myfloat_t)
     * @param xf final value (myfloat_t)
     * 
     * @return void
     */

    //spacing for the values
    myfloat_t a_x = dx;

    //fills a vector with values separated by a_x starting from xo [xo, xo + a, xo + 2a, ...]
    std::generate(out_vec.begin(), out_vec.end(), [n=0, &xo, &a_x]() mutable { return n++ * a_x + xo; });   
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

void EmGrad::read_exp_img(std::string fname){

  
  std::ifstream file;
  file.open(fname);

  if (!file.good()){
    error("The file " + fname + " does not exist");
  }

  //Read defocus
  file >> defocus;
  
  //Read quaternions
  for (int i=0; i<4; i++) file >> quat[i];

    //Create inverse quat
  quat_inv = myvector_t(4, 0);

  float quat_abs = quat[0]*quat[0] + 
                   quat[1]*quat[1] +
                   quat[2]*quat[2] +
                   quat[3]*quat[3];

  quat_inv[0] = -quat[0] / quat_abs;
  quat_inv[1] = -quat[1] / quat_abs;
  quat_inv[2] = -quat[2] / quat_abs;
  quat_inv[3] =  quat[3] / quat_abs;

  //Read image
  for (int i=0; i<n_pixels*n_pixels; i++){

    file >> Iexp[i];
  }
}

void EmGrad::calc_I(std::vector<Vector> &r_a, mymatrix_t &I_c){

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

  #pragma omp for
  for (int atom=0; atom<n_atoms; atom++){

    m_x = (int) std::round(abs(r_a[atom][0] - x_grid[0])/pixel_size);
    m_y = (int) std::round(abs(r_a[atom][1] - y_grid[0])/pixel_size);

    for (int i=0; i<=2*n_neigh+2; i++){
      
      ind_i = m_x - n_neigh - 1 + i;
      ind_j = m_y - n_neigh - 1 + i;

      if (ind_i<0 || ind_i>=n_pixels) gauss_x[i] = 0;
      else {

        myfloat_t expon_x = (x_grid[ind_i] - r_a[atom][0])/sigma;
        gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
      }
            
      if (ind_j<0 || ind_j>=n_pixels) gauss_y[i] = 0;
      else{

        myfloat_t expon_y = (y_grid[ind_j] - r_a[atom][1])/sigma;
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

        I_c[ind_i][ind_j] += gauss_x[i]*gauss_y[j];
      }
    }
  }

  for (int i=0; i<n_pixels; i++){ 
    for (int j=0; j<n_pixels; j++){ 
        
      I_c[i][j] *= norm;
    }
  }
}

void EmGrad::L2_grad(std::vector<Vector> &r_a, mymatrix_t &I_c, mymatrix_t &I_e,
                      std::vector<Vector> &gr_r, myfloat_t &s){

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

  // std::vector<size_t> x_sel, y_sel;
  #pragma omp for
  for (int atom=0; atom<n_atoms; atom++){

    m_x = (int) std::round(abs(r_a[atom][0] - x_grid[0])/pixel_size);
    m_y = (int) std::round(abs(r_a[atom][1] - y_grid[0])/pixel_size);

    for (int i=0; i<=2*n_neigh+2; i++){
      
      ind_i = m_x - n_neigh - 1 + i;
      ind_j = m_y - n_neigh - 1 + i;

      if (ind_i<0 || ind_i>=n_pixels) gauss_x[i] = 0;
      else {

        myfloat_t expon_x = (x_grid[ind_i] - r_a[atom][0])/sigma;
        gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
      }
            
      if (ind_j<0 || ind_j>=n_pixels) gauss_y[i] = 0;
      else{

        myfloat_t expon_y = (y_grid[ind_j] - r_a[atom][1])/sigma;
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

        s1 += (I_c[ind_i][ind_j] - I_e[ind_i][ind_j]) * (x_grid[ind_i] - r_a[atom][0]) * gauss_x[i] * gauss_y[j];
        s2 += (I_c[ind_i][ind_j] - I_e[ind_i][ind_j]) * (y_grid[ind_j] - r_a[atom][1]) * gauss_x[i] * gauss_y[j];
      }
    }

    gr_r[atom][0] = s1 * 2*norm / (sigma * sigma);
    gr_r[atom][1] = s2 * 2*norm / (sigma * sigma);
  }
  
  s = 0;
  for (int i=0; i<n_pixels; i++){ 
    for (int j=0; j<n_pixels; j++){ 
      
      s += (I_c[i][j] - I_e[i][j]) * (I_c[i][j] - I_e[i][j]);
    }
  }
}


// calculator
void EmGrad::calculate() {

  if(pbc) makeWhole();

  // Reset image
  //Icalc = mymatrix_t(n_pixels, myvector_t(n_pixels, 0));
  Icalc = myvector_t(n_pixels*n_pixels, 0.0);

  pos = getPositions();
  s_cv = 0;

  #pragma omp parallel num_threads(OpenMP::getNumThreads())
  {
    int m_x, m_y;
    int ind_i, ind_j;

    // std::vector<size_t> x_sel, y_sel;
    myvector_t gauss_x(2*n_neigh+3, 0.0);
    myvector_t gauss_y(2*n_neigh+3, 0.0);

    // mymatrix_t Icalc_thr(n_pixels, myvector_t(n_pixels, 0.0));

    #pragma omp for reduction(vec_float_plus : Icalc)
    for (int atom=0; atom<n_atoms; atom++){

      m_x = (int) std::round(abs(pos[atom][0] - x_grid[0])/pixel_size);
      m_y = (int) std::round(abs(pos[atom][1] - y_grid[0])/pixel_size);

      #pragma omp simd
      for (int i=0; i<=2*n_neigh+2; i++){
        
        ind_i = m_x - n_neigh - 1 + i;
        ind_j = m_y - n_neigh - 1 + i;

        if (ind_i<0 || ind_i>=n_pixels) gauss_x[i] = 0;
        else {

          myfloat_t expon_x = (x_grid[ind_i] - pos[atom][0])/sigma;
          gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
        }
              
        if (ind_j<0 || ind_j>=n_pixels) gauss_y[i] = 0;
        else{

          myfloat_t expon_y = (y_grid[ind_j] - pos[atom][1])/sigma;
          gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
        }
      }

      //Calculate the image and the gradient
      for (int i=0; i<=2*n_neigh+2; i++){
        
        ind_i = m_x - n_neigh - 1 + i;
        if (ind_i<0 || ind_i>=n_pixels) continue;
        
        #pragma omp simd
        for (int j=0; j<=2*n_neigh+2; j++){

          ind_j = m_y - n_neigh - 1 + j;
          if (ind_j<0 || ind_j>=n_pixels) continue;

          //Icalc_thr[ind_i][ind_j] += gauss_x[i]*gauss_y[j];
          //Icalc[ind_i][ind_j] += gauss_x[i]*gauss_y[j];
          Icalc[ind_i*n_pixels + ind_j] += gauss_x[i]*gauss_y[j];
        }
      }
    }

    #pragma omp for
    for (int i=0; i<n_pixels; i++){ 
      #pragma omp simd
      for (int j=0; j<n_pixels; j++){ 
          
        //Icalc[i][j] += Icalc_thr[i][j]*norm;
        //Icalc[i][j] *= norm;
        Icalc[i*n_pixels + j] *= norm;
      }
    }
    

    #pragma omp for reduction(+:s_cv)
    for (int i=0; i<n_pixels; i++){ 
      #pragma omp simd
      for (int j=0; j<n_pixels; j++){ 
        
        //s_cv += (Icalc[i][j] - Iexp[i][j]) * (Icalc[i][j] - Iexp[i][j]);
        s_cv += (Icalc[i*n_pixels + j] - Iexp[i*n_pixels + j]) * (Icalc[i*n_pixels + j] - Iexp[i*n_pixels + j]);
      }
    }

    #pragma omp for
    for (int atom=0; atom<n_atoms; atom++){

      m_x = (int) std::round(abs(pos[atom][0] - x_grid[0])/pixel_size);
      m_y = (int) std::round(abs(pos[atom][1] - y_grid[0])/pixel_size);

      #pragma omp simd
      for (int i=0; i<=2*n_neigh+2; i++){
        
        ind_i = m_x - n_neigh - 1 + i;
        ind_j = m_y - n_neigh - 1 + i;

        if (ind_i<0 || ind_i>=n_pixels) gauss_x[i] = 0;
        else {

          myfloat_t expon_x = (x_grid[ind_i] - pos[atom][0])/sigma;
          gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
        }
              
        if (ind_j<0 || ind_j>=n_pixels) gauss_y[i] = 0;
        else{

          myfloat_t expon_y = (y_grid[ind_j] - pos[atom][1])/sigma;
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

          //s1 += (Icalc[ind_i][ind_j] - Iexp[ind_i][ind_j]) * (x_grid[ind_i] - pos[atom][0]) * gauss_x[i] * gauss_y[j];
          //s2 += (Icalc[ind_i][ind_j] - Iexp[ind_i][ind_j]) * (y_grid[ind_j] - pos[atom][1]) * gauss_x[i] * gauss_y[j];
          s1 += (Icalc[ind_i*n_pixels + ind_j] - Iexp[ind_i*n_pixels + ind_j]) * (x_grid[ind_i] - pos[atom][0]) * gauss_x[i] * gauss_y[j];
          s2 += (Icalc[ind_i*n_pixels + ind_j] - Iexp[ind_i*n_pixels + ind_j]) * (y_grid[ind_j] - pos[atom][1]) * gauss_x[i] * gauss_y[j];
        }
      }

      emgrad_der[atom][0] = s1 * 2*norm / (sigma * sigma);
      emgrad_der[atom][1] = s2 * 2*norm / (sigma * sigma);
    }
    
    #pragma omp for
    for(unsigned int i=0; i<getNumberOfAtoms(); i++) setAtomsDerivatives(i, emgrad_der[i]);
  }
  

  setValue(s_cv);
  // setBoxDerivatives(virial);
  setBoxDerivativesNoPbc();
}
}
}
