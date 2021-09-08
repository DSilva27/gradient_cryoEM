#include "gradcv.h"
#include <chrono>

#pragma omp declare reduction(vec_float_plus : std::vector<myfloat_t> : \
                              std::transform(omp_out.begin(), omp_out.end(), \
                              omp_in.begin(), omp_out.begin(), std::plus<myfloat_t>())) \
                              initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

Grad_cv::Grad_cv(){}

//Grad_cv::~Grad_cv(){}

void Grad_cv::init_variables(std::string pf, std::string cf, 
                            std::string imf, std::string gradf,
                            const char *type){

  p_type = type;

  params_file = "data/input/" + pf;
  coords_file = "data/input/" + cf;
  image_file = "data/images/" + imf;
  json_file = "data/output/" + gradf;

  //############################### Read parameters and coordinates #####################################
  read_coord();
  read_parameters(params_file);
  n_pixels_fft_1d = n_pixels/2 + 1;
  n_neigh = (int) std::ceil(sigma_cv * sigma_reach / pixel_size);

  // Initialize and read experimental images
  exp_imgs = mydataset_t(n_imgs);

  for (int i = 0; i < n_imgs; i++){

    exp_imgs[i].inten = myvector_t(n_pixels*n_pixels, 0.0);
    read_exp_img("data/images/Icalc_"+std::to_string(i)+".txt", &exp_imgs[i]);
  }

  grad_r = myvector_t(3*n_atoms, 0.0);

  //If we are creating images
  if (strcmp(p_type, "D") == 0){

    //Generate random defocus and calculate the phase
    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_real_distribution<myfloat_t> dist_def(min_defocus, max_defocus);
    
    defocus = dist_def(engine);
    phase = defocus * M_PI * 2. * 10000 * elecwavel;

    std::cout << "Defocus used: " << defocus << std::endl;

      //############################ CALCULATING RANDOM QUATERNIONS #######################################
    //Create a uniform distribution from 0 to 1
    std::uniform_real_distribution<myfloat_t> dist_quat(0, 1);
    myfloat_t u1, u2, u3;

      //Generate random numbers betwwen 0 and 1
    u1 = dist_quat(engine); u2 = dist_quat(engine); u3 = dist_quat(engine);
    quat = myvector_t(4, 0);

    //Random quaternion vector, check Shoemake, Graphic Gems III, p. 124-132
    quat[0] = std::sqrt(1 - u1) * sin(2 * M_PI * u2);
    quat[1] = std::sqrt(1 - u1) * cos(2 * M_PI * u2);
    quat[2] = std::sqrt(u1) * sin(2 * M_PI * u3);
    quat[3] = std::sqrt(u1) * cos(2 * M_PI * u3);
  } 

/*   else if (strcmp(p_type, "G") == 0){

    Iexp = mymatrix_t(n_pixels, myvector_t(n_pixels, 0));
    quat = myvector_t(n_pixels, 0);
    quat_inv = myvector_t(n_pixels, 0);

    
    //Turn grad_* into a n_pixels vector and fill it with zeros
    grad_r = mymatrix_t(n_atoms, myvector_t(3, 0.0));

    std::cout << "Variables initialized" << std::endl;

    //#################### Read experimental image (includes defocus and quaternions) ###########################
    read_exp_img(image_file, defocus, quat, quat_inv, Iexp);
    phase = defocus * M_PI * 2. * 10000 * elecwavel;
  } */

  //######################### Preparing FFTWs and allocating memory for images and gradients ##################
  //Prepare FFTs
  prepare_FFTs();
  
  //Calculate minimum and maximum values for the linspace-like vectors x and y
  grid_min = -pixel_size * (n_pixels - 1)*0.5;
  grid_max = pixel_size * (n_pixels - 1)*0.5 + pixel_size;

  //Assign memory space required to fill the vectors
  grid.resize(n_pixels); 

  //Generate them
  arange(grid, grid_min, grid_max, pixel_size);

  norm = 1. / (2*M_PI * sigma_cv*sigma_cv * n_atoms);

  std::cout << "Variables initialized" << std::endl;
}

void Grad_cv::read_coord(){

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

  std::ifstream coord_file;
  coord_file.open(coords_file);

  std::cout << "\n +++++++++++++++++++++++++++++++++++++++++ \n";
  std::cout << "\n   READING EM2D COORDINATES            \n\n";
  std::cout << " +++++++++++++++++++++++++++++++++++++++++ \n";
  
  if (!coord_file.good()){
    myError("Opening file: %s", coords_file.c_str());
  }

  coord_file >> n_atoms;
  r_coord = myvector_t(3*n_atoms, 0.0);

  for (int i=0; i<3*n_atoms; i++){

    coord_file >> r_coord[i];
  } 

  std::cout << "Number of atoms: " << n_atoms << std::endl;

  std::cout << n_atoms << std::endl;
}

void Grad_cv::prepare_FFTs(){
  /**
   * @brief Plan the FFTs that will be used in the future
   * 
   */

  std::string wisdom_file;

  wisdom_file = "data/FFTW_wisdom/wisdom" + std::to_string(n_pixels) + ".txt";

  //Check if wisdom file exists and import it if that's the case
  //The plans only depend on the number of pixels!
  if (std::filesystem::exists(wisdom_file)) myfftw_import_wisdom_from_filename(wisdom_file.c_str());
  

  //Create plans for the fftw
  release_FFT_plans();
  mycomplex_t *tmp_map, *tmp_map2;

  //temporal variables used to create the plans
  tmp_map = (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) *
                                          n_pixels *
                                          n_pixels);
  tmp_map2 = (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) *
                                           n_pixels *
                                           n_pixels);
  
  fft_plan_r2c_forward = myfftw_plan_dft_r2c_2d(
      n_pixels, n_pixels,
      (myfloat_t *) tmp_map, tmp_map2, FFTW_MEASURE | FFTW_DESTROY_INPUT);

  fft_plan_c2r_backward = myfftw_plan_dft_c2r_2d(
      n_pixels, n_pixels, tmp_map,
      (myfloat_t *) tmp_map2, FFTW_MEASURE | FFTW_DESTROY_INPUT);

  if (fft_plan_r2c_forward == 0 || fft_plan_c2r_backward == 0){
    myError("Planning FFTs");
  }

  myfftw_free(tmp_map);
  myfftw_free(tmp_map2);

  
  //If the wisdom file doesn't exists, then create a new one
  if (!std::filesystem::exists(wisdom_file)) myfftw_export_wisdom_to_filename(wisdom_file.c_str());

  fft_plans_created = 1;
}

void Grad_cv::quaternion_rotation(myvector_t &q, myvector_t &r_ref, myvector_t &r_rot){

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

    r_rot[i] = Q[0][0]*r_ref[i] + Q[0][1]*r_ref[i + n_atoms] + Q[0][2]*r_ref[i + 2*n_atoms];
    r_rot[i + n_atoms] = Q[1][0]*r_ref[i] + Q[1][1]*r_ref[i + n_atoms] + Q[1][2]*r_ref[i + 2*n_atoms];
    r_rot[i + 2*n_atoms] = Q[2][0]*r_ref[i] + Q[2][1]*r_ref[i + n_atoms] + Q[2][2]*r_ref[i + 2*n_atoms];
  }
}

void Grad_cv::quaternion_rotation(myvector_t &q, myvector_t &r_ref){

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
  for (unsigned int i=0; i<n_atoms; i++){

    x_tmp = Q[0][0]*r_ref[i] + Q[0][1]*r_ref[i + n_atoms] + Q[0][2]*r_ref[i + 2*n_atoms];
    y_tmp = Q[1][0]*r_ref[i] + Q[1][1]*r_ref[i + n_atoms] + Q[1][2]*r_ref[i + 2*n_atoms];
    z_tmp = Q[2][0]*r_ref[i] + Q[2][1]*r_ref[i + n_atoms] + Q[2][2]*r_ref[i + 2*n_atoms];

    r_ref[i] = x_tmp;
    r_ref[i + n_atoms] = y_tmp;
    r_ref[i + 2*n_atoms] = z_tmp;
  }
}

void Grad_cv::L2_grad(myvector_t &r_a, myvector_t &I_c, myvector_t &I_e,
                      myvector_t &gr_r, myfloat_t &s){

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
  #pragma omp parallel
  {
    int m_x, m_y;
    int ind_i, ind_j;

    // std::vector<size_t> x_sel, y_sel;
    myvector_t gauss_x(2*n_neigh+3, 0.0);
    myvector_t gauss_y(2*n_neigh+3, 0.0);

    #pragma omp for
    for (int atom=0; atom<n_atoms; atom++){

      m_x = (int) std::round(abs(r_a[atom] - grid_min)/pixel_size);
      m_y = (int) std::round(abs(r_a[atom + n_atoms] - grid_min)/pixel_size);

      #pragma omp simd
      for (int i=0; i<=2*n_neigh+2; i++){
        
        ind_i = m_x - n_neigh - 1 + i;
        ind_j = m_y - n_neigh - 1 + i;

        if (ind_i<0 || ind_i>=n_pixels) gauss_x[i] = 0;
        else {

          myfloat_t expon_x = (grid[ind_i] - r_a[atom])/sigma_cv;
          gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
        }
              
        if (ind_j<0 || ind_j>=n_pixels) gauss_y[i] = 0;
        else{

          myfloat_t expon_y = (grid[ind_j] - r_a[atom + n_atoms])/sigma_cv;
          gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
        }
      }

      myfloat_t s1=0, s2=0;

      //Calculate the image and the gradient
      for (int i=0; i<=2*n_neigh+2; i++){
        
        ind_i = m_x - n_neigh - 1 + i;
        if (ind_i<0 || ind_i>=n_pixels) continue;
        
        #pragma omp simd
        for (int j=0; j<=2*n_neigh+2; j++){

          ind_j = m_y - n_neigh - 1 + j;
          if (ind_j<0 || ind_j>=n_pixels) continue;

          s1 += (I_c[ind_i*n_pixels + ind_j] - I_e[ind_i*n_pixels + ind_j]) * (grid[ind_i] - r_a[atom]) * gauss_x[i] * gauss_y[j];
          s2 += (I_c[ind_i*n_pixels + ind_j] - I_e[ind_i*n_pixels + ind_j]) * (grid[ind_j] - r_a[atom + n_atoms]) * gauss_x[i] * gauss_y[j];
        }
      }

      gr_r[atom] = s1 * 2*norm / (sigma_cv * sigma_cv);
      gr_r[atom + n_atoms] = s2 * 2*norm / (sigma_cv * sigma_cv);
      gr_r[atom + 2*n_atoms] = 0.0;
    }  
  
    #pragma omp for simd reduction(+ : s)
    for (int i=0; i<n_pixels*n_pixels; i++) s += (I_c[i] - I_e[i]) * (I_c[i] - I_e[i]);
  }
}

void Grad_cv::calc_I(myvector_t &r_a, myvector_t &I_c){

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

  #pragma omp parallel
  {
    int m_x, m_y;
    int ind_i, ind_j;

    // std::vector<size_t> x_sel, y_sel;
    myvector_t gauss_x(2*n_neigh+3, 0.0);
    myvector_t gauss_y(2*n_neigh+3, 0.0);

    #pragma omp for reduction(vec_float_plus : I_c)
    for (int atom=0; atom<n_atoms; atom++){

      m_x = (int) std::round(abs(r_a[atom] - grid_min)/pixel_size);
      m_y = (int) std::round(abs(r_a[atom + n_atoms] - grid_min)/pixel_size);

      #pragma omp simd
      for (int i=0; i<=2*n_neigh+2; i++){
        
        ind_i = m_x - n_neigh - 1 + i;
        ind_j = m_y - n_neigh - 1 + i;

        if (ind_i<0 || ind_i>=n_pixels) gauss_x[i] = 0;
        else {

          myfloat_t expon_x = (grid[ind_i] - r_a[atom])/sigma_cv;
          gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
        }
              
        if (ind_j<0 || ind_j>=n_pixels) gauss_y[i] = 0;
        else{

          myfloat_t expon_y = (grid[ind_j] - r_a[atom + n_atoms])/sigma_cv;
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

          I_c[ind_i*n_pixels + ind_j] += gauss_x[i]*gauss_y[j];
        }
      }
    }

    #pragma omp for simd
    for (int i=0; i<n_pixels*n_pixels; i++) I_c[i] *= norm;
  }
}

void Grad_cv::calc_ctf(mycomplex_t* ctf){

  /**
   * @brief calculate the ctf for the convolution with the calculated image
   * 
   */

  myfloat_t radsq, val, normctf;

  for (int i=0; i<n_pixels_fft_1d; i++){ 
    for (int j=0; j<n_pixels_fft_1d; j++){

        radsq = (myfloat_t)(i * i + j * j) / n_pixels_fft_1d / n_pixels_fft_1d 
                                           / pixel_size / pixel_size;

        val = exp(-b_factor * radsq * 0.5) * 
              ( -CTF_amp * cos(phase * radsq * 0.5) 
              - sqrt(1 - CTF_amp*CTF_amp) * sin(phase * radsq * 0.5) );

        if (i==0 && j==0) normctf = (myfloat_t) val;

        ctf[i * n_pixels_fft_1d + j][0] = val/normctf;
        ctf[i * n_pixels_fft_1d + j][1] = 0;

        ctf[(n_pixels - i - 1) 
            * n_pixels_fft_1d + j][0] = val/normctf;

        ctf[(n_pixels - i - 1) 
            * n_pixels_fft_1d + j][1] = 0;      
    }
  }
}

void Grad_cv::conv_proj_ctf(){
  
  mycomplex_t *CTF = (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) * n_pixels * n_pixels_fft_1d);

  memset(CTF, 0, n_pixels * n_pixels_fft_1d * sizeof(mycomplex_t));

  for (int i = 0; i < n_pixels*n_pixels_fft_1d; i++){

      CTF[i][0] = 0.f;
      CTF[i][1] = 0.f;
  }

  calc_ctf(CTF);

  // std::ofstream ctf_file("data/output/ctf_file.txt");

  // for (int i=0; i<n_pixels*n_pixels_fft_1d; i++){

  //   ctf_file << CTF[i][0] << std::endl;
  // }
  // ctf_file.close();

   myfloat_t *localproj = (myfloat_t *) myfftw_malloc(sizeof(myfloat_t) * n_pixels * n_pixels);

  for (int i=0; i<n_pixels*n_pixels; i++) localproj[i] = Icalc[i];

  mycomplex_t *projFFT;
  myfloat_t *conv_proj_ctf;

  projFFT = (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) *
                                          n_pixels *
                                          n_pixels);

  conv_proj_ctf = (myfloat_t *) myfftw_malloc(sizeof(myfloat_t) *
                                           n_pixels *
                                           n_pixels);

  myfftw_execute_dft_r2c(fft_plan_r2c_forward, localproj, projFFT);

  
  for (int i=0; i<n_pixels * n_pixels_fft_1d; i++){

    projFFT[i][0] = projFFT[i][0]*CTF[i][0] + projFFT[i][1]*CTF[i][1];
    projFFT[i][1] = projFFT[i][0]*CTF[i][1] - projFFT[i][1]*CTF[i][0];
  }

  myfftw_execute_dft_c2r(fft_plan_c2r_backward, projFFT, conv_proj_ctf);

  int norm = n_pixels * n_pixels;

  for (int i=0; i<n_pixels*n_pixels; i++){ 

    Icalc[i] = conv_proj_ctf[i]/norm;
  }

  myfftw_free(localproj);
  myfftw_free(projFFT);
  myfftw_free(CTF);

}

void Grad_cv::I_with_noise(myvector_t &I, myfloat_t std=0.1){

  /**
   * @brief Blurs an image using gaussian noise
   * 
   * @param I image to which the noise will be applied
   * 
   */

  // Define random generator with Gaussian distribution
  const myfloat_t mean = 0.0;
  std::default_random_engine generator;
  std::normal_distribution<myfloat_t> dist(mean, std);

  // Add Gaussian noise
  for (int i=0; i<n_pixels*n_pixels; i++){

    I[i] += dist(generator);
  }
}

void Grad_cv::gaussian_normalization(){

  myfloat_t mu=0, var=0;

  myfloat_t rad_sq = pixel_size * (n_pixels + 1) * 0.5;
  rad_sq = rad_sq * rad_sq;

  std::vector <int> ins_circle;
  where(grid, grid, ins_circle, rad_sq);
  int N = ins_circle.size()/2; //Number of indexes

  myfloat_t curr_float;
  //Calculate the mean
  for (int i=0; i<N; i++) {

    curr_float = Icalc[ins_circle[i]*n_pixels + ins_circle[i+1]];
    mu += curr_float;
    var += curr_float * curr_float;
  }

  mu /= N;
  var /= N;

  var = std::sqrt(var - mu*mu);

  //Add gaussian noise with std equal to the intensity variance of the image times the SNR
  I_with_noise(Icalc, var * SNR);

  mu = 0; var = 0;
  for (int i=0; i<n_pixels*n_pixels; i++) {

    curr_float = Icalc[i];
    mu += curr_float;
    var += curr_float * curr_float;
  }

  mu /= n_pixels*n_pixels;
  var /= n_pixels*n_pixels;

  var = std::sqrt(var - mu*mu);

  for (int i=0; i<n_pixels*n_pixels; i++){

    Icalc[i] /= var;
  }
}

void Grad_cv::grad_run(){


  //Rotate the coordinates
  quaternion_rotation(quat, r_coord);
  
  std::cout << "\n Calculating CV and its gradient..." << std::endl;

  // Comment if using l2-norm
  // correlation(x_coord, y_coord, z_coord, 
  //             Icalc, grad_x, grad_y, s_cv);

  //Uncomment if you want to use l2_norm
  calc_I(r_coord, Icalc);
  L2_grad(r_coord, Icalc, Iexp, grad_r, s_cv);

  std::cout <<"\n ...done" << std::endl;
  
  //Rotating gradient
  //quaternion_rotation(quat_inv, grad_x, grad_y, grad_z);
  results_to_json(s_cv, grad_r);
}

void Grad_cv::gen_run(bool use_qt){

  Icalc = myvector_t(n_pixels*n_pixels, 0);
  
  //Rotate the coordinates
   if (use_qt) {
    
    std::cout << "Using random quaternions: " << use_qt << std::endl;
    quaternion_rotation(quat, r_coord);
  }

  else {quat = myvector_t(4, 0); quat_inv = myvector_t(4, 0);}

  std::cout << "\n Performing image projection ..." << std::endl;
  calc_I(r_coord, Icalc);
  std::cout << "... done" << std::endl;
    
  //The convoluted image was written in Iexp because we need two images to test the cv and the gradient
  // std::cout << "\n Applying CTF to calcualted image ..." << std::endl;
  // conv_proj_ctf();
  // std::cout << "... done" << std::endl;

  if (use_qt) gaussian_normalization();

  print_image(Icalc, image_file);
  std::cout << "\n The calculated image (with ctf) was saved in " << image_file << std::endl;
}

/* void Grad_cv::parallel_run(){
  
  std::cout << "Running gradient for " << n_imgs << " images..." << std::endl;
  
  #pragma omp parallel
  {
    // Gradient variables
    mymatrix_t r_cr_thr(n_atoms, myvector_t(3, 0.0));
    myvector_t I_c_thr(n_pixels*n_pixels, 0.0);
    mymatrix_t grad_thr(n_atoms, myvector_t(3, 0.0));
    myfloat_t s_thr;

    #pragma omp for
    for (int i=0; i<n_imgs; i++){

      quaternion_rotation(exp_imgs[i].q, r_coord, r_cr_thr);

      calc_I(r_cr_thr, I_c_thr);
      L2_grad(r_cr_thr, I_c_thr, exp_imgs[i].inten, grad_thr, s_thr);
      quaternion_rotation(exp_imgs[i].q_inv, grad_thr);

      I_c_thr = myvector_t(n_pixels*n_pixels, 0.0);

      #pragma omp critical
      {
        for (int j=0; j<n_atoms; j++){

          grad_r[j][0] += grad_thr[j][0];
          grad_r[j][1] += grad_thr[j][1];
          grad_r[j][2] += grad_thr[j][2];
        }

        s_cv += s_thr;
      }
    }
  }

  std::cout << "...done" << std::endl;
  results_to_json(s_cv, grad_r);
  std::cout << "Results saved." << std::endl;
} */

void Grad_cv::test_parallel_num(){
  
  //std::cout << "Running gradient for " << n_imgs << " images..." << std::endl;
  
  myvector_t r_rot = r_coord;
  myvector_t Icalc(n_pixels*n_pixels, 0.0);
  myvector_t grad_tmp(3*n_atoms);
  myfloat_t s_tmp;
  s_cv = 0;

  myfloat_t s1, s2, num_grad;


  for (int i=0; i<n_imgs; i++){

    quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

    calc_I(r_rot, Icalc);
    L2_grad(r_rot, Icalc, exp_imgs[i].inten, grad_tmp, s_tmp);
    quaternion_rotation(exp_imgs[i].q_inv, grad_tmp);

    Icalc = myvector_t(n_pixels*n_pixels, 0.0);

    #pragma omp parallel for simd 
    for (int j=0; j<3*n_atoms; j++) grad_r[j] += grad_tmp[j];

    s_cv += s_tmp;
  }

  s1 = s_cv; s_cv = 0;
  myfloat_t dt = 0.0001;
  int index = 13;

  grad_r = myvector_t(3*n_atoms, 0.0);
  r_coord[index] += dt;

  for (int i=0; i<n_imgs; i++){

    quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

    calc_I(r_rot, Icalc);
    L2_grad(r_rot, Icalc, exp_imgs[i].inten, grad_tmp, s_tmp);
    quaternion_rotation(exp_imgs[i].q_inv, grad_tmp);

    Icalc = myvector_t(n_pixels*n_pixels, 0.0);

    #pragma omp parallel for
    for (int j=0; j<3*n_atoms; j++) grad_r[j] += grad_tmp[j];

    s_cv += s_tmp;
  }

  s2 = s_cv;
  num_grad = (s2 - s1)/dt;

  std::cout.precision(10);                                                                                                                                              
  std::cout << s1 << ", " << s2 << std::endl;                                                                                                                           
  std::cout << std::scientific << "grad_A: " << grad_r[index] << "\n"                                                                                                   
            << "grad_N: " << num_grad << std::endl;                                                                                                                     



  // std::cout << "...done" << std::endl;
  // results_to_json(s_cv, grad_r);
  // std::cout << "Results saved." << std::endl;
}

void Grad_cv::test_parallel_time(){
  
  std::cout << "Running gradient for " << n_imgs << " images..." << std::endl;
  auto start1 = std::chrono::high_resolution_clock::now();

  // Gradient variables
  myvector_t r_rot = r_coord;
  myvector_t Icalc(n_pixels*n_pixels, 0.0);
  myvector_t grad_tmp(3*n_atoms, 0.0);
  myfloat_t s_tmp;
  s_cv = 0;

  for (int k=0; k<1000; k++){

    for (int i=0; i<n_imgs; i++){

      quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

      calc_I(r_rot, Icalc);
      L2_grad(r_rot, Icalc, exp_imgs[i].inten, grad_tmp, s_tmp);
      quaternion_rotation(exp_imgs[i].q_inv, grad_tmp);

      Icalc = myvector_t(n_pixels*n_pixels, 0.0);

      #pragma omp parallel for simd
      for (int j=0; j<3*n_atoms; j++) grad_r[j] += grad_tmp[j];
      s_cv += s_tmp;
    }

  grad_r = myvector_t(3*n_atoms, 0.0);
  }
  
  auto end1 = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  
  std::cout << "Parallel time: " << duration1.count()/1000 << " µs"<< std::endl;
}

void Grad_cv::test_serial_time(){
  
  std::cout << "Running gradient for " << n_imgs << " images..." << std::endl;
  auto start1 = std::chrono::high_resolution_clock::now();

  // Gradient variables
  myvector_t r_cr(3*n_atoms, 0.0);
  
  myvector_t grad_it(3*n_atoms, 0.0);
  myfloat_t s_it;
  myvector_t I_c(n_pixels*n_pixels, 0.0);

  for (int j=0; j<1000; j++){
  
    s_cv = 0;
    grad_r = myvector_t(3*n_atoms, 0.0);

    for (int i=0; i<n_imgs; i++){

      quaternion_rotation(exp_imgs[i].q, r_coord, r_cr);

      calc_I(r_cr, I_c);
      L2_grad(r_cr, I_c, exp_imgs[i].inten, grad_it, s_it);
      quaternion_rotation(exp_imgs[i].q_inv, grad_it);

      for (int j=0; j<3*n_atoms; j++) grad_r[j] += grad_it[j];

      s_cv += s_it;
      I_c = myvector_t(n_pixels*n_pixels, 0.0);
    }
  }
  
  auto end1 = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  
  std::cout << "Serial time: " << duration1.count()/1000 << " µs"<< std::endl;
}
// Utilities
void Grad_cv::arange(myvector_t &out_vec, myfloat_t xo, myfloat_t xf, myfloat_t dx){

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

void Grad_cv::print_image(myvector_t &Im, std::string fname){

  std::ofstream matrix_file;
  matrix_file.open (fname);

  std::cout.precision(3);

  matrix_file << std::scientific << std::showpos << defocus << " \n";

  for (int i=0; i<4; i++){

    matrix_file << std::scientific << std::showpos << quat[i] << " \n";
  }

  for (int i=0; i<n_pixels; i++){
    for (int j=0; j<n_pixels; j++){

      matrix_file << std::scientific << std::showpos << Im[i*n_pixels + j] << " " << " \n"[j==n_pixels-1];
    }
  }

  matrix_file.close();
}

void Grad_cv::read_exp_img(std::string fname, myimage_t *IMG){

  std::ifstream file;
  file.open(fname);

  if (!file.good()){
    myError("Opening file: %s", fname.c_str());
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

    file >> IMG->inten[i];
  }
  file.close();
}

int Grad_cv::read_parameters(std::string fileinput){ 

  std::ifstream input(fileinput);
  if (!input.good())
  {
    myError("Opening file: %s", fileinput.c_str());
  }

  char line[512] = {0};
  char saveline[512];

  std::cout << "\n +++++++++++++++++++++++++++++++++++++++++ \n";
  std::cout << "\n   READING EM2D PARAMETERS            \n\n";
  std::cout << " +++++++++++++++++++++++++++++++++++++++++ \n";

  while (input.getline(line, 512))
  {
    strcpy(saveline, line);
    char *token = strtok(line, " ");

    if (token == NULL || line[0] == '#' || strlen(token) == 0){
      // comment or blank line
    }

    else if (strcmp(token, "N_IMGS") == 0){
      token = strtok(NULL, " ");
      n_imgs = atoi(token);

      if (n_imgs <= 0) myError("Invalid number of images");
      std::cout << "Number of images " << n_imgs << "\n";

      yesNimgs = true;
    }

    else if (strcmp(token, "PIXEL_SIZE") == 0){
      token = strtok(NULL, " ");
      pixel_size = atof(token);
      
      if (pixel_size < 0) myError("Negative pixel size");
      std::cout << "Pixel Size " << pixel_size << "\n";

      yesPixSi = true;
    }

    else if (strcmp(token, "NUMBER_PIXELS") == 0){
      
      token = strtok(NULL, " ");
      n_pixels = int(atoi(token));

      if (n_pixels < 0){

        myError("Negative Number of Pixels");
      }

      std::cout << "Number of Pixels " << n_pixels << "\n";
      yesNumPix = true;
    }
    // CTF PARAMETERS
    else if (strcmp(token, "CTF_ENV") == 0)
    {
      token = strtok(NULL, " ");
      b_factor = atof(token);
      if (b_factor < 0)
      {
        myError("Negative B Env.");
      }
      std::cout << "B Env. " << b_factor << "\n";
      yesBFact = true;
    }

    else if (strcmp(token, "CTF_AMPLITUDE") == 0)
    {
      token = strtok(NULL, " ");
      CTF_amp = atof(token);
      if (CTF_amp < 0){
        myError("Negative amplitude");
      }

      std::cout << "CTF Amp. " << CTF_amp << "\n";
      yesAMP = true;
    }

    else if (strcmp(token, "CTF_DEFOCUS") == 0)
    {
      token = strtok(NULL, " ");
      min_defocus = atof(token);
      if (min_defocus < 0) myError("Negative min defocus");

      token = strtok(NULL, " ");
      max_defocus = atof(token);
      if (max_defocus < 0) myError("Negative max defocus");

      std::cout << "Defocus " << min_defocus << " " << max_defocus << "\n";
      yesDefocus = true;
    }

    else if (strcmp(token, "ELECTRON_WAVELENGTH") == 0)
    {
      token = strtok(NULL, " ");
      elecwavel = atof(token);
      if (elecwavel < 0.0150)
      {
        myError("Wrong electron wave length %lf. "
                "Has to be in Angstrom (A)",
                elecwavel);
      }
      std::cout << "Electron wave length in (A) is: " << elecwavel << "\n";
    }
    // CV PARAMETERS
    else if (strcmp(token, "SIGMA") == 0)
    {
      token = strtok(NULL, " ");
      sigma_cv = atof(token);
      if (sigma_cv < 0)
      {
        myError("Negative standard deviation for the gaussians");
      }
      std::cout << "Sigma " << sigma_cv << "\n";
      yesSigmaCV = true;
    }
    else if (strcmp(token, "SIGMA_REACH") == 0)
    {
      token = strtok(NULL, " ");
      sigma_reach = atof(token);
      if (sigma_reach < 0)
      {
        myError("Negative sigma reach");
      }
      std::cout << "Sigma Reach " << sigma_reach << "\n";
      yesSigmaReach = true;
    }
  }
  input.close();

  if (not(yesNimgs)){
    myError("Input missing: please provide N_IMGS");
  }
  if (not(yesPixSi)){
    myError("Input missing: please provide PIXEL_SIZE");
  }
  if (not(yesNumPix)){
    myError("Input missing: please provide n_pixels");
  }
  if (not(yesBFact)){
    myError("Input missing: please provide CTF_ENV");
  }
  if (not(yesAMP)){
    myError("Input missing: please provide CTF_AMPLITUD");
  }
  if (not(yesSigmaCV)){
    myError("Input missing: please provide SIGMA");
  }
  if (not(yesSigmaReach)){
    myError("Input missing: please provide SIGMA_REACH");
  }

  if (not(yesDefocus) && p_type == "D"){
    myError("Input missing: please provide CTF_DEFOCUS")
  }

  if (elecwavel == 0.019688)
    std::cout << "Using default electron wave length: 0.019688 (A) of 300kV "
            "microscope\n";

  return 0;
}

void Grad_cv::results_to_json(myfloat_t s, myvector_t &sgrad){

  std::ofstream gradfile;

  gradfile.open(json_file); 

  //begin json file
  gradfile << "[" << std::endl;
  gradfile << std::setw(4) << "{" << std::endl;

  //save colvar
  std::cout.precision(17);
  gradfile << std::setprecision(15) << std::setw(10) << "\"s\": "
           << s << "," << std::endl;

  //begin sgrad_x
  gradfile << std::setw(17) << "\"sgrad_x\":"
           << "[" << std::endl;

  for (int j=0; j<n_atoms; j++) {

    if (j == n_atoms - 1) {

      gradfile << std::setw(17) << std::left << " " << sgrad[j]
               << std::endl;
    }

    else {
      gradfile << std::setw(17) << std::left << " " << sgrad[j] << ","
               << std::endl;
    }
  }

  gradfile << std::setw(6) << " "
           << "]," << std::endl;
  //end sgrad_x

  //begin sgrad_y
  gradfile << std::setw(17) << "\"sgrad_y\":"
           << "[" << std::endl;

  for (int j=0; j<n_atoms; j++) {

    if (j == n_atoms - 1) {

      gradfile << std::setw(17) << std::left << " " << sgrad[j + n_atoms]
               << std::endl;
    }

    else {
      gradfile << std::setw(17) << std::left << " " << sgrad[j + n_atoms] << ","
               << std::endl;
    }
  }

  gradfile << std::setw(6) << " "
           << "]," << std::endl;

  //end sgrad_y

  //begin sgrad_z
  gradfile << std::setw(17) << "\"sgrad_z\":"
           << "[" << std::endl;

  for (int j=0; j<n_atoms; j++) {

    if (j == n_atoms - 1) {

      gradfile << std::setw(17) << std::left << " " << sgrad[j + 2*n_atoms]
               << std::endl;
    }

    else {
      gradfile << std::setw(17) << std::left << " " << sgrad[j + 2*n_atoms] << ","
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

void Grad_cv::release_FFT_plans(){

  if (fft_plans_created)
  {
    myfftw_destroy_plan(fft_plan_r2c_forward);
    myfftw_destroy_plan(fft_plan_c2r_backward);
    myfftw_cleanup();
  }
  fft_plans_created = 0;
}

void Grad_cv::where(myvector_t &x_vec, myvector_t &y_vec,
                    std::vector<int> &out_vec, myfloat_t radius){                       
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

    for (int i=0; i<n_pixels; i++){
      for (int j=0; j<n_pixels; j++){

        if ( x_vec[i]*x_vec[i] + y_vec[j]*y_vec[j] <= radius){

          out_vec.push_back(i);
          out_vec.push_back(j);
      }
    }
  }
}
