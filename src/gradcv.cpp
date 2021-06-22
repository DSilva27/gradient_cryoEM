#include "gradcv.h"


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
  number_pixels_fft_1d = number_pixels/2 + 1;
  n_atoms = x_coord.size();

  //If we are creating images
  if (p_type == "D"){

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
    quaternions = myvector_t(4, 0);

    //Random quaternion vector, check Shoemake, Graphic Gems III, p. 124-132
    quaternions[0] = std::sqrt(1 - u1) * sin(2 * M_PI * u2);
    quaternions[1] = std::sqrt(1 - u1) * cos(2 * M_PI * u2);
    quaternions[2] = std::sqrt(u1) * sin(2 * M_PI * u3);
    quaternions[3] = std::sqrt(u1) * cos(2 * M_PI * u3);
  }

  else if (p_type == "G"){

    Iexp = mymatrix_t(number_pixels, myvector_t(number_pixels, 0));
    quaternions = myvector_t(number_pixels, 0);

    
    //Turn grad_* into a number_pixels vector and fill it with zeros
    grad_x = (myfloat_t*) malloc(sizeof(myfloat_t) * n_atoms);
    grad_y = (myfloat_t*) malloc(sizeof(myfloat_t) * n_atoms);
    grad_z = (myfloat_t*) malloc(sizeof(myfloat_t) * n_atoms);

    for (int i=0; i<n_atoms; i++){

      grad_x[i] = 0;
      grad_y[i] = 0;
      grad_z[i] = 0;
    }
    std::cout << "Variables initialized" << std::endl;

    //#################### Read experimental image (includes defocus and quaternions) ###########################
    read_exp_img(image_file);
    phase = defocus * M_PI * 2. * 10000 * elecwavel;
  }

  //######################### Preparing FFTWs and allocating memory for images and gradients ##################
  //Prepare FFTs
  prepare_FFTs();
  
  //Turn Icalc into a number_pixels x number_pixels matrix and fill it with zeros
  Icalc = mymatrix_t(number_pixels, myvector_t(number_pixels, 0));
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

  int N; //to store the number of atoms
  coord_file >> N;

  myfloat_t a; //auxiliary variable to read from the file

  for (int col=0; col<N * 3; col++){

    coord_file >> a;
    
    if (col < N) x_coord.push_back(a);
    else if (col >= N && col < 2*N) y_coord.push_back(a);
    else if (col >= 2*N) z_coord.push_back(a);
  } 



std::cout << "Number of atoms: " << n_atoms << std::endl;
}

void Grad_cv::prepare_FFTs(){
  /**
   * @brief Plan the FFTs that will be used in the future
   * 
   */

  std::string wisdom_file;

  wisdom_file = "data/FFTW_wisdom/wisdom" + std::to_string(number_pixels) + ".txt";

  //Check if wisdom file exists and import it if that's the case
  //The plans only depend on the number of pixels!
  if (std::filesystem::exists(wisdom_file)) fftwf_import_wisdom_from_filename(wisdom_file.c_str());
  

  //Create plans for the fftw
  release_FFT_plans();
  mycomplex_t *tmp_map, *tmp_map2;

  //temporal variables used to create the plans
  tmp_map = (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) *
                                          number_pixels *
                                          number_pixels);
  tmp_map2 = (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) *
                                           number_pixels *
                                           number_pixels);
  
  fft_plan_r2c_forward = myfftw_plan_dft_r2c_2d(
      number_pixels, number_pixels,
      (myfloat_t *) tmp_map, tmp_map2, FFTW_MEASURE | FFTW_DESTROY_INPUT);

  fft_plan_c2r_backward = myfftw_plan_dft_c2r_2d(
      number_pixels, number_pixels, tmp_map,
      (myfloat_t *) tmp_map2, FFTW_MEASURE | FFTW_DESTROY_INPUT);

  if (fft_plan_r2c_forward == 0 || fft_plan_c2r_backward == 0){
    myError("Planning FFTs");
  }

  myfftw_free(tmp_map);
  myfftw_free(tmp_map2);

  
  //If the wisdom file doesn't exists, then create a new one
  if (!std::filesystem::exists(wisdom_file)) fftwf_export_wisdom_to_filename(wisdom_file.c_str());

  fft_plans_created = 1;
}

void Grad_cv::center_coord(myvector_t &x_a,  myvector_t &y_a,  myvector_t &z_a){

  /**
   * @brief Centers the coordinates of the biomolecule around its center of mass 
   * 
   * @param x_a stores the values of x
   * @param y_a stores the values of y
   * @param z_a stores the values of z
   * 
   * @return void
   */

  myfloat_t x_mean, y_mean, z_mean;
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

void Grad_cv::quaternion_rotation(myvector_t &q){

/**
 * @brief Rotates a biomolecule using the quaternions rotation matrix
 *        according to (https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion)
 * 
 * @param q vector that stores the parameters for the rotation myvector_t (4)
 * @param x_coord original coordinates x
 * @param y_coord original coordinates y
 * @param z_coord original coordinates z
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

  myvector_t x_r(n_atoms, 0); myvector_t z_r(n_atoms, 0); myvector_t y_r(n_atoms, 0);

  for (unsigned int i=0; i<n_atoms; i++){

    x_r[i] = x_coord[i]*Q[0][0] + y_coord[i]*Q[0][1] + z_coord[i]*Q[0][2];
    y_r[i] = x_coord[i]*Q[1][0] + y_coord[i]*Q[1][1] + z_coord[i]*Q[1][2];
    z_r[i] = x_coord[i]*Q[2][0] + y_coord[i]*Q[2][1] + z_coord[i]*Q[2][2];
  }

  x_coord = x_r;
  y_coord = y_r;
  z_coord = z_r;
}

void Grad_cv::calc_I_and_grad(){

  /**
   * @brief Calculates the image from the 3D model by representing the atoms as gaussians and using a 2d Grid
   * 
   * @param x_coord original coordinates x
   * @param y_coord original coordinates y
   * @param z_coord original coordinates z
   * @param sigma standard deviation for the gaussians, equal for all the atoms  (myfloat_t)
   * @paramsigma_reachnumber of sigmas used for cutoff (int)
   * @param number_pixels Resolution of the calculated image
   * @param Icalc Matrix used to store the calculated image
   * @param x values in x for the grid
   * @param y values in y for the grid
   * 
   * @return void
   * 
   */

  //Calculate minimum and maximum values for the linspace-like vectors x and y
  myfloat_t min = -pixel_size * (number_pixels + 1)*0.5;
  myfloat_t max = pixel_size * (number_pixels - 3)*0.5 + pixel_size;

  //Assign memory space required to fill the vectors
  x.resize(number_pixels); y.resize(number_pixels);

  //Generate them
  arange(x, min, max, pixel_size);
  arange(y, min, max, pixel_size);

  //Vectors used for masked selection of coordinates
  std::vector <size_t> x_sel;
  std::vector <size_t> y_sel;

  //Vectors to store the values of the gaussians
  myvector_t g_x(number_pixels, 0.0);
  myvector_t g_y(number_pixels, 0.0);

  for (int atom=0; atom<n_atoms; atom++){

    //calculates the indices that satisfy |x - x_atom| <= sigma_reach*sigma
    where(x, x_sel, x_coord[atom], sigma_reach * sigma_cv);
    where(y, y_sel, y_coord[atom], sigma_reach * sigma_cv);

    //calculate the gaussians
    for (int i=0; i<x_sel.size(); i++){

      g_x[x_sel[i]] = std::exp( -0.5 * (std::pow( (x[x_sel[i]] - x_coord[atom])/sigma_cv, 2 )) );
    }

    for (int i=0; i<y_sel.size(); i++){

      g_y[y_sel[i]] = std::exp( -0.5 * (std::pow( (y[y_sel[i]] - y_coord[atom])/sigma_cv, 2 )) );
    }

    myfloat_t s1=0, s2=0;
    //Calculate the image and the gradient
    for (int i=0; i<number_pixels; i++){ 
      for (int j=0; j<number_pixels; j++){ 
        
        Icalc[i][j] += g_x[i] * g_y[j];
        s1 += (x[i] - x_coord[atom]) * g_x[i] * g_y[j] * Iexp[i][j];
        s2 += (y[j] - y_coord[atom]) * g_x[i] * g_y[j] * Iexp[i][j];
      }
    }

    grad_x[atom] = -s1 * sqrt_2pi / sigma_cv;
    grad_y[atom] = -s2 * sqrt_2pi / sigma_cv;

    // *(grad_x + atom) = -s1 * sqrt_2pi / sigma_cv;
    // *(grad_y + atom) = -s2 * sqrt_2pi / sigma_cv;

    //Reset the vectors for the gaussians and selection
    x_sel.clear(); y_sel.clear();
    g_x = myvector_t(number_pixels, 0);
    g_y = myvector_t(number_pixels, 0);
  }

  for (int i=0; i<number_pixels; i++){ 
    for (int j=0; j<number_pixels; j++){ 
        
      Icalc[i][j] *= sqrt_2pi * sigma_cv;
    }
  }
}

void Grad_cv::calc_I(){

  /**
   * @brief Calculates the image from the 3D model by representing the atoms as gaussians and using a 2d Grid
   * 
   * @param x_coord original coordinates x
   * @param y_coord original coordinates y
   * @param z_coord original coordinates z
   * @param sigma standard deviation for the gaussians, equal for all the atoms  (myfloat_t)
   * @paramsigma_reachnumber of sigmas used for cutoff (int)
   * @param number_pixels Resolution of the calculated image
   * @param Icalc Matrix used to store the calculated image
   * @param x values in x for the grid
   * @param y values in y for the grid
   * 
   * @return void
   * 
   */

  //Calculate minimum and maximum values for the linspace-like vectors x and y
  myfloat_t min = -pixel_size * (number_pixels + 1)*0.5;
  myfloat_t max = pixel_size * (number_pixels - 3)*0.5 + pixel_size;

  //Assign memory space required to fill the vectors
  x.resize(number_pixels); y.resize(number_pixels);

  //Generate them
  arange(x, min, max, pixel_size);
  arange(y, min, max, pixel_size);

  //Vectors used for masked selection of coordinates
  std::vector <size_t> x_sel;
  std::vector <size_t> y_sel;

  //Vectors to store the values of the gaussians
  myvector_t g_x(number_pixels, 0.0);
  myvector_t g_y(number_pixels, 0.0);

  for (int atom=0; atom<x_coord.size(); atom++){

    //calculates the indices that satisfy |x - x_atom| <= sigma_reach*sigma
    where(x, x_sel, x_coord[atom], sigma_reach * sigma_cv);
    where(y, y_sel, y_coord[atom], sigma_reach * sigma_cv);

    //calculate the gaussians
    for (int i=0; i<x_sel.size(); i++){

      g_x[x_sel[i]] = std::exp( -0.5 * (std::pow( (x[x_sel[i]] - x_coord[atom])/sigma_cv, 2 )) );
    }

    for (int i=0; i<y_sel.size(); i++){

      g_y[y_sel[i]] = std::exp( -0.5 * (std::pow( (y[y_sel[i]] - y_coord[atom])/sigma_cv, 2 )) );
    }

    //Calculate the image
    for (int i=0; i<Icalc.size(); i++){ 
      for (int j=0; j<Icalc[0].size(); j++){ 
        
        Icalc[i][j] += g_x[i] * g_y[j];     
      }
    }

    //Reset the vectors for the gaussians and selection
    x_sel.clear(); y_sel.clear();

    g_x.clear(); g_y.clear();
    g_x.resize(number_pixels); g_y.resize(number_pixels);
    std::fill(g_x.begin(), g_x.end(), 0);
    std::fill(g_y.begin(), g_y.end(), 0);
  }

  for (int i=0; i<Icalc.size(); i++){ 
    for (int j=0; j<Icalc[0].size(); j++){ 
        
      Icalc[i][j] *= std::sqrt(2. * M_PI) * sigma_cv;     
    }
  }
}


void Grad_cv::calc_ctf(mycomplex_t* ctf){

  /**
   * @brief calculate the ctf for the convolution with the calculated image
   * 
   */

  myfloat_t radsq, val, normctf;

  for (int i=0; i<number_pixels_fft_1d; i++){ 
    for (int j=0; j<number_pixels_fft_1d; j++){

        radsq = (myfloat_t)(i * i + j * j) / number_pixels_fft_1d / number_pixels_fft_1d 
                                           / pixel_size / pixel_size;

        val = exp(-b_factor * radsq * 0.5) * 
              ( -CTF_amp * cos(phase * radsq * 0.5) 
              - sqrt(1 - CTF_amp*CTF_amp) * sin(phase * radsq * 0.5) );

        if (i==0 && j==0) normctf = (myfloat_t) val;

        ctf[i * number_pixels_fft_1d + j][0] = val/normctf;
        ctf[i * number_pixels_fft_1d + j][1] = 0;

        ctf[(number_pixels - i - 1) 
            * number_pixels_fft_1d + j][0] = val/normctf;

        ctf[(number_pixels - i - 1) 
            * number_pixels_fft_1d + j][1] = 0;      
    }
  }
}

void Grad_cv::conv_proj_ctf(){
  
  mycomplex_t *CTF = (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) * number_pixels * number_pixels_fft_1d);

  memset(CTF, 0, number_pixels * number_pixels_fft_1d * sizeof(mycomplex_t));

  for (int i = 0; i < number_pixels*number_pixels_fft_1d; i++){

      CTF[i][0] = 0.f;
      CTF[i][1] = 0.f;
  }

  calc_ctf(CTF);

  // std::ofstream ctf_file("data/output/ctf_file.txt");

  // for (int i=0; i<number_pixels*number_pixels_fft_1d; i++){

  //   ctf_file << CTF[i][0] << std::endl;
  // }
  // ctf_file.close();

   myfloat_t *localproj = (myfloat_t *) myfftw_malloc(sizeof(myfloat_t) * number_pixels * number_pixels);

  for (int i=0; i<number_pixels; i++){ 
    for (int j=0; j<number_pixels; j++){

      localproj[i * number_pixels + j] = Icalc[i][j];
    }
  }

  mycomplex_t *projFFT;
  myfloat_t *conv_proj_ctf;

  projFFT = (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) *
                                          number_pixels *
                                          number_pixels);

  conv_proj_ctf = (myfloat_t *) myfftw_malloc(sizeof(myfloat_t) *
                                           number_pixels *
                                           number_pixels);

  myfftw_execute_dft_r2c(fft_plan_r2c_forward, localproj, projFFT);

  
  for (int i=0; i<number_pixels * number_pixels_fft_1d; i++){

    projFFT[i][0] = projFFT[i][0]*CTF[i][0] + projFFT[i][1]*CTF[i][1];
    projFFT[i][1] = projFFT[i][0]*CTF[i][1] - projFFT[i][1]*CTF[i][0];
  }

  myfftw_execute_dft_c2r(fft_plan_c2r_backward, projFFT, conv_proj_ctf);

  int norm = number_pixels * number_pixels;

  for (int i=0; i<number_pixels; i++){ 
    for (int j=0; j<number_pixels; j++){

      Icalc[i][j] = conv_proj_ctf[i * number_pixels + j]/norm;
    }
  }

  myfftw_free(localproj);
  myfftw_free(projFFT);
  myfftw_free(CTF);

}

void Grad_cv::I_with_noise(mymatrix_t &I, myfloat_t std=0.1){

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
  for (int i=0; i<number_pixels; i++){
    for (int j=0; j<number_pixels; j++){

      I[i][j] += dist(generator);
    }
  }
}

void Grad_cv::gaussian_normalization(){

  myfloat_t mu=0, var=0;

  myfloat_t rad_sq = pixel_size * (number_pixels + 1) * 0.5;
  rad_sq = rad_sq * rad_sq;

  std::vector <int> ins_circle;
  where(x, y, ins_circle, rad_sq);
  int N = ins_circle.size()/2; //Number of indexes

  myfloat_t curr_float;
  //Calculate the mean
  for (int i=0; i<N; i++) {

    curr_float = Icalc[ins_circle[i]][ins_circle[i+1]];
    mu += curr_float;
    var += curr_float * curr_float;
  }

  mu /= N;
  var /= N;

  var = std::sqrt(var - mu*mu);

  //Add gaussian noise with std equal to the intensity variance of the image times the SNR
  I_with_noise(Icalc, var * SNR);

  mu = 0; var = 0;
  for (int i=0; i<number_pixels; i++) {
    for (int j=0; j<number_pixels; j++){

      curr_float = Icalc[i][j];
      mu += curr_float;
      var += curr_float * curr_float;
    }
  }

  mu /= number_pixels*number_pixels;
  var /= number_pixels*number_pixels;

  var = std::sqrt(var - mu*mu);

  for (int i=0; i<number_pixels; i++){
    for (int j=0; j<number_pixels; j++){

      Icalc[i][j] = Icalc[i][j] / var;
    }
  }
}

myfloat_t Grad_cv::collective_variable(){

  /**
   * @brief Calculates the collective variable (s). Which is the cross-correlation between the calculated image
   *        and an experimental or synthetic image.
   * 
   *        s = - sum_w sum_{x, y} Icalcc(x, y, phi_w) * Iw(x, y)
   * 
   *      ! phi_w is a rotation (to be developed)
   * 
   * @param Icalc calculated image (matrix of myfloat_ts)
   * @param Iexp experimental or synthetic image (matrix of myfloat_ts)
   * 
   * @return s the value of the collective variable
   */

  myfloat_t s=0; //to store the collective variable

  //TODO: improve the raising of this warning
  if (Icalc.size() != Iexp.size()){ 

    std::cout << "Intensity matrices must have the same dimension" << std::endl;
    return 0;
  }

  myfloat_t Icalc_sum = 0;
  myfloat_t Iexp_sum = 0;

  for (int i=0; i<number_pixels; i++){
    for (int j=0; j<number_pixels; j++){

      s += Icalc[i][j] * Iexp[i][j];
      Icalc_sum += Icalc[i][j] * Icalc[i][j];
      Iexp_sum += Iexp[i][j] * Iexp[i][j];
    }
  }

  //Normalize cv (still have to normalize the gradient!)
  return -s; // / (Icalc_sum * Iexp_sum);
}

void Grad_cv::gradient(myvector_t &r, myvector_t &r_a, myvector_t &sgrad, const char *R){

  /**
   * @brief Calculates the gradient of the colective variable (s) for image w along r_a 
   *        An example respect to coordinate x:
   *        
   *        ds_w/dx_i = - sum_{x, y} (x - x_i)/sigma_cv^2 * Icalc(x, y, phi_w)Iw(x, y)
   * 
   *        ! phi_w is a rotation (to be developed)
   * 
   * @param Icalc calculated image (matrix of myfloat_ts)
   * @param Iexp experimental or synthetic image (matrix of myfloat_ts)
   * @param r coordinates for the grid (e.g x or y)
   * @param r_a coordinates for the atoms (e.g x_a or y_a)
   * @param sigma standard deviation of the gaussians used for the atoms
   * 
   * @return void
   * 
   */

  // TODO I still have to normalize the gradient

  //Creates matrices of dimension number_pixels x number_pixels filled with zeros
  mymatrix_t Sxy(number_pixels, myvector_t (number_pixels, 0));
  mymatrix_t Iexp_T(number_pixels, myvector_t (number_pixels, 0));

  transpose(Iexp, Iexp_T);
  
  if (R = "x") matmul(Icalc, Iexp_T, Sxy);
  if (R = "y") matmul(Iexp_T, Icalc, Sxy);

  for (int i=0; i<n_atoms; i++){
    for (int j=0; j<r.size(); j++){

      sgrad[i] += (r_a[i] - r[j]) * Sxy[j][j];
    }
    sgrad[i] /= std::pow(sigma_cv, 2);
  }
}

void Grad_cv::grad_run(){


  //Rotate the coordinates
  //quaternion_rotation(quaternions);

  std::cout << "\n Performing image projection ..." << std::endl;
  calc_I_and_grad();
  std::cout << "... done" << std::endl;

    
  //The convoluted image was written in Iexp because we need two images to test the cv and the gradient
  /* std::cout << "\n Applying CTF to calcualted image ..." << std::endl;
  conv_proj_ctf();
  std::cout << "... done" << std::endl;
 */

  //Add gaussian noise with std equal to the intensity variance of the image times the SNR
  //I_with_noise(Icalc, Icalc_var * SNR);  
  
  std::cout << "\n Calculating CV and its gradient..." << std::endl;

  s_cv = collective_variable();
  //gradient(x, x_coord, grad_x, "x");
  //gradient(y, y_coord, grad_y, "y");
  //grad_z is already initialized with zeros

  results_to_json(s_cv, grad_x, grad_y, grad_z);
  std::cout <<"\n ...done" << std::endl;
}

void Grad_cv::gen_run(){


  //Rotate the coordinates
  //quaternion_rotation(quaternions);

  std::cout << "\n Performing image projection ..." << std::endl;
  calc_I();
  std::cout << "... done" << std::endl;

    
  //The convoluted image was written in Iexp because we need two images to test the cv and the gradient
  // std::cout << "\n Applying CTF to calcualted image ..." << std::endl;
  // conv_proj_ctf();
  // std::cout << "... done" << std::endl;

  //gaussian_normalization();

  print_image(Icalc, image_file);
  std::cout << "\n The calculated image (with ctf) was saved in " << image_file << std::endl;
}

// Utilities
void Grad_cv::where(myvector_t &inp_vec, std::vector<size_t> &out_vec, 
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

    for (int i=0; i<number_pixels; i++){
      for (int j=0; j<number_pixels; j++){

        if ( x_vec[i]*x_vec[i] + y_vec[j]*y_vec[j] <= radius){

          out_vec.push_back(i);
          out_vec.push_back(j);
      }
    }
  }
}

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

void Grad_cv::transpose(mymatrix_t &A, mymatrix_t &A_T){

  /**
   * @brief Computes the transpose of a matrix
   * 
   * @param A: original matrix (mymatrix_t)
   * @param A_T: matrix where the transpose of A will be stored (mymatrix_t)
   * 
   * @return void
   */

  for (int i=0; i<A.size(); i++){
    for (int j=0; j<A[0].size(); j++){

      A_T[i][j] = A[j][i]; //turns Aij to Aji
    }
  }
}

void Grad_cv::matmul(mymatrix_t &A, mymatrix_t &B, mymatrix_t &C){

  /**
   * @brief Performs matrix multiplication of 2D matrices. The second
   *        one is always transposed. So A * B_T = C
   * 
   * @param A matrix to be multiplied by the left (mymatrix_t)
   * @param B matrix to be multiplied by the right (mymatrix_t)
   * @param C matrix used to store the multiplication (mymatrix_t)
   * 
   * @return void
   */

  for (int i=0; i<C.size(); i++){
    for (int j=0; j<C[0].size(); j++){

      for (int k=0; k<A.size(); k++){

          C[i][j] += A[i][k] * B[j][k]; //Typical matrix multiplication cij = sum_k aik bkj
      }
    }
  }
}

void Grad_cv::print_image(mymatrix_t &Im, std::string fname){

  std::ofstream matrix_file;
  matrix_file.open (fname);

  std::cout.precision(3);

  matrix_file << std::scientific << std::showpos << defocus << " \n";

  for (int i=0; i<4; i++){

    matrix_file << std::scientific << std::showpos << quaternions[i] << " \n";
  }

  for (int i=0; i<number_pixels; i++){
    for (int j=0; j<number_pixels; j++){

      matrix_file << std::scientific << std::showpos << Im[i][j] << " " << " \n"[j==number_pixels-1];
    }
  }

  matrix_file.close();
}

void Grad_cv::read_exp_img(std::string fname){

  std::ifstream file;
  file.open(fname);

  if (!file.good()){
    myError("Opening file: %s", fname.c_str());
  }

  //Read defocus
  file >> defocus;
  
  //Read quaternions
  for (int i=0; i<4; i++) file >> quaternions[i];

  //Read image
  for (int i=0; i<number_pixels; i++){
    for (int j=0; j<number_pixels; j++){

      file >> Iexp[i][j];
    }
  }
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

    else if (strcmp(token, "PIXEL_SIZE") == 0){
      token = strtok(NULL, " ");
      pixel_size = atof(token);
      
      if (pixel_size < 0) myError("Negative pixel size");
      std::cout << "Pixel Size " << pixel_size << "\n";

      yesPixSi = true;
    }

    else if (strcmp(token, "NUMBER_PIXELS") == 0){
      
      token = strtok(NULL, " ");
      number_pixels = int(atoi(token));

      if (number_pixels < 0){

        myError("Negative Number of Pixels");
      }

      std::cout << "Number of Pixels " << number_pixels << "\n";
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

  if (not(yesPixSi)){
    myError("Input missing: please provide PIXEL_SIZE");
  }
  if (not(yesNumPix)){
    myError("Input missing: please provide NUMBER_PIXELS");
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


/* void Grad_cv::load_parameters() { 

  char letter, eq;
  myfloat_t value;

  std::string input_path = "data/input/";
  std::ifstream file;
  file.open(input_path + params_file);


  if (file.fail()){
    std::cout << "Warning: file " + params_file + " doesn't exist" << std::endl;

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
      number_pixels = (int)value;
      break;

    case 'P':
      pixel_size = (float)value;
      break;

    default:

      std::cout << "InputError: undefined parameter in parameters.txt" << std::endl;
      break;
    }
  }

  file.close();
} */

void Grad_cv::load_quaternions(myvector_t &q){
  
  char letter, eq;
  myfloat_t value;

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

void Grad_cv::results_to_json(myfloat_t s, myfloat_t* sgrad_x, myfloat_t* sgrad_y, myfloat_t* sgrad_z) {

  std::ofstream gradfile;

  gradfile.open(json_file);

  //begin json file
  gradfile << "[" << std::endl;
  gradfile << std::setw(4) << "{" << std::endl;

  //save colvar
  gradfile << std::setw(10) << "\"s\": "
           << s << "," << std::endl;

  //begin sgrad_x
  gradfile << std::setw(17) << "\"sgrad_x\":"
           << "[" << std::endl;

  for (int j=0; j<n_atoms; j++) {

    if (j == n_atoms - 1) {

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

  for (int j=0; j<n_atoms; j++) {

    if (j == n_atoms - 1) {

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

  for (int j=0; j<n_atoms; j++) {

    if (j == n_atoms - 1) {

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

void Grad_cv::release_FFT_plans(){

  if (fft_plans_created)
  {
    myfftw_destroy_plan(fft_plan_r2c_forward);
    myfftw_destroy_plan(fft_plan_c2r_backward);
    myfftw_cleanup();
  }
  fft_plans_created = 0;
}

