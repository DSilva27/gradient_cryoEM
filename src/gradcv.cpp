#include "gradcv.h"

void run_emgrad(std::string coord_file, std::string param_file, std::string img_prefix, int n_imgs, int rank, int world_size, int ntomp){

  // Read parameters
  myparam_t emgrad_param;
  read_parameters(param_file, &emgrad_param, rank);
  emgrad_param.n_imgs = n_imgs;

  // Read coordinates
  myvector_t r_coord;
  coord_file = coord_file;

  int coord_size;
  if (rank==0){
    read_coord(coord_file, r_coord, rank);
    coord_size = r_coord.size();
  } 

  MPI_Bcast(&coord_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) r_coord.resize(coord_size);

  MPI_Bcast(&r_coord[0], r_coord.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  emgrad_param.n_atoms = r_coord.size()/3;

  // Create grid
  emgrad_param.gen_grid();
  emgrad_param.calc_neigh();
  emgrad_param.calc_norm();

  // Load experimental images
  mydataset_t exp_imgs;
  //MPI_Barrier(MPI_COMM_WORLD);
  load_dataset(img_prefix, emgrad_param.n_imgs, emgrad_param.n_pixels, exp_imgs, rank, world_size);

  // Prepare for cv/gradient calculation
  myvector_t r_rot = r_coord;
  myfloat_t cv, acc_cv, total_cv;
  myvector_t Icalc(emgrad_param.n_pixels * emgrad_param.n_pixels, 0.0);
  
  myvector_t grad_tmp(emgrad_param.n_atoms*3, 0.0);
  myvector_t grad_acc(emgrad_param.n_atoms*3, 0.0);

  acc_cv = 0.0;

  for (int i=0; i<exp_imgs.size(); i++){

    quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

    // Calculate Image
    calc_img_omp(r_rot, Icalc, &emgrad_param, ntomp);

    // Gradient
    L2_grad(r_rot, Icalc, exp_imgs[i].I, grad_tmp, cv, &emgrad_param);
    acc_cv += cv;
    
    quaternion_rotation(exp_imgs[i].q_inv, grad_tmp);

    for (int j=0; j<emgrad_param.n_atoms*3; j++){
      grad_acc[j] += grad_tmp[j];
    }

    Icalc = myvector_t(emgrad_param.n_pixels*emgrad_param.n_pixels, 0.0);
  }

  MPI_Reduce(&acc_cv, &total_cv, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &grad_acc[0], emgrad_param.n_atoms*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    std::ofstream outfile;

    outfile.open("COLVAR_ref", std::ios_base::app); // append instead of overwrite
    outfile << std::setprecision(15) << total_cv << std::endl; 
  }
}

void run_gen(std::string coord_file, std::string param_file, std::string img_prefix, int n_imgs, int rank, int world_size, int ntomp){

  // Read parameters
  myparam_t emgrad_param;

  emgrad_param.mode = "gen";
  read_parameters(param_file, &emgrad_param, rank);
  emgrad_param.n_imgs = n_imgs;

  // Read coordinates
  myvector_t r_coord;
  coord_file = coord_file;
  int coord_size;

  // Read coordinates on rank 0
  if (rank==0){
    read_coord(coord_file, r_coord, rank);
    coord_size = r_coord.size();
  } 

  // Broadcast number of atoms to each MPI process
  MPI_Bcast(&coord_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) r_coord.resize(coord_size);
  emgrad_param.n_atoms = r_coord.size()/3;

  // Broadcast coordinates
  MPI_Bcast(&r_coord[0], r_coord.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Create grid
  emgrad_param.gen_grid();
  emgrad_param.calc_neigh();
  emgrad_param.calc_norm();

  int imgs_per_process = emgrad_param.n_imgs/world_size;

  if (imgs_per_process < 1) myError("The number of images must be bigger than the number of processes!")
  
  int start_img = rank*imgs_per_process;
  int end_img = start_img + imgs_per_process;

  //Generate random defocus and calculate the phase
  std::random_device seeder;
  std::mt19937 engine(seeder());
  //std::uniform_real_distribution<myfloat_t> dist_def(min_defocus, max_defocus);

  // Random dist for quaternions
  // Create a uniform distribution from 0 to 1
  std::uniform_real_distribution<myfloat_t> dist_quat(0, 1);
  myfloat_t u1, u2, u3;

  mydataset_t exp_imgs(imgs_per_process);

  int counter = 0;
  for (int i=start_img; i<end_img; i++){

    //exp_imgs[counter].defocus = dist_def(engine);
    //phase = defocus * M_PI * 2. * 10000 * elecwavel;

    // Generate random numbers betwwen 0 and 1
    u1 = dist_quat(engine); u2 = dist_quat(engine); u3 = dist_quat(engine);

    // Fill img structure with its quaternions
    exp_imgs[counter].q[0] = std::sqrt(1 - u1) * sin(2 * M_PI * u2);
    exp_imgs[counter].q[1] = std::sqrt(1 - u1) * cos(2 * M_PI * u2);
    exp_imgs[counter].q[2] = std::sqrt(u1) * sin(2 * M_PI * u3);
    exp_imgs[counter].q[3] = std::sqrt(u1) * cos(2 * M_PI * u3);

    // Allocate memory for later
    exp_imgs[counter].I = myvector_t(emgrad_param.n_pixels*emgrad_param.n_pixels, 0.0);

    // Set image name
    exp_imgs[counter].fname = img_prefix + std::to_string(i) + ".txt";

    counter++;
  } 

  if (rank == 0) printf("\nGenerating images...\n");
  myvector_t r_rot = r_coord;

  for (int i=0; i<exp_imgs.size(); i++){

    quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

    // calculate image projection
    calc_img_omp(r_rot, exp_imgs[i].I, &emgrad_param, ntomp);
    gaussian_normalization(exp_imgs[i].I, &emgrad_param, 1.0);

    // print image
    print_image(&exp_imgs[i], emgrad_param.n_pixels);

  }
  if (rank == 0) printf("...done\n");
}

void run_num_test(std::string coord_file, std::string param_file, std::string img_prefix, int n_imgs, int rank, int world_size, int ntomp){

  // Read parameters
  myparam_t emgrad_param;

  emgrad_param.mode = "num_test";
  read_parameters(param_file, &emgrad_param, rank);
  emgrad_param.n_imgs = n_imgs;

  // Read coordinates
  myvector_t r_coord;
  coord_file = coord_file;

  int coord_size;
  if (rank==0){
    read_coord(coord_file, r_coord, rank);
    coord_size = r_coord.size();
  } 

  MPI_Bcast(&coord_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) r_coord.resize(coord_size);

  MPI_Bcast(&r_coord[0], r_coord.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  emgrad_param.n_atoms = r_coord.size()/3;

  // Create grid
  emgrad_param.gen_grid();
  emgrad_param.calc_neigh();
  emgrad_param.calc_norm();

  // Load experimental images
  mydataset_t exp_imgs;
  //MPI_Barrier(MPI_COMM_WORLD);
  load_dataset(img_prefix, emgrad_param.n_imgs, emgrad_param.n_pixels, exp_imgs, rank, world_size);


  myfloat_t acc_cv, total_cv1, total_cv2;

  acc_cv = 0.0;
  #pragma omp parallel num_threads(ntomp)
  {
    myvector_t r_rot = r_coord;
    myfloat_t cv;
    myvector_t Icalc(emgrad_param.n_pixels * emgrad_param.n_pixels, 0.0);
  
    myvector_t grad_tmp(emgrad_param.n_atoms*3, 0.0);

    #pragma omp for reduction(+ : acc_cv)
    for (int i=0; i<exp_imgs.size(); i++){

      quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

      // Calculate Image
      calc_img(r_rot, Icalc, &emgrad_param);

      cv = 0.0;
      for (size_t j=0; j<Icalc.size(); j++) cv += (Icalc[j] - exp_imgs[i].I[j])*(Icalc[j] - exp_imgs[i].I[j]);

      // Gradient
      //L2_grad(r_rot, Icalc, exp_imgs[i].I, grad_tmp, cv, &emgrad_param);
      acc_cv += cv;

      Icalc = myvector_t(emgrad_param.n_pixels*emgrad_param.n_pixels, 0.0);
    }
  }

  MPI_Reduce(&acc_cv, &total_cv1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // ******************************************* END OF FIRST STEP *************************************
  
  myfloat_t dt = 0.001;
  int index = 0 + emgrad_param.n_atoms;

  r_coord[index] += dt;
  acc_cv = 0.0;

  myvector_t grad_acc(emgrad_param.n_atoms*3, 0.0);

  #pragma omp parallel num_threads(ntomp)
  {
    myvector_t r_rot = r_coord;
    myfloat_t cv;
    myvector_t Icalc(emgrad_param.n_pixels * emgrad_param.n_pixels, 0.0);

    myvector_t grad_tmp(emgrad_param.n_atoms*3, 0.0);
    
    #pragma omp for reduction(vec_float_plus : grad_acc) reduction(+ : acc_cv)
    for (int i=0; i<exp_imgs.size(); i++){

      quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

      // Calculate Image
      calc_img(r_rot, Icalc, &emgrad_param);

      // Gradient
      L2_grad(r_rot, Icalc, exp_imgs[i].I, grad_tmp, cv, &emgrad_param);
      acc_cv += cv;
      
      quaternion_rotation(exp_imgs[i].q_inv, grad_tmp);

      for (size_t j=0; j<grad_acc.size(); j++){
        grad_acc[j] += grad_tmp[j];
      }

      Icalc = myvector_t(emgrad_param.n_pixels*emgrad_param.n_pixels, 0.0);
    }
  }

  MPI_Reduce(&acc_cv, &total_cv2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &grad_acc[0], emgrad_param.n_atoms*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  myfloat_t num_grad = (total_cv2 - total_cv1)/dt;

   if (rank == 0){

    printf("%s\n",std::string(35,'*').c_str());
    printf("* CV1: %f, CV2: %f\n", total_cv1, total_cv2);
    printf("*  Analytical gradient: %f  *\n", grad_acc[index]);
    printf("*  Numerical gradient: %f   *\n", num_grad);
    printf("%s\n",std::string(35,'*').c_str());
    
  }

  if (rank == 0){

    // printf("%s\n",std::string(35,'*').c_str());
    // printf("*  Analytical gradient: %f  *\n", grad_acc[index]);
    // printf("*  Numerical gradient: %f   *\n", num_grad);
    // printf("%s\n",std::string(35,'*').c_str());

   
    std::ofstream outfile;

    outfile.open("NUM_TESTS", std::ios_base::app); // append instead of overwrite
    outfile << std::setprecision(5) << " " << num_grad << " " << grad_acc[index];
    outfile.close();
  }
}

void run_num_test_omp(std::string coord_file, std::string param_file, std::string img_prefix, int n_imgs, int rank, int world_size, int ntomp){

  // Read parameters
  myparam_t emgrad_param;

  emgrad_param.mode = "num_test";
  read_parameters(param_file, &emgrad_param, rank);
  emgrad_param.n_imgs = n_imgs;

  // Read coordinates
  myvector_t r_coord;
  coord_file = coord_file;

  int coord_size;
  if (rank==0){
    read_coord(coord_file, r_coord, rank);
    coord_size = r_coord.size();
  } 

  MPI_Bcast(&coord_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) r_coord.resize(coord_size);

  MPI_Bcast(&r_coord[0], r_coord.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  emgrad_param.n_atoms = r_coord.size()/3;

  // Create grid
  emgrad_param.gen_grid();
  emgrad_param.calc_neigh();
  emgrad_param.calc_norm();

  // Load experimental images
  mydataset_t exp_imgs;
  //MPI_Barrier(MPI_COMM_WORLD);
  load_dataset(img_prefix, emgrad_param.n_imgs, emgrad_param.n_pixels, exp_imgs, rank, world_size);

  // Prepare for cv/gradient calculation
  myvector_t r_rot = r_coord;
  myfloat_t cv, acc_cv, total_cv1, total_cv2;
  myvector_t Icalc(emgrad_param.n_pixels * emgrad_param.n_pixels, 0.0);
  
  myvector_t grad_tmp(emgrad_param.n_atoms*3, 0.0);

  acc_cv = 0.0;

  for (int i=0; i<exp_imgs.size(); i++){

    quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

    // Calculate Image
    calc_img_omp(r_rot, Icalc, &emgrad_param, ntomp);

    // Gradient
    L2_grad_omp_N(r_rot, Icalc, exp_imgs[i].I, grad_tmp, cv, &emgrad_param, ntomp);
    acc_cv += cv;

    Icalc = myvector_t(emgrad_param.n_pixels*emgrad_param.n_pixels, 0.0);
  }

  MPI_Reduce(&acc_cv, &total_cv1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // ******************************************* END OF FIRST STEP *************************************

  
  myfloat_t dt = 0.001;
  int index = 0;

  r_coord[index] += dt;
  acc_cv = 0.0;
  myvector_t grad_acc(emgrad_param.n_atoms*3, 0.0);

  for (int i=0; i<exp_imgs.size(); i++){

    quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

    // Calculate Image
    calc_img_omp(r_rot, Icalc, &emgrad_param, ntomp);

    // Gradient
    L2_grad_omp_N(r_rot, Icalc, exp_imgs[i].I, grad_tmp, cv, &emgrad_param, ntomp);
    acc_cv += cv;
    
    quaternion_rotation(exp_imgs[i].q_inv, grad_tmp);

    for (int j=0; j<emgrad_param.n_atoms*3; j++){
      grad_acc[j] += grad_tmp[j];
    }

    Icalc = myvector_t(emgrad_param.n_pixels*emgrad_param.n_pixels, 0.0);
  }

  MPI_Reduce(&acc_cv, &total_cv2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &grad_acc[0], emgrad_param.n_atoms*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  myfloat_t num_grad = (total_cv2 - total_cv1)/dt;

  if (rank == 0){

    myfloat_t rel_err = std::abs(grad_acc[index] - num_grad)/std::abs(num_grad);
    printf("%s\n",std::string(35,'*').c_str());
    printf("CV1: %f, CV2: %f\n", total_cv1, total_cv2);
    printf("*  Analytical gradient: %f  *\n", grad_acc[index]);
    printf("*  Numerical gradient: %f   *\n", num_grad);
    printf("*  Relative error: %f       *\n", rel_err);
    printf("%s\n",std::string(35,'*').c_str());
  }

  // if (rank == 0){
   
  //   std::ofstream outfile;

  //   outfile.open("NUM_TESTS", std::ios_base::app); // append instead of overwrite
  //   outfile << std::setprecision(5) << " " << num_grad << " " << grad_acc[index] << std::endl; 
  //   outfile.close();
  // }
}

void run_time_test(std::string coord_file, std::string param_file, std::string img_prefix, int n_imgs, int rank, int world_size, int ntomp){

  // Read parameters
  myparam_t emgrad_param;

  emgrad_param.mode = "time_test";
  read_parameters(param_file, &emgrad_param, rank);
  emgrad_param.n_imgs = n_imgs;

  // Read coordinates
  myvector_t r_coord;
  coord_file = coord_file;

  int coord_size;
  if (rank==0){
    read_coord(coord_file, r_coord, rank);
    coord_size = r_coord.size();
  } 

  MPI_Bcast(&coord_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) r_coord.resize(coord_size);

  MPI_Bcast(&r_coord[0], r_coord.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  emgrad_param.n_atoms = r_coord.size()/3;

  // Create grid
  emgrad_param.gen_grid();
  emgrad_param.calc_neigh();
  emgrad_param.calc_norm();

  // Load experimental images
  mydataset_t exp_imgs;
  //MPI_Barrier(MPI_COMM_WORLD);
  load_dataset(img_prefix, emgrad_param.n_imgs, emgrad_param.n_pixels, exp_imgs, rank, world_size);

  myfloat_t t, dt;
  t = 0; dt = 0;

  int n_tries = 1000;
  for (int iTrial=0; iTrial < n_tries; iTrial++){

    MPI_Barrier(MPI_COMM_WORLD);
    const myfloat_t t0 = omp_get_wtime();

    myfloat_t acc_cv = 0.0;
    myfloat_t total_cv;

    myvector_t grad_acc(emgrad_param.n_atoms*3, 0.0);

    #pragma omp parallel num_threads(ntomp)
    {
      myvector_t r_rot = r_coord;
      myfloat_t cv;
      myvector_t Icalc(emgrad_param.n_pixels * emgrad_param.n_pixels, 0.0);

      myvector_t grad_tmp(emgrad_param.n_atoms*3, 0.0);
      
      #pragma omp for reduction(vec_float_plus : grad_acc) reduction(+ : acc_cv)
      for (int i=0; i<exp_imgs.size(); i++){

        quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

        // Calculate Image
        calc_img(r_rot, Icalc, &emgrad_param);

        // Gradient
        L2_grad(r_rot, Icalc, exp_imgs[i].I, grad_tmp, cv, &emgrad_param);
        acc_cv += cv;
        
        quaternion_rotation(exp_imgs[i].q_inv, grad_tmp);

        for (size_t j=0; j<grad_acc.size(); j++){
          grad_acc[j] += grad_tmp[j];
        }

        Icalc = myvector_t(emgrad_param.n_pixels*emgrad_param.n_pixels, 0.0);
      }
    }

    MPI_Reduce(&acc_cv, &total_cv, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &grad_acc[0], emgrad_param.n_atoms*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const myfloat_t t1 = omp_get_wtime();

    const myfloat_t ts   = t1-t0; // time in seconds
    const myfloat_t tms  = ts*1.0e3; // time in milliseconds

    t += tms;
    dt += tms*tms;
  }

  if (rank == 0){

    t /= n_tries;
    dt = std::sqrt(dt / n_tries - t*t);
    printf("*****************************************************\n");
    printf("\033[1m%s\033[0m\n%8s   \033[0m%8.1fms +- %.1fms \033[0m\n",
    "Average performance:", "", t, dt);
    printf("*****************************************************\n");
  }

  if (rank == 0){ 
    
    // t /= n_tries;
    // dt = std::sqrt(dt / n_tries - t*t);
   
    std::ofstream outfile;

    outfile.open("TIME_TESTS", std::ios_base::app); // append instead of overwrite
    outfile << std::setprecision(5) << " " << t << " " << dt; 
    outfile.close();
  }
}

void run_time_test_omp(std::string coord_file, std::string param_file, std::string img_prefix, int n_imgs, int rank, int world_size, int ntomp){

  // Read parameters
  myparam_t emgrad_param;

  emgrad_param.mode = "time_test";
  read_parameters(param_file, &emgrad_param, rank);
  emgrad_param.n_imgs = n_imgs;

  // Read coordinates
  myvector_t r_coord;
  coord_file = coord_file;

  int coord_size;
  if (rank==0){
    read_coord(coord_file, r_coord, rank);
    coord_size = r_coord.size();
  } 

  MPI_Bcast(&coord_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) r_coord.resize(coord_size);

  MPI_Bcast(&r_coord[0], r_coord.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  emgrad_param.n_atoms = r_coord.size()/3;

  // Create grid
  emgrad_param.gen_grid();
  emgrad_param.calc_neigh();
  emgrad_param.calc_norm();

  // Load experimental images
  mydataset_t exp_imgs;
  //MPI_Barrier(MPI_COMM_WORLD);
  load_dataset(img_prefix, emgrad_param.n_imgs, emgrad_param.n_pixels, exp_imgs, rank, world_size);

  myfloat_t t, dt;

  t = 0; dt = 0;
  int n_tries = 1000;
  
  for (int iTrial=0; iTrial < n_tries; iTrial++){

    MPI_Barrier(MPI_COMM_WORLD);
    const myfloat_t t0 = omp_get_wtime();

    myfloat_t acc_cv = 0.0;
    myfloat_t total_cv;

    myvector_t grad_acc(emgrad_param.n_atoms*3, 0.0);

    myvector_t r_rot = r_coord;
    myfloat_t cv;
    myvector_t Icalc(emgrad_param.n_pixels * emgrad_param.n_pixels, 0.0);

    myvector_t grad_tmp(emgrad_param.n_atoms*3, 0.0);
    
    for (int i=0; i<exp_imgs.size(); i++){

      quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

      // Calculate Image
      calc_img_omp(r_rot, Icalc, &emgrad_param, ntomp);

      // Gradient
      L2_grad_omp(r_rot, Icalc, exp_imgs[i].I, grad_tmp, cv, &emgrad_param, ntomp);
      acc_cv += cv;
      
      quaternion_rotation(exp_imgs[i].q_inv, grad_tmp);

      for (size_t j=0; j<grad_acc.size(); j++){
        grad_acc[j] += grad_tmp[j];
      }

      Icalc = myvector_t(emgrad_param.n_pixels*emgrad_param.n_pixels, 0.0);
    }

    MPI_Reduce(&acc_cv, &total_cv, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &grad_acc[0], emgrad_param.n_atoms*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const myfloat_t t1 = omp_get_wtime();

    const myfloat_t ts   = t1-t0; // time in seconds
    const myfloat_t tms  = ts*1.0e3; // time in milliseconds

    t += tms;
    dt += tms*tms;
  }

  // if (rank == 0){

  //   t /= n_tries;
  //   dt = std::sqrt(dt / n_tries - t*t);
  //   printf("*****************************************************\n");
  //   printf("\033[1m%s\033[0m\n%8s   \033[0m%8.1fms +- %.1fms \033[0m\n",
  //   "Average performance:", "", t, dt);
  //   printf("*****************************************************\n");
  // }

  if (rank == 0){ 
    
    t /= n_tries;
    dt = std::sqrt(dt / n_tries - t*t);
   
    std::ofstream outfile;

    outfile.open("TIME_TESTS", std::ios_base::app); // append instead of overwrite
    outfile << std::setprecision(5) << " " << t << " " << dt << std::endl; 
    outfile.close();
  }
}

void run_grad_descent(std::string coord_file, std::string param_file, std::string img_prefix, std::string out_prefix,
                      int n_imgs, int rank, int world_size, int ntomp, int n_steps, int stride, std::string d0){

  // Read parameters
  myparam_t emgrad_param;
  emgrad_param.mode = "grad_descent";
  read_parameters(param_file, &emgrad_param, rank);
  emgrad_param.n_imgs = n_imgs;

  // Read coordinates
  myvector_t r_coord;

  int coord_size;
  if (rank==0){
    read_coord(coord_file, r_coord, rank);
    coord_size = r_coord.size();
  } 

  MPI_Bcast(&coord_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) r_coord.resize(coord_size);

  MPI_Bcast(&r_coord[0], r_coord.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  emgrad_param.n_atoms = r_coord.size()/3;

  // Create grid
  emgrad_param.gen_grid();
  emgrad_param.calc_neigh();
  emgrad_param.calc_norm();

  // Load experimental images
  mydataset_t exp_imgs;
  //MPI_Barrier(MPI_COMM_WORLD);
  load_dataset(img_prefix, emgrad_param.n_imgs, emgrad_param.n_pixels, exp_imgs, rank, world_size);

  myvector_t r_rot = r_coord;
  myvector_t Icalc(emgrad_param.n_pixels * emgrad_param.n_pixels, 0.0);
  myvector_t grad_tmp(emgrad_param.n_atoms*3, 0.0);
  myvector_t grad_hm(emgrad_param.n_atoms*3, 0.0);

  myvector_t grad_l2(emgrad_param.n_atoms*3, 0.0);
  myfloat_t total_l2, acc_l2, l2, v_hm;
  myfloat_t old_l2 = 0.0;

  std::ofstream outfile;

  if (rank == 0){  
    outfile.open (out_prefix + "colvar.txt");
    outfile << "step    L2    HARM" << std::endl;
  }

  myfloat_t d0_flt;
  myvector_t d0_vec;

  std::string::const_iterator it = d0.begin();
  while (it != d0.end() && std::isdigit(*it)) ++it;
  bool d0IsNum = !d0.empty() && it == d0.end();

  if (d0IsNum){

    d0_flt = mystod(d0);
  }

  else {

    read_ref_d(d0, d0_vec);
  }

  myfloat_t diff;

  for (int step=0; step<n_steps; step++){    
    
    if (emgrad_param.l2_weight != 0){

      acc_l2 = 0;
      grad_l2 = myvector_t(emgrad_param.n_atoms*3, 0.0);

      for (int i=0; i<exp_imgs.size(); i++){

        quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);

        // Calculate Image
        calc_img_omp(r_rot, Icalc, &emgrad_param, ntomp);
        
        // Gradient
        L2_grad_omp_N(r_rot, Icalc, exp_imgs[i].I, grad_tmp, l2, &emgrad_param, ntomp);
        acc_l2 += l2;
        
        quaternion_rotation(exp_imgs[i].q_inv, grad_tmp);

        for (size_t j=0; j<grad_l2.size(); j++) grad_l2[j] += grad_tmp[j];
        
        Icalc = myvector_t(emgrad_param.n_pixels*emgrad_param.n_pixels, 0.0);
      }

      MPI_Reduce(&acc_l2, &total_l2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &grad_l2[0], emgrad_param.n_atoms*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (emgrad_param.hm_weight != 0){
      
      if (d0IsNum){

        harm_pot(r_coord, emgrad_param.hm_weight, d0_flt, v_hm, grad_hm, ntomp);
      }

      else {
        
        harm_pot(r_coord, emgrad_param.hm_weight, d0_vec, v_hm, grad_hm, ntomp);
      }
    }

    if (step%stride==0 && rank==0){

      outfile << step << "    " << total_l2 << "    " << v_hm << std::endl;
      std::cout << step << "    " << total_l2 << "    " << v_hm << std::endl;
    } 

    for (size_t j=0; j<grad_l2.size(); j++){
      
      r_coord[j] = r_coord[j] - emgrad_param.learn_rate * (emgrad_param.l2_weight * grad_l2[j] +\
                                                           emgrad_param.l2_weight * grad_hm[j]);
    } 

    diff = abs(total_l2 - old_l2);
    MPI_Bcast(&diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (diff <= emgrad_param.tol) {

      if (rank == 0){
        outfile << step << "    " << total_l2 << "    " << v_hm << std::endl; 
        std::cout << "Stopping simulation at step: " << step << std::endl;
      }

      
      break;
      //printf("Hi I am rank %d and I stopped\n", rank);
    }

    old_l2 = total_l2;
  }

  if (rank==0){
    
    outfile.close();
    //calc_img_omp(r_coord, Icalc, &emgrad_param, ntomp);
    //print_image(out_prefix + , Icalc, emgrad_param.n_pixels);
    print_coords(out_prefix + "coords.txt", r_coord, emgrad_param.n_atoms);
  }
}

void calc_img(myvector_t &r_a, myvector_t &I_c, myparam_t *PARAM){

  int m_x, m_y;
  int ind_i, ind_j;

  // std::vector<size_t> x_sel, y_sel;
  myvector_t gauss_x(2*PARAM->n_neigh+3, 0.0);
  myvector_t gauss_y(2*PARAM->n_neigh+3, 0.0);

  for (int atom=0; atom<PARAM->n_atoms; atom++){

    m_x = (int) std::round(abs(r_a[atom] - PARAM->grid[0])/PARAM->pixel_size);
    m_y = (int) std::round(abs(r_a[atom + PARAM->n_atoms] - PARAM->grid[0])/PARAM->pixel_size);

    for (int i=0; i<=2*PARAM->n_neigh+2; i++){
      
      ind_i = m_x - PARAM->n_neigh - 1 + i;
      ind_j = m_y - PARAM->n_neigh - 1 + i;

      if (ind_i<0 || ind_i>=PARAM->n_pixels) gauss_x[i] = 0;
      else {

        myfloat_t expon_x = (PARAM->grid[ind_i] - r_a[atom])/PARAM->sigma;
        gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
      }
            
      if (ind_j<0 || ind_j>=PARAM->n_pixels) gauss_y[i] = 0;
      else{

        myfloat_t expon_y = (PARAM->grid[ind_j] - r_a[atom + PARAM->n_atoms])/PARAM->sigma;
        gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
      }
    }

    //Calculate the image and the gradient
    for (int i=0; i<=2*PARAM->n_neigh+2; i++){
      
      ind_i = m_x - PARAM->n_neigh - 1 + i;
      if (ind_i<0 || ind_i>=PARAM->n_pixels) continue;
      
      for (int j=0; j<=2*PARAM->n_neigh+2; j++){

        ind_j = m_y - PARAM->n_neigh - 1 + j;
        if (ind_j<0 || ind_j>=PARAM->n_pixels) continue;

        I_c[ind_i*PARAM->n_pixels + ind_j] += gauss_x[i]*gauss_y[j];
      }
    }
  }

  for (int i=0; i<I_c.size(); i++) I_c[i] *= PARAM->norm; 
}

void calc_img_omp(myvector_t &r_a, myvector_t &I_c, myparam_t *PARAM, int ntomp){

  #pragma omp parallel num_threads(ntomp)
  {
    int m_x, m_y;
    int ind_i, ind_j;

    // std::vector<size_t> x_sel, y_sel;
    myvector_t gauss_x(2*PARAM->n_neigh+3, 0.0);
    myvector_t gauss_y(2*PARAM->n_neigh+3, 0.0);

    //myvector_t I_c_thr = I_c;

    #pragma omp for reduction(vec_float_plus : I_c)
    for (int atom=0; atom<PARAM->n_atoms; atom++){

      m_x = (int) std::round(abs(r_a[atom] - PARAM->grid[0])/PARAM->pixel_size);
      m_y = (int) std::round(abs(r_a[atom + PARAM->n_atoms] - PARAM->grid[0])/PARAM->pixel_size);

      for (int i=0; i<=2*PARAM->n_neigh+2; i++){
        
        ind_i = m_x - PARAM->n_neigh - 1 + i;
        ind_j = m_y - PARAM->n_neigh - 1 + i;

        if (ind_i<0 || ind_i>=PARAM->n_pixels) gauss_x[i] = 0;
        else {

          myfloat_t expon_x = (PARAM->grid[ind_i] - r_a[atom])/PARAM->sigma;
          gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
        }
              
        if (ind_j<0 || ind_j>=PARAM->n_pixels) gauss_y[i] = 0;
        else{

          myfloat_t expon_y = (PARAM->grid[ind_j] - r_a[atom + PARAM->n_atoms])/PARAM->sigma;
          gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
        }
      }

      //Calculate the image and the gradient
      for (int i=0; i<=2*PARAM->n_neigh+2; i++){
        
        ind_i = m_x - PARAM->n_neigh - 1 + i;
        if (ind_i<0 || ind_i>=PARAM->n_pixels) continue;
        
        for (int j=0; j<=2*PARAM->n_neigh+2; j++){

          ind_j = m_y - PARAM->n_neigh - 1 + j;
          if (ind_j<0 || ind_j>=PARAM->n_pixels) continue;

          I_c[ind_i*PARAM->n_pixels + ind_j] += gauss_x[i]*gauss_y[j];
        }
      }
    }

  }
  for (int i=0; i<I_c.size(); i++) I_c[i] *= PARAM->norm;
}

void L2_grad(myvector_t &r_a, myvector_t &I_c, myvector_t &I_e,
            myvector_t &gr_r, myfloat_t &cv, myparam_t *PARAM){

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

  myfloat_t norm = 2 * PARAM->norm / (PARAM->sigma * PARAM->sigma);

  // std::vector<size_t> x_sel, y_sel;
  myvector_t gauss_x(2*PARAM->n_neigh+3, 0.0);
  myvector_t gauss_y(2*PARAM->n_neigh+3, 0.0);

  for (int atom=0; atom<PARAM->n_atoms; atom++){

    m_x = (int) std::round(abs(r_a[atom] - PARAM->grid[0])/PARAM->pixel_size);
    m_y = (int) std::round(abs(r_a[atom + PARAM->n_atoms] - PARAM->grid[0])/PARAM->pixel_size);

    for (int i=0; i<=2*PARAM->n_neigh+2; i++){
      
      ind_i = m_x - PARAM->n_neigh - 1 + i;
      ind_j = m_y - PARAM->n_neigh - 1 + i;

      if (ind_i<0 || ind_i>=PARAM->n_pixels) gauss_x[i] = 0;
      else {

        myfloat_t expon_x = (PARAM->grid[ind_i] - r_a[atom])/PARAM->sigma;
        gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
      }
            
      if (ind_j<0 || ind_j>=PARAM->n_pixels) gauss_y[i] = 0;
      else{

        myfloat_t expon_y = (PARAM->grid[ind_j] - r_a[atom + PARAM->n_atoms])/PARAM->sigma;
        gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
      }
    }

    myfloat_t s1=0, s2=0;

    //Calculate the image and the gradient
    for (int i=0; i<=2*PARAM->n_neigh+2; i++){
      
      ind_i = m_x - PARAM->n_neigh - 1 + i;
      if (ind_i<0 || ind_i>=PARAM->n_pixels) continue;
      
      for (int j=0; j<=2*PARAM->n_neigh+2; j++){

        ind_j = m_y - PARAM->n_neigh - 1 + j;
        if (ind_j<0 || ind_j>=PARAM->n_pixels) continue;

        s1 += (I_c[ind_i*PARAM->n_pixels + ind_j] - I_e[ind_i*PARAM->n_pixels + ind_j]) 
              * (PARAM->grid[ind_i] - r_a[atom]) * gauss_x[i] * gauss_y[j];

        s2 += (I_c[ind_i*PARAM->n_pixels + ind_j] - I_e[ind_i*PARAM->n_pixels + ind_j]) 
              * (PARAM->grid[ind_j] - r_a[atom + PARAM->n_atoms]) * gauss_x[i] * gauss_y[j];
      }
    }

    gr_r[atom] = s1 * norm;
    gr_r[atom + PARAM->n_atoms] = s2 * norm;
    gr_r[atom + 2*PARAM->n_atoms] = 0;
  }

  cv = 0;
  for (int i=0; i<I_c.size(); i++){ 
    
    cv += (I_c[i] - I_e[i])*(I_c[i] - I_e[i]);
  }
}

void L2_grad_omp_N(myvector_t &r_a, myvector_t &I_c, myvector_t &I_e,
            myvector_t &gr_r, myfloat_t &cv, myparam_t *PARAM, int ntomp){

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

  cv = 0;

  myfloat_t Ccc = 0.0, Coc = 0.0;

  for (size_t i=0; i<I_c.size(); i++){

    Ccc += I_c[i] * I_c[i];
    Coc += I_e[i] * I_c[i];
  }

  myfloat_t N = Coc/Ccc;
  myfloat_t norm = 2*N * PARAM->norm / (PARAM->sigma * PARAM->sigma);

  #pragma omp parallel num_threads(ntomp)
  {

    int m_x, m_y;
    int ind_i, ind_j;

    // std::vector<size_t> x_sel, y_sel;
    myvector_t gauss_x(2*PARAM->n_neigh+3, 0.0);
    myvector_t gauss_y(2*PARAM->n_neigh+3, 0.0);

    #pragma omp for
    for (int atom=0; atom<PARAM->n_atoms; atom++){

      m_x = (int) std::round(abs(r_a[atom] - PARAM->grid[0])/PARAM->pixel_size);
      m_y = (int) std::round(abs(r_a[atom + PARAM->n_atoms] - PARAM->grid[0])/PARAM->pixel_size);

      for (int i=0; i<=2*PARAM->n_neigh+2; i++){
        
        ind_i = m_x - PARAM->n_neigh - 1 + i;
        ind_j = m_y - PARAM->n_neigh - 1 + i;

        if (ind_i<0 || ind_i>=PARAM->n_pixels) gauss_x[i] = 0;
        else {

          myfloat_t expon_x = (PARAM->grid[ind_i] - r_a[atom])/PARAM->sigma;
          gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
        }
              
        if (ind_j<0 || ind_j>=PARAM->n_pixels) gauss_y[i] = 0;
        else{

          myfloat_t expon_y = (PARAM->grid[ind_j] - r_a[atom + PARAM->n_atoms])/PARAM->sigma;
          gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
        }
      }

      myfloat_t s1=0, s2=0, s3=0, s4=0;

      //Calculate the image and the gradient
      for (int i=0; i<=2*PARAM->n_neigh+2; i++){
        
        ind_i = m_x - PARAM->n_neigh - 1 + i;
        if (ind_i<0 || ind_i>=PARAM->n_pixels) continue;
        
        for (int j=0; j<=2*PARAM->n_neigh+2; j++){

          ind_j = m_y - PARAM->n_neigh - 1 + j;
          if (ind_j<0 || ind_j>=PARAM->n_pixels) continue;

          s1 += (N*I_c[ind_i*PARAM->n_pixels + ind_j] - I_e[ind_i*PARAM->n_pixels + ind_j]) 
                * (PARAM->grid[ind_i] - r_a[atom]) * gauss_x[i] * gauss_y[j];

          s2 += (N*I_c[ind_i*PARAM->n_pixels + ind_j] - I_e[ind_i*PARAM->n_pixels + ind_j]) 
                * (PARAM->grid[ind_j] - r_a[atom + PARAM->n_atoms]) * gauss_x[i] * gauss_y[j];
        }
      }

      gr_r[atom] = s1 * norm;
      gr_r[atom + PARAM->n_atoms] = s2 * norm;
      gr_r[atom + 2*PARAM->n_atoms] = 0;
    }

    #pragma omp for reduction(+ : cv)
    for (int i=0; i<I_c.size(); i++){ 
      
      cv += (N*I_c[i] - I_e[i])*(N*I_c[i] - I_e[i]);
    }
  }
}

void L2_grad_omp(myvector_t &r_a, myvector_t &I_c, myvector_t &I_e,
            myvector_t &gr_r, myfloat_t &cv, myparam_t *PARAM, int ntomp){

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

  cv = 0;
  #pragma omp parallel num_threads(ntomp)
  {

    int m_x, m_y;
    int ind_i, ind_j;

    myfloat_t norm = 2*PARAM->norm / (PARAM->sigma * PARAM->sigma);

    // std::vector<size_t> x_sel, y_sel;
    myvector_t gauss_x(2*PARAM->n_neigh+3, 0.0);
    myvector_t gauss_y(2*PARAM->n_neigh+3, 0.0);

    #pragma omp for
    for (int atom=0; atom<PARAM->n_atoms; atom++){

      m_x = (int) std::round(abs(r_a[atom] - PARAM->grid[0])/PARAM->pixel_size);
      m_y = (int) std::round(abs(r_a[atom + PARAM->n_atoms] - PARAM->grid[0])/PARAM->pixel_size);

      for (int i=0; i<=2*PARAM->n_neigh+2; i++){
        
        ind_i = m_x - PARAM->n_neigh - 1 + i;
        ind_j = m_y - PARAM->n_neigh - 1 + i;

        if (ind_i<0 || ind_i>=PARAM->n_pixels) gauss_x[i] = 0;
        else {

          myfloat_t expon_x = (PARAM->grid[ind_i] - r_a[atom])/PARAM->sigma;
          gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
        }
              
        if (ind_j<0 || ind_j>=PARAM->n_pixels) gauss_y[i] = 0;
        else{

          myfloat_t expon_y = (PARAM->grid[ind_j] - r_a[atom + PARAM->n_atoms])/PARAM->sigma;
          gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
        }
      }

      myfloat_t s1=0, s2=0;

      //Calculate the image and the gradient
      for (int i=0; i<=2*PARAM->n_neigh+2; i++){
        
        ind_i = m_x - PARAM->n_neigh - 1 + i;
        if (ind_i<0 || ind_i>=PARAM->n_pixels) continue;
        
        for (int j=0; j<=2*PARAM->n_neigh+2; j++){

          ind_j = m_y - PARAM->n_neigh - 1 + j;
          if (ind_j<0 || ind_j>=PARAM->n_pixels) continue;

          s1 += (I_c[ind_i*PARAM->n_pixels + ind_j] - I_e[ind_i*PARAM->n_pixels + ind_j]) 
                * (PARAM->grid[ind_i] - r_a[atom]) * gauss_x[i] * gauss_y[j];

          s2 += (I_c[ind_i*PARAM->n_pixels + ind_j] - I_e[ind_i*PARAM->n_pixels + ind_j]) 
                * (PARAM->grid[ind_j] - r_a[atom + PARAM->n_atoms]) * gauss_x[i] * gauss_y[j];
        }
      }

      gr_r[atom] = s1 * norm;
      gr_r[atom + PARAM->n_atoms] = s2 * norm;
      gr_r[atom + 2*PARAM->n_atoms] = 0;
    }

    #pragma omp for reduction(+ : cv)
    for (int i=0; i<I_c.size(); i++){ 
      
      cv += (I_c[i] - I_e[i])*(I_c[i] - I_e[i]);
    }
  }
}

void harm_pot(myvector_t &r_a, myfloat_t k, myfloat_t d0, myfloat_t &vhm, myvector_t &grad, int ntomp){
    
  size_t n_atoms = r_a.size()/3.0;

  vhm = 0;
  #pragma omp parallel num_threads(ntomp)
  {
    myfloat_t d_p, d_m;
    
    #pragma omp for reduction(+ : vhm)
    for (size_t i=0; i<n_atoms; i++){

      if (i == 0){

        d_p = (r_a[i] - r_a[i+1])*(r_a[i] - r_a[i+1]) + \
              (r_a[i+n_atoms] - r_a[i+1+n_atoms])*(r_a[i+n_atoms] - r_a[i+1+n_atoms]) + \
              (r_a[i+2*n_atoms] - r_a[i+1+2*n_atoms])*(r_a[i+2*n_atoms] - r_a[i+1+2*n_atoms]);

        d_p = std::sqrt(d_p);

        grad[i] = k*(d0/d_p - 1) * (r_a[i+1] - r_a[i]);
        grad[i+n_atoms] = k*(d0/d_p - 1) * (r_a[i+1+n_atoms] - r_a[i+n_atoms]);
        grad[i+2*n_atoms] = k*(d0/d_p - 1) * (r_a[i+1+2*n_atoms] - r_a[i+2*n_atoms]);

        vhm += 0.5 * k * (d_p - d0)*(d_p - d0);
      }

      else if (i == n_atoms-1){

        d_m = (r_a[i] - r_a[i-1])*(r_a[i] - r_a[i-1]) + \
              (r_a[i+n_atoms] - r_a[i-1+n_atoms])*(r_a[i+n_atoms] - r_a[i-1+n_atoms]) + \
              (r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms])*(r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms]);

        d_m = std::sqrt(d_m);

        grad[i] = k*(d0/d_m - 1) * (r_a[i] - r_a[i-1]);
        grad[i+n_atoms] = k*(d0/d_m - 1) * (r_a[i+n_atoms] - r_a[i-1+n_atoms]);
        grad[i+2*n_atoms] = k*(d0/d_m - 1) * (r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms]);
      }

      else {

        d_p = (r_a[i] - r_a[i+1])*(r_a[i] - r_a[i+1]) + \
              (r_a[i+n_atoms] - r_a[i+1+n_atoms])*(r_a[i+n_atoms] - r_a[i+1+n_atoms]) + \
              (r_a[i+2*n_atoms] - r_a[i+1+2*n_atoms])*(r_a[i+2*n_atoms] - r_a[i+1+2*n_atoms]);

        d_p = std::sqrt(d_p);

        d_m = (r_a[i] - r_a[i-1])*(r_a[i] - r_a[i-1]) + \
              (r_a[i+n_atoms] - r_a[i-1+n_atoms])*(r_a[i+n_atoms] - r_a[i-1+n_atoms]) + \
              (r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms])*(r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms]);

        d_m = std::sqrt(d_m);

        grad[i] = k*((d0/d_p - 1) * (r_a[i+1] - r_a[i]) - \
                    (d0/d_m - 1) * (r_a[i] - r_a[i-1]));

        grad[i+n_atoms] = k*((d0/d_p - 1) * (r_a[i+1+n_atoms] - r_a[i+n_atoms]) - \
                            (d0/d_m - 1) * (r_a[i+n_atoms] - r_a[i-1+n_atoms]));

        grad[i+2*n_atoms] = k*((d0/d_p - 1) * (r_a[i+1+2*n_atoms] - r_a[i+2*n_atoms]) - \
                              (d0/d_m - 1) * (r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms]));
      
        vhm += 0.5 * k * (d_p - d0)*(d_p - d0);
      }

    }
  }
}

void harm_pot(myvector_t &r_a, myfloat_t k, myvector_t &d0, myfloat_t &vhm, myvector_t &grad, int ntomp){
    
  size_t n_atoms = r_a.size()/3.0;

  vhm = 0.0;
  #pragma omp parallel num_threads(ntomp)
  {
    myfloat_t d_p, d_m;
    
    #pragma omp for reduction(+ : vhm)
    for (size_t i=0; i<n_atoms; i++){

      if (i == 0){

        d_p = (r_a[i] - r_a[i+1])*(r_a[i] - r_a[i+1]) + \
              (r_a[i+n_atoms] - r_a[i+1+n_atoms])*(r_a[i+n_atoms] - r_a[i+1+n_atoms]) + \
              (r_a[i+2*n_atoms] - r_a[i+1+2*n_atoms])*(r_a[i+2*n_atoms] - r_a[i+1+2*n_atoms]);

        d_p = std::sqrt(d_p);

        grad[i] = k*(d0[i]/d_p - 1) * (r_a[i+1] - r_a[i]);
        grad[i+n_atoms] = k*(d0[i]/d_p - 1) * (r_a[i+1+n_atoms] - r_a[i+n_atoms]);
        grad[i+2*n_atoms] = k*(d0[i]/d_p - 1) * (r_a[i+1+2*n_atoms] - r_a[i+2*n_atoms]);

        vhm += 0.5 * k * (d_p - d0[i])*(d_p - d0[i]);
      }

      else if (i == n_atoms-1){

        d_m = (r_a[i] - r_a[i-1])*(r_a[i] - r_a[i-1]) + \
              (r_a[i+n_atoms] - r_a[i-1+n_atoms])*(r_a[i+n_atoms] - r_a[i-1+n_atoms]) + \
              (r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms])*(r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms]);

        d_m = std::sqrt(d_m);

        grad[i] = k*(d0[i]/d_m - 1) * (r_a[i] - r_a[i-1]);
        grad[i+n_atoms] = k*(d0[i]/d_m - 1) * (r_a[i+n_atoms] - r_a[i-1+n_atoms]);
        grad[i+2*n_atoms] = k*(d0[i]/d_m - 1) * (r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms]);
      }

      else {

        d_p = (r_a[i] - r_a[i+1])*(r_a[i] - r_a[i+1]) + \
              (r_a[i+n_atoms] - r_a[i+1+n_atoms])*(r_a[i+n_atoms] - r_a[i+1+n_atoms]) + \
              (r_a[i+2*n_atoms] - r_a[i+1+2*n_atoms])*(r_a[i+2*n_atoms] - r_a[i+1+2*n_atoms]);

        d_p = std::sqrt(d_p);

        d_m = (r_a[i] - r_a[i-1])*(r_a[i] - r_a[i-1]) + \
              (r_a[i+n_atoms] - r_a[i-1+n_atoms])*(r_a[i+n_atoms] - r_a[i-1+n_atoms]) + \
              (r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms])*(r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms]);

        d_m = std::sqrt(d_m);

        grad[i] = k*((d0[i]/d_p - 1) * (r_a[i+1] - r_a[i]) - \
                    (d0[i]/d_m - 1) * (r_a[i] - r_a[i-1]));

        grad[i+n_atoms] = k*((d0[i]/d_p - 1) * (r_a[i+1+n_atoms] - r_a[i+n_atoms]) - \
                            (d0[i]/d_m - 1) * (r_a[i+n_atoms] - r_a[i-1+n_atoms]));

        grad[i+2*n_atoms] = k*((d0[i]/d_p - 1) * (r_a[i+1+2*n_atoms] - r_a[i+2*n_atoms]) - \
                              (d0[i]/d_m - 1) * (r_a[i+2*n_atoms] - r_a[i-1+2*n_atoms]));
      
        vhm += 0.5 * k * (d_p - d0[i])*(d_p - d0[i]);
      }

    }
  }
}

void gaussian_normalization(myvector_t &I_c, param_device *PARAM, myfloat_t SNR=1.0){

  myfloat_t mu=0, var=0;

  myfloat_t rad_sq = PARAM->pixel_size * (PARAM->n_pixels + 1) * 0.5;
  rad_sq = rad_sq * rad_sq;

  std::vector <int> ins_circle;
  where(PARAM->grid, PARAM->grid, ins_circle, rad_sq);
  int N = ins_circle.size()/2; //Number of indexes

  myfloat_t curr_float;
  //Calculate the mean
  for (int i=0; i<N; i++) {

    curr_float = I_c[ins_circle[i]*PARAM->n_pixels + ins_circle[i+1]];
    mu += curr_float;
    var += curr_float * curr_float;
  }

  mu /= N;
  var /= N;

  var = std::sqrt(var - mu*mu);

  std::default_random_engine generator;
  std::normal_distribution<myfloat_t> dist(0.0, var * SNR);

  // Add Gaussian noise
  mu = 0; var = 0;
  for (size_t i=0; i<I_c.size(); i++){

    I_c[i] += dist(generator);
    curr_float = I_c[i];
    mu += curr_float;
    var += curr_float * curr_float;
  }

  mu /= I_c.size();
  var /= I_c.size();

  var = std::sqrt(var - mu*mu);

  for (size_t i=0; i<I_c.size(); i++){

    I_c[i] /= var;
  }
}

void quaternion_rotation(myvector_t &q, myvector_t &r_ref, myvector_t &r_rot){

/**
 * @brief Rotates a biomolecule using the quaternions rotation matrix
 *        according to (https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion)
 * 
 * @param q vector that stores the parameters for the rotation myvector_t (4)
 * @param r_ref original coordinates 
 * @param r_rot stores the rotated values 
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
  
  int n_atoms = r_ref.size()/3;

  for (int i=0; i<n_atoms; i++){

    r_rot[i] = Q[0][0]*r_ref[i] + Q[0][1]*r_ref[i + n_atoms] + Q[0][2]*r_ref[i + 2*n_atoms];
    r_rot[i + n_atoms] = Q[1][0]*r_ref[i] + Q[1][1]*r_ref[i + n_atoms] + Q[1][2]*r_ref[i + 2*n_atoms];
    r_rot[i + 2*n_atoms] = Q[2][0]*r_ref[i] + Q[2][1]*r_ref[i + n_atoms] + Q[2][2]*r_ref[i + 2*n_atoms];
  }
}

void quaternion_rotation(myvector_t &q, myvector_t &r_ref){

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

  int n_atoms = r_ref.size()/3;

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

void load_dataset(std::string img_prefix, int n_imgs, int n_pixels,
                  mydataset_t &dataset, int rank, int world_size){

  if (n_imgs < world_size) 
    myError("The number of images %d is too low for your ranks %d", n_imgs, world_size);

  int imgs_per_process = n_imgs / world_size;
  int start_img = rank*imgs_per_process;
  int end_img = start_img + imgs_per_process;

  dataset = mydataset_t(imgs_per_process);

  int counter = 0;
  for (int i=start_img; i < end_img; i++){

    dataset[counter].I = myvector_t(n_pixels*n_pixels);
    read_exp_img(img_prefix + std::to_string(i) + ".txt", &dataset[counter]);
    counter++;
  }
}

void read_exp_img(std::string fname, myimage_t *IMG){

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
  int n_pixels = IMG->I.size();
  for (int i=0; i<n_pixels; i++){

    file >> IMG->I[i];
  }
  file.close();
}

void read_parameters(std::string fname, myparam_t *PARAM, int rank){ 

  std::ifstream input(fname);
  if (!input.good())
  {
    myError("Opening file: %s", fname.c_str());
  }

  char line[512] = {0};
  char saveline[512];

  if (rank==0) std::cout << "\n +++++++++++++++++++++++++++++++++++++++++ \n";
  if (rank==0) std::cout << "\n   READING EM2D PARAMETERS            \n\n";
  if (rank==0) std::cout << " +++++++++++++++++++++++++++++++++++++++++ \n";

  bool yesPixSi = false, yesNumPix = false; 
  bool yesSigma = false, yesCutoff = false;
  bool yesLearnRate = false, yesL2Weight = false, yesHmWeight = false, yesTol = false;

  while (input.getline(line, 512)){

    strcpy(saveline, line);
    char *token = strtok(line, " ");

    if (token == NULL || line[0] == '#' || strlen(token) == 0){
      // comment or blank line
    }

    else if (strcmp(token, "PIXEL_SIZE") == 0){

      token = strtok(NULL, " ");
      PARAM->pixel_size = atof(token);
      
      if (PARAM->pixel_size < 0) myError("Negative pixel size");
      if (rank==0) std::cout << "Pixel Size " << PARAM->pixel_size << "\n";

      yesPixSi = true;
    }

    else if (strcmp(token, "NUMBER_PIXELS") == 0){
      
      token = strtok(NULL, " ");
      PARAM->n_pixels = int(atoi(token));

      if (PARAM->n_pixels < 0){

        myError("Negative Number of Pixels");
      }

      if (rank==0) std::cout << "Number of Pixels " << PARAM->n_pixels << "\n";
      yesNumPix = true;
    }

    // CV PARAMETERS
    else if (strcmp(token, "SIGMA") == 0){

      token = strtok(NULL, " ");
      PARAM->sigma = atof(token);

      if (PARAM->sigma < 0) myError("Negative standard deviation for the gaussians");
      
      if (rank==0) std::cout << "Sigma " << PARAM->sigma << "\n";
      yesSigma = true;
    }

    else if (strcmp(token, "CUTOFF") == 0){

      token = strtok(NULL, " ");
      PARAM->cutoff = atof(token);
      if (PARAM->cutoff < 0) myError("Negative cutoff");
      
      if (rank==0) std::cout << "Cutoff " << PARAM->cutoff << "\n";
      yesCutoff = true;
    }

    else if (strcmp(token, "LEARN_RATE") == 0){

      token = strtok(NULL, " ");
      PARAM->learn_rate = atof(token);
      if (PARAM->learn_rate < 0) myError("Negative Learning Rate");

      if (rank == 0) std::cout << "Learning rate " << PARAM->learn_rate << "\n";
      yesLearnRate = true;
    }

    else if (strcmp(token, "L2_WEIGHT") == 0){

      token = strtok(NULL, " ");
      PARAM->l2_weight = atof(token);
      if (PARAM->l2_weight < 0) myError("Negative Weight for L2-Norm");

      if (rank == 0) std::cout << "L2 Weight " << PARAM->l2_weight << "\n";
      yesL2Weight = true;
    }

    else if (strcmp(token, "HM_WEIGHT") == 0){

      token = strtok(NULL, " ");
      PARAM->hm_weight = atof(token);
      if (PARAM->hm_weight < 0) myError("Negative Weight for Harmonic Potential");

      if (rank == 0) std::cout << "Harmonic Potential Weight " << PARAM->hm_weight << "\n";
      yesHmWeight = true;
    }

    else if (strcmp(token, "TOLERANCE") == 0){

      token = strtok(NULL, " ");
      PARAM->tol = atof(token);

      if (rank == 0) std::cout << "TOLERANCE " << PARAM->tol << "\n";
      yesTol = true;
    }

    // else {
    //   myError("Unknown parameter %s", token);
    // }       
  }
  input.close();

  if (not(yesPixSi)){
    myError("Input missing: please provide PIXEL_SIZE");
  }
  if (not(yesNumPix)){
    myError("Input missing: please provide n_pixels");
  }
  if (not(yesSigma)){
    myError("Input missing: please provide SIGMA");
  }
  if (not(yesCutoff)){
    myError("Input missing: please provide CUTOFF");
  }

  if (PARAM->mode == "grad_descent"){
    
    if (not(yesLearnRate)){
      myError("Input for gradient descent missing: please provide LEARN_RATE")
    }

    if (not(yesL2Weight)){
      myError("Input for gradient descent missing: please provide L2_WEIGHT");
    }

    if (not(yesHmWeight)){
      myError("Input for gradient descent missing: please provide HM_WEIGHT")
    }

    if (not(yesTol)){
      myError("Input for gradient descent missing: please provide TOLERANCE")
    }

  }
}

void read_coord(std::string fname, myvector_t &r_coord, int rank){

    /**
   * @brief reads data from a pdb and stores the coordinates in vectors
   * 
   * @param fname name of the File with the coordinates
   * @param r_a stores the x coordinates of the atoms
   * 
   * @return void
   */

    // ! I'm not going to go into much detail here, because this will be replaced soon.
    // ! It's just for testing

    std::ifstream infile;
    infile.open(fname);

    if (!infile.good()) myError("Opening file: %s", fname.c_str());

    int N; //to store the number of atoms
    infile >> N;
    r_coord = myvector_t(N*3, 0.0);

    for (int i = 0; i < N * 3; i++){

        infile >> r_coord[i];
    }
    infile.close();

    //if (rank == 0) std::cout << "Number of atoms: " << N << std::endl;
    std::cout << "Number of atoms: " << N << std::endl;
}

void read_ref_d(std::string d0, myvector_t &d0_vec){

    std::ifstream infile;
    infile.open(d0);

    if (!infile.good()) myError("Opening file: %s", d0.c_str());

    size_t N; //to store the number of atoms
    infile >> N;
    d0_vec = myvector_t(N, 0.0);

    for (size_t i=0; i<N; i++){

        infile >> d0_vec[i];
    }
    infile.close();
}

void print_image(myimage_t *IMG, int n_pixels){

  std::ofstream matrix_file;
  matrix_file.open (IMG->fname);

  // std::cout.precision(3);

  matrix_file << std::scientific << std::showpos << IMG->defocus << " \n";

  for (int i=0; i<4; i++){

    matrix_file << std::scientific << std::showpos << IMG->q[i] << " \n";
  }

  for (int i=0; i<n_pixels; i++){
    for (int j=0; j<n_pixels; j++){

      matrix_file << std::scientific << std::showpos << IMG->I[i*n_pixels + j] << " " << " \n"[j==n_pixels-1];
    }
  }

  matrix_file.close();
}

void print_image(std::string fname, myvector_t &IMG, int n_pixels){

  std::ofstream matrix_file;
  matrix_file.open (fname);

  for (int i=0; i<n_pixels; i++){
    for (int j=0; j<n_pixels; j++){

      matrix_file << std::scientific << std::showpos << IMG[i*n_pixels + j] << " " << " \n"[j==n_pixels-1];
    }
  }

  matrix_file.close();
}

void print_coords(std::string fname, myvector_t &coords, int n_atoms){

  std::ofstream outfile;
  outfile.open(fname);

  // std::cout.precision(3);

  outfile << std::scientific << std::showpos << n_atoms << " \n";

  for (int i=0; i<3; i++){
    for (int j=0; j<n_atoms; j++){

      outfile << std::scientific << std::showpos << coords[i*n_atoms + j] << " " << " \n"[j==n_atoms-1];
    }
  }

  outfile.close();
}

void where(myvector_t &x_vec, myvector_t &y_vec,
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

    for (size_t i=0; i<x_vec.size(); i++){
      for (size_t j=0; j<x_vec.size(); j++){

        if ( x_vec[i]*x_vec[i] + y_vec[j]*y_vec[j] <= radius){

          out_vec.push_back(i);
          out_vec.push_back(j);
      }
    }
  }
}