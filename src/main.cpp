#include "gradcv.h"
#include <mpi.h>
#include <omp.h>
#include <typeinfo>
#include <thread>

void parse_args(int argc, char* argv[], int rank, int world_size, std::string &type,
                std::string &coord_file, std::string &img_prefix, int &ntomp, int &img_p);

int main(int argc, char *argv[]){

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string mode;
  std::string coord_file, img_prefix, param_file, out_prefix;
  int ntomp, n_imgs, gd_steps, gd_stride;

  parse_args(argc, argv, rank, world_size, mode, coord_file, param_file,
            img_prefix, out_prefix, ntomp, n_imgs, gd_steps, gd_stride);

  omp_set_num_threads(ntomp);

  if (mode == "grad"){

    run_emgrad(coord_file, img_prefix, n_imgs, rank, world_size, ntomp);
  }
  
  else if (mode == "gen"){

    run_gen(coord_file, img_prefix, n_imgs, rank, world_size, ntomp);
  }

  else if (mode == "num_test"){
    
    if (rank == 0){

      std::ofstream outfile;

      outfile.open("NUM_TESTS", std::ios_base::app); // append instead of overwrite
      outfile << world_size << " " << ntomp; 
      outfile.close();  
    }

    //run_num_test(coord_file, img_prefix, n_imgs, rank, world_size, ntomp);
    //MPI_Barrier(MPI_COMM_WORLD);
    run_num_test_omp(coord_file, img_prefix, n_imgs, rank, world_size, ntomp);
  }

  else if (mode == "time_test"){

    if (rank == 0){

      std::ofstream outfile;

      outfile.open("TIME_TESTS", std::ios_base::app); // append instead of overwrite
      outfile << world_size << " " << ntomp; 
      outfile.close();  
    }

    run_time_test(coord_file, img_prefix, n_imgs, rank, world_size, ntomp);
    run_time_test_omp(coord_file, img_prefix, n_imgs, rank, world_size, ntomp);
  }

  else if (mode == "grad_descent"){

    run_grad_descent(coord_file, param_file, img_prefix, out_prefix, n_imgs, 
                     rank, world_size, ntomp, gd_steps, gd_stride);
  }
  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}

void parse_args(int argc, char* argv[], int rank, int world_size, std::string &type,
                std::string &coord_file, std::string &param_file, std::string &img_prefix, 
                std::string &out_prefix, int &ntomp, int &n_imgs, int &nsteps, int &stride){

  bool yesMode = false;
  bool yesCoord = false;
  bool yesParams = false;
  bool yesNtomp = false;
  bool yesNimgs = false;
  bool yesImgPfx = false;
  bool yesGdSteps = false;
  bool yesGdStride = false;
  bool yesOutPfx = false;

  for (int i = 1; i < argc; i++){

    if ( strcmp(argv[i], "grad") == 0){
      type = argv[i];
      printf("Gradient Calculation Mode\n");

      yesMode = true;
    }
    
    else if ( strcmp(argv[i], "gen") == 0){
      type = argv[i]; 
      printf("Image Generation Mode\n");

      yesMode = true;
    } 

    else if ( strcmp(argv[i], "num_test") == 0){
      type = argv[i]; 
      printf("Numerical test\n");

      yesMode = true;
    } 

    else if ( strcmp(argv[i], "time_test") == 0){
      type = argv[i]; 
      printf("Testing run-time\n");

      yesMode = true;
    } 

    else if ( strcmp(argv[i], "grad_descent") == 0){
      type = argv[i]; 
      printf("Performing gradient descent\n");

      yesMode = true;
    } 

    else if ( strcmp(argv[i], "-f") == 0){
      coord_file = argv[i+1]; 
      yesCoord = true; 
      i++; continue;
    }

    else if ( strcmp(argv[i], "-p") == 0){
      param_file = argv[i+1]; 
      yesParams = true; 
      i++; continue;
    }

    else if ( strcmp(argv[i], "-out_pfx") == 0){
      out_prefix = argv[i+1]; 
      yesOutPfx = true; 
      i++; continue;
    }

    else if ( strcmp(argv[i], "-ntomp") == 0){
      ntomp = std::atoi(argv[i+1]); 
      yesNtomp = true;
      i++; continue;
    } 

    else if ( strcmp(argv[i], "-n_imgs") == 0){
      n_imgs = std::atoi(argv[i+1]);
      yesNimgs = true;
      i++; continue;
    }

    else if ( strcmp(argv[i], "-img_pfx") == 0){
      img_prefix = argv[i+1];
      yesImgPfx = true;
      i++; continue;
    }

    else if ( strcmp(argv[i], "-nsteps") == 0){
      nsteps = std::atoi(argv[i+1]);
      yesGdSteps = true;
      i++; continue;
    }

    else if ( strcmp(argv[i], "-stride") == 0){
      stride = std::atoi(argv[i+1]);
      yesGdStride = true;
      i++; continue;
    }

    else{
      myError("Unknown argumen %s", argv[i]);
    }
  }

  if (rank == 0){

    if (!yesMode) myError("You should specify either image generation (gen) or gradient calculation (grad)");
    if (!yesCoord) coord_file = "input/coord.txt"; //myError("Missing parameter -f. You should use the name of your coordinate file without path.");
    if (!yesNtomp) ntomp = 1;
    if (!yesNimgs) n_imgs = 1;//myError("You should specify the number of images -n_imgs");
    if (!yesImgPfx) img_prefix = "images/Icalc_"; //myError("You should specify the prefix of your image -img_pfx");
    if (!yesParams) param_file = "parameters.txt";

    if (type == "grad_descent"){
      
      if (!yesGdSteps) myError("You should specify the number of steps for gradient descent simulation");
      if (!yesGdStride) stride = -1;
      if (!yesOutPfx) myError("You should provice the prefix for the outputs of your simulation!");
    }

    else {

      nsteps = -1;
      stride = -1;
    }

    int req_cores = world_size*ntomp;
    int avail_cores = std::thread::hardware_concurrency();
    if (req_cores > avail_cores) myError("%d Cores required, but only %d available\n", req_cores, avail_cores);

    printf("Using %d OMP Threads and %d MPI Ranks\n", ntomp, world_size);
    printf("Image Prefix: %s\n", img_prefix.c_str());
    printf("Number of images: %d\n", n_imgs);  
    if (type == "grad_descent") printf("Performing %d gradient descent steps\n", nsteps);
  }
}


