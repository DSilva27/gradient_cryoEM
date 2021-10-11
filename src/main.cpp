#include "gradcv.h"
#include <mpi.h>
#include <omp.h>
#include <typeinfo>
#include <thread>

void parse_args(int argc, char* argv[], int rank, int world_size, std::string &type,
                std::string &coord_file, std::string &img_prefix, int &ntomp, int &img_p);

int main(int argc, char* argv[]){

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string type;
  std::string coord_file, img_prefix;
  int ntomp, n_imgs;

  parse_args(argc, argv, rank, world_size, type, coord_file, img_prefix, ntomp, n_imgs);

  omp_set_num_threads(ntomp);

  Grad_cv test;
  std::cout << "type: " << type << std::endl;
  test.init_variables(coord_file, img_prefix, n_imgs, type, rank, world_size);
  test.gen_run();

  // Finalize the MPI environment.
  MPI_Finalize();
  
  return 0;
}

void parse_args(int argc, char* argv[], int rank, int world_size, std::string &type,
                std::string &coord_file, std::string &img_prefix, int &ntomp, int &n_imgs){

  bool yesMode = false;
  bool yesCoord = false;
  bool yesNtomp = false;
  bool yesNimgs = false;
  bool yesImgPfx = false;

  for (int i = 1; i < argc; i++){

    if ( strcmp(argv[i], "grad") == 0){
      type = "G";
      yesMode = true;
    }
    
    else if ( strcmp(argv[i], "gen") == 0){
      type = "D"; 
      yesMode = true;
    } 

    else if ( strcmp(argv[i], "-f") == 0){
      coord_file = argv[i+1]; 
      yesCoord = true; 
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

    else{
      myError("Unknown argumen %s", argv[i]);
    }
  }

  if (rank == 0){

    if (!yesMode) myError("You should specify either image generation (gen) or gradient calculation (grad)");
    if (!yesCoord) coord_file = "coord.txt"; //myError("Missing parameter -f. You should use the name of your coordinate file without path.");
    if (!yesNtomp) ntomp = 1;
    if (!yesNimgs) n_imgs = 1;//myError("You should specify the number of images -n_imgs");
    if (!yesImgPfx) img_prefix = "Icalc_"; //myError("You should specify the prefix of your image -img_pfx");

    int req_cores = world_size*ntomp;
    int avail_cores = std::thread::hardware_concurrency();
    if (req_cores > avail_cores) myError("%d Cores required, but only %d available\n", req_cores, avail_cores);

    if ( type =="G") printf("Gradient Calculation Mode\n");
    if ( type =="D") printf("Image Generation Mode\n");
    printf("Using %d OMP Threads and %d MPI Ranks\n", ntomp, world_size);
    printf("Image Prefix: %s\n", img_prefix.c_str());
    printf("Number of images: %d\n", n_imgs);  
  }
}
