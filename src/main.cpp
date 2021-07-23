#include "gradcv.h"
#include <typeinfo>

int main(int argc, char* argv[]){

  std::string param_file, coord_file;
  std::string out_im_file, out_json_file;
  const char *type;

  param_file = "parameters.txt";
  coord_file = "coord.txt";

  std::cout << argc << std::endl;
  out_im_file = "Icalc_" + std::to_string(std::atoi(argv[1])) + ".txt";
  out_json_file = "grad_" + std::to_string(std::atoi(argv[1])) + ".json";

  if ( strcmp(argv[2], "-grad") == 0) type = "G";
  else if ( strcmp(argv[2], "-gen") == 0) type = "D";
  
  else {
    std::cout << "Missing argument.\n"
              << "Read the README for more information" << std::endl;
  }

  Grad_cv test;
  test.init_variables(param_file, coord_file, out_im_file, out_json_file, type);

  if ( strcmp(argv[2], "-grad") == 0) test.grad_run();
  else if ( strcmp(argv[2], "-gen") == 0){
    
    if ( strcmp(argv[3], "-qt") == 0){

      test.gen_run(true);
    }

    else test.gen_run(false);
  } 
  
  return 0;
}