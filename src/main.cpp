#include "gradcv.h"

int main(int argc, char* argv[]){

    std::string out_im_file;

    out_im_file = "Icalc_" + std::to_string(std::atoi(argv[1])) + ".txt";

    Grad_cv test("parameters.txt", "coord.txt", out_im_file);

    test.init_variables();
    test.run();
    
    return 0;
}