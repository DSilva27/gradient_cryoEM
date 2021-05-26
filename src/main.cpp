#include "gradcv.h"

int main(){

    Grad_cv test("parameters.txt", "coord.txt");

    test.init_variables();
    test.run();
    
    return 0;
}