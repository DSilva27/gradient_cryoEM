#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

typedef std::vector <double> vec;
typedef std::vector <vec> mat;
template <typename T> std::string type_name();

void linspace(vec &out_vec, double, double, int);
void apply_mask(vec &inp_vec, vec &out_vec, double, double);

int main(){

    int res = 10;
    double xo = 0.5;
    double xf = 1.4;

/*     vec x(res);
    vec x_sel;
    
    linspace(x, xo, xf, res);
    apply_mask(x, x_sel, 0.5, 0.5); 

    for (int i=0; i<x_sel.size(); i++){

        std::cout << x_sel[i] << std::endl;
    }*/

    vec x(res, 0);
    std::cout << "before clear: " << x.size() << std::endl;

    x.clear();
    std::cout << "after clear: " << x.size() << std::endl;
}

void apply_mask(vec &inp_vec, vec &out_vec, double x_res, double limit){  

    std::vector<size_t> index;

    auto it = std::find_if(std::begin(inp_vec), std::end(inp_vec), [&](double i){return std::abs(i - x_res) <= limit;});
    
    while (it != std::end(inp_vec)) {
        index.emplace_back(std::distance(std::begin(inp_vec), it));
        it = std::find_if(std::next(it), std::end(inp_vec), [&](double i){return std::abs(i - x_res) <= limit;});
    }

    for (int i=0; i<index.size(); i++){
        out_vec.push_back(inp_vec[index[i]]);
    }
}

void linspace(vec &out_vec, double xo, double xf, int n){

    double a_x = (xf - xo)/(n - 1.0);
    std::generate(out_vec.begin(), out_vec.end(), [n=0, &xo, &a_x]() mutable { return n++ * a_x + xo; });   
}