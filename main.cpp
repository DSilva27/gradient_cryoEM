#include <iostream>
#include <string> 
#include <fstream>
#include <vector>
#include <cmath>

#include <stdio.h>
#include <stdlib.h>

typedef std::vector <double> vec;
typedef std::vector <vec> mat;

void read_data(vec &x_c, vec &y_c, vec &z_c);
void center_coord(vec &x_c, vec &y_c, vec &z_c);
void quaternion_rotation(vec &q, vec &x_c, vec &y_c, vec &z_c, vec &x_r, vec &y_r, vec &z_r);
void I_calculated(vec &q, vec &x_c, vec &y_c, vec &z_c, vec &x, vec &y, double sigma, int n, float res);

int main(){
  
  vec x_coord; vec y_coord; vec z_coord;

  read_data(x_coord, y_coord, z_coord);
  center_coord(x_coord, y_coord, z_coord);
  
  return 0;
}

void I_calculated(vec &q, vec &x_c, vec &y_c, double sigma, int n, float res, mat &Ixy, vec &x, vec &y){

  double xmin = std::min_element(x_c.begin(), x_c.end());
  double xmax = std::max_element(x_c.begin(), x_c.end());
    
  double ymin = std::min_element(y_c.begin(), y_c.end());
  double ymax = std::max_element(y_c.begin(), y_c.end());

  double a_x = (xmax - xmin)/(res-1.0);
  double a_y = (ymax - ymin)/(res-1.0);

  x.resize(res); y.resize(res);

  std::generate(x.begin(), x.end(), [n=0, &a_x]() mutable { return n++ * a_x; });
  std::generate(y.begin(), y.end(), [n=0, &a_y]() mutable { return n++ * a_y; });

  Ixy.resize(res, res);

  for (int i=0; i<res; i++){ for (int j=0; j<res; j++){ Ixy[i][j] = 0; }}

  for (int atom=0; atom<x_c.size(); atom++){

    std::vector <double> g_x(res, 0.0);
    std::vector <double> g_y(res, 0.0);     
  }

  
}

void quaternion_rotation(vec &q, vec &x_c, vec &y_c, vec &z_c, vec &x_r, vec &y_r, vec &z_r){

  double q00 = 1 - 2*std::pow(q[2],2) - 2*std::pow(q[3],2);
  double q01 = 2*q[1]*q[2] - 2*q[3]*q[0];
  double q02 = 2*q[1]*q[3] + 2*q[2]*q[0];
  vec q0{ q00, q01, q02 };
  
  double q10 = 2*q[1]*q[2] + 2*q[3]*q[0];
  double q11 = 1 - 2*std::pow(q[1],2) - 2*std::pow(q[3],2);
  double q12 = 2*q[2]*q[3] - 2*q[1]*q[0];
  vec q1{ q10, q11, q12 };

  double q20 = 2*q[1]*q[3] - 2*q[2]*q[0];
  double q21 = 2*q[2]*q[3] - 2*q[1]*q[0];
  double q22 = 1 - 2*std::pow(q[1],2) - 2*std::pow(q[2],2);
  vec q2{ q20, q21, q22}

  std::vector <vec> Q{ q0, q1, q2 };
  
  int n = x_c.size();
  
  for (unsigned int i=0; i<n; i++){

    x_r[i] = x_c[i]*Q[0][0] + y_c[i]*Q[1][0] + z_c[i]*Q[2][0];
    y_r[i] = x_c[i]*Q[0][1] + y_c[i]*Q[1][1] + z_c[i]*Q[2][1];
    z_r[i] = x_c[i]*Q[0][2] + y_c[i]*Q[1][2] + z_c[i]*Q[2][2];

  }
}

void center_coord(vec &x_c, vec &y_c, vec &z_c){

  double x_mean, y_mean, z_mean;
  int n = x_c.size();

  for (unsigned int i=0; i<n; i++){

    x_mean += x_c[i];
    y_mean += y_c[i];
    z_mean += z_c[i];
  }

  x_mean /= n;
  y_mean /= n;
  z_mean /= n;

  for (unsigned int i=0; i<n; i++){

    x_c[i] -= x_mean;
    y_c[i] -= y_mean;
    z_c[i] -= z_mean;
  }
}

void read_data(vec &x_c, vec &y_c, vec &z_c){

  std::system("awk '($1==\"ATOM\") {print $7 \"\t\" $8 \"\t\" $9}' 1xck.pdb > tmp.txt");
  std::system("wc -l tmp.txt > n_atoms.txt");

  
  float x, y, z;

    std::ifstream file;

  int M;
  
  file.open("n_atoms.txt");
  file >> M;
  file.close();

  file.open("tmp.txt");

  for (unsigned int i=0; i<M; i++){

    file >> x >> y >> z;
    
    x_c.push_back(x);
    y_c.push_back(y);
    z_c.push_back(z);
  }

  file.close();

  std::system("rm tmp.txt && rm n_atoms.txt");
}
