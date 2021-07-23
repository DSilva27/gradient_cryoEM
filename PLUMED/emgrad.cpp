#include "/Data/Packages/Research/plumed2/src/isdb/colvar/Colvar.h"
#include "/Data/Packages/Research/plumed2/src/isdb/colvar/ActionRegister.h"
#include "/Data/Packages/Research/plumed2/src/isdb/core/PlumedMain.h"
#include "/Data/Packages/Research/plumed2/src/isdb/tools/Pbc.h"

#include <string>
#include <cmath>
#include <algorithm>

typedef double myfloat_t;
typedef myfloat_t mycomplex_t[2];
typedef std::vector <myfloat_t> myvector_t;
typedef std::vector <myvector_t> mymatrix_t;


namespace PLMD {
namespace isdb {

class EmGrad : public Colvar {

 private:
  bool pbc;
  myfloat_t sigma, cutoff, pixel_size;
  int n_pixels;
  
  void read_coord(myvector_t &, myvector_t &, myvector_t &, myfloat_t &);
  void arange(myvector_t &, myfloat_t, myfloat_t, myfloat_t);
  void where(myvector_t &, std::vector<size_t> &, myfloat_t, myfloat_t);

 public:
  static void registerKeywords( Keywords& keys );
  explicit EmGrad(const ActionOptions&);
// active methods:
  void calculate() override;
};

PLUMED_REGISTER_ACTION(EmGrad,"EMGRAD")

void EmGrad::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords( keys );
  keys.add("atoms","ATOMS","The atoms for which we will calculate the neighbors");
  keys.add("compulsory","SIGMA","Standard deviation of the gaussians.");
  keys.add("compulsory","CUTOFF","Neighbor cutoff.");
  keys.add("compulsory","N_PIXELS","Number of pixels for projection.");
  keys.add("compulsory","PIXEL_SIZE","Size of each pixel.");
}

EmGrad::EmGrad(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true)
{
  std::vector<AtomNumber> atoms;
  parseAtomList("ATOMS",atoms);
  if(atoms.size()<1)
    error("You should define at least one atom");
  
  parse("SIGMA",sigma);
  parse("CUTOFF",cutoff);
  parse("N_PIXELS",n_pixels);
  parse("PIXEL_SIZE",pixel_size);

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;
  checkRead();

  log.printf("  atoms involved : ");
  for(unsigned i=0; i<atoms.size(); ++i) log.printf("%d ",atoms[i].serial());
  log.printf("\n");
  log.printf("  standard deviation for gaussians : %1f\n", sigma);
  log.printf("  neighbor list cutoff : %lf\n", cutoff);
  
  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");

  // log << " Bibliography" << plumed.cite("Bonomi, Camilloni, Bioinformatics, 33, 3999 (2017)") << "\n";

  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms(atoms);
}

void EmGrad::arange(myvector_t &out_vec, myfloat_t xo, myfloat_t xf, myfloat_t dx){

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

void EmGrad::where(myvector_t &inp_vec, std::vector<size_t> &out_vec, 
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


// calculator
void EmGrad::calculate() {

  if(pbc) makeWhole();

  
  myvector_t x, y;

  myfloat_t min = -pixel_size * (n_pixels + 1)*0.5;
  myfloat_t max = pixel_size * (n_pixels - 3)*0.5 + pixel_size;

  //Assign memory space required to fill the vectors
  x.resize(n_pixels); y.resize(n_pixels);

  //Generate them
  arange(x, min, max, pixel_size);

  //Vectors used for masked selection of coordinates
  std::vector <size_t> x_sel;
  where(x, x_sel, getPosition(0)[0], cutoff * sigma);

  const myfloat_t counter = x_sel.size();
  
  Vector distance=delta(getPosition(0),getPosition(1));
  setAtomsDerivatives(0, 1 * distance);
  setBoxDerivativesNoPbc();
  setValue(counter);
}

}
}



