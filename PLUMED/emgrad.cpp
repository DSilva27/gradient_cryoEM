
#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"
#include "core/PlumedMain.h"
#include "tools/Pbc.h"

#include <string>
#include <cmath>

namespace PLMD {
namespace isdb {

class EmGrad : public Colvar {
  bool pbc;
  double Sigma;
  double Cutoff;

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
  keys.add("compulsory","Sigma","Standard deviation of the gaussians.");
  keys.add("compulsory","cutoff","Neighbor cutoff.");
}

EmGrad::EmGrad(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true)
{
  std::vector<AtomNumber> atoms;
  parseAtomList("ATOMS",atoms);
  if(atoms.size()<1)
    error("You should define at least one atom");
  parse("Sigma",R0_);
  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;
  checkRead();

  log.printf("  between atoms %d %d\n",atoms[0].serial(),atoms[1].serial());
  log.printf("  with Forster radius set to %lf\n",R0_);

  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");

  log << " Bibliography" << plumed.cite("Bonomi, Camilloni, Bioinformatics, 33, 3999 (2017)") << "\n";

  addValueWithDerivatives();
  setNotPeriodic();

  requestAtoms(atoms);
}


// calculator
void EmGrad::calculate() {

  if(pbc) makeWhole();

  Vector distance=delta(getPosition(0),getPosition(1));
  const double dist_mod=distance.modulo();
  const double inv_dist_mod=1.0/dist_mod;

  const double ratiosix=std::pow(dist_mod/R0_,6);
  const double fret_eff = 1.0/(1.0+ratiosix);

  const double der = -6.0*fret_eff*fret_eff*ratiosix*inv_dist_mod;

  setAtomsDerivatives(0,-inv_dist_mod*der*distance);
  setAtomsDerivatives(1, inv_dist_mod*der*distance);
  setBoxDerivativesNoPbc();
  setValue(fret_eff);

}

}
}



