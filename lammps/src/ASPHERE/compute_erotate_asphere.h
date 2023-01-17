/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(erotate/asphere,ComputeERotateAsphere);
// clang-format on
#else

#ifndef LMP_COMPUTE_EROTATE_ASPHERE_H
#define LMP_COMPUTE_EROTATE_ASPHERE_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeERotateAsphere : public Compute {
 public:
  ComputeERotateAsphere(class LAMMPS *, int, char **);
  void init();
  double compute_scalar();

 private:
  double pfactor;
  class AtomVecEllipsoid *avec_ellipsoid;
  class AtomVecLine *avec_line;
  class AtomVecTri *avec_tri;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute erotate/asphere requires atom style ellipsoid or line or tri

Self-explanatory.

E: Compute erotate/asphere requires extended particles

This compute cannot be used with point particles.

*/
