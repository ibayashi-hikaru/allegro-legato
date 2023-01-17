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
ComputeStyle(erotate/rigid,ComputeERotateRigid);
// clang-format on
#else

#ifndef LMP_COMPUTE_EROTATE_RIGID_H
#define LMP_COMPUTE_EROTATE_RIGID_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeERotateRigid : public Compute {
 public:
  ComputeERotateRigid(class LAMMPS *, int, char **);
  ~ComputeERotateRigid();
  void init();
  double compute_scalar();

 private:
  int irfix;
  char *rfix;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix ID for compute erotate/rigid does not exist

Self-explanatory.

E: Compute erotate/rigid with non-rigid fix-ID

Self-explanatory.

*/
