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
ComputeStyle(pair,ComputePair);
// clang-format on
#else

#ifndef LMP_COMPUTE_PAIR_H
#define LMP_COMPUTE_PAIR_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputePair : public Compute {
 public:
  ComputePair(class LAMMPS *, int, char **);
  ~ComputePair();
  void init();
  double compute_scalar();
  void compute_vector();

 private:
  int evalue, npair, nsub;
  char *pstyle;
  class Pair *pair;
  double *one;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Unrecognized pair style in compute pair command

Self-explanatory.

E: Energy was not tallied on needed timestep

You are using a thermo keyword that requires potentials to
have tallied energy, but they didn't on this timestep.  See the
variable doc page for ideas on how to make this work.

U: Compute pair must use group all

Pair styles accumulate energy on all atoms.

*/
