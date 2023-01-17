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

#ifdef IMPROPER_CLASS
// clang-format off
ImproperStyle(class2,ImproperClass2);
// clang-format on
#else

#ifndef LMP_IMPROPER_CLASS2_H
#define LMP_IMPROPER_CLASS2_H

#include "improper.h"

namespace LAMMPS_NS {

class ImproperClass2 : public Improper {
 public:
  ImproperClass2(class LAMMPS *);
  virtual ~ImproperClass2();
  virtual void compute(int, int);
  virtual void coeff(int, char **);
  void write_restart(FILE *);
  virtual void read_restart(FILE *);
  void write_data(FILE *);

 protected:
  double *k0, *chi0;
  double *aa_k1, *aa_k2, *aa_k3, *aa_theta0_1, *aa_theta0_2, *aa_theta0_3;
  int *setflag_i, *setflag_aa;

  void allocate();
  void angleangle(int, int);
  void cross(double *, double *, double *);
  double dot(double *, double *);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

W: Improper problem: %d %ld %d %d %d %d

Conformation of the 4 listed improper atoms is extreme; you may want
to check your simulation geometry.

E: Incorrect args for improper coefficients

Self-explanatory.  Check the input script or data file.

*/
