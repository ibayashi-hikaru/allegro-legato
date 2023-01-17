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

#ifdef FIX_CLASS
// clang-format off
FixStyle(tfmc,FixTFMC);
// clang-format on
#else

#ifndef LMP_FIX_TFMC_H
#define LMP_FIX_TFMC_H

#include "fix.h"

namespace LAMMPS_NS {

class FixTFMC : public Fix {
 public:
  FixTFMC(class LAMMPS *, int, char **);
  ~FixTFMC();
  int setmask();
  void init();
  void initial_integrate(int);

 private:
  double d_max;
  double T_set;
  double mass_min;
  double **xd;
  int mass_require;
  int seed;
  int comflag, rotflag, xflag, yflag, zflag;
  int nmax;
  class RanMars *random_num;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix tfmc displacement length must be > 0

Self-explanatory.

E: Fix tfmc temperature must be > 0

Self-explanatory.

E: Illegal fix tfmc random seed

Seeds can only be nonzero positive integers.

E: Fix tfmc is not compatible with fix shake

These two commands cannot currently be used together.

*/
