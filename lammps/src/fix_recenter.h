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
FixStyle(recenter,FixRecenter);
// clang-format on
#else

#ifndef LMP_FIX_RECENTER_H
#define LMP_FIX_RECENTER_H

#include "fix.h"

namespace LAMMPS_NS {

class FixRecenter : public Fix {
 public:
  FixRecenter(class LAMMPS *, int, char **);
  int setmask();
  void init();
  void initial_integrate(int);
  void initial_integrate_respa(int, int, int);
  double compute_scalar();
  double compute_vector(int);

 private:
  int group2bit, scaleflag;
  int xflag, yflag, zflag;
  int xinitflag, yinitflag, zinitflag;
  int nlevels_respa;
  double xcom, ycom, zcom, xinit, yinit, zinit, masstotal, distance, shift[3];
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Could not find fix recenter group ID

A group ID used in the fix recenter command does not exist.

E: Fix recenter group has no atoms

Self-explanatory.

W: Fix recenter should come after all other integration fixes

Other fixes may change the position of the center-of-mass, so
fix recenter should come last.

*/
