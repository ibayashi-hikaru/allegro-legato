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
FixStyle(READ_RESTART,FixReadRestart);
// clang-format on
#else

#ifndef LMP_FIX_READ_RESTART_H
#define LMP_FIX_READ_RESTART_H

#include "fix.h"

namespace LAMMPS_NS {

class FixReadRestart : public Fix {
 public:
  int *count;
  double **extra;

  FixReadRestart(class LAMMPS *, int, char **);
  ~FixReadRestart();
  int setmask();

  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

 private:
  int nextra;    // max number of extra values for any atom
};

}    // namespace LAMMPS_NS

#endif
#endif
/* ERROR/WARNING messages:

*/
