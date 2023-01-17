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

/* ----------------------------------------------------------------------
   Contributing author: Zheng GONG (ENS de Lyon, z.gong@outlook.com)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(accelerate/cos,FixAccelerateCos);
// clang-format on
#else

#ifndef LMP_FIX_ACCELERATE_COS_H
#define LMP_FIX_ACCELERATE_COS_H

#include "fix.h"

namespace LAMMPS_NS {

class FixAccelerateCos : public Fix {
 public:
  FixAccelerateCos(class LAMMPS *, int, char **);
  virtual ~FixAccelerateCos(){};
  int setmask();
  virtual void init(){};
  void setup(int);
  virtual void post_force(int);

 protected:
  double acceleration;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix accelerate/cos cannot be used with 2d systems

Self-explanatory.

*/
