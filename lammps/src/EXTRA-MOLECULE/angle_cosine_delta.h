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

#ifdef ANGLE_CLASS
// clang-format off
AngleStyle(cosine/delta,AngleCosineDelta);
// clang-format on
#else

#ifndef LMP_ANGLE_COSINE_DELTA_H
#define LMP_ANGLE_COSINE_DELTA_H

#include "angle_cosine_squared.h"

namespace LAMMPS_NS {

class AngleCosineDelta : public AngleCosineSquared {
 public:
  AngleCosineDelta(class LAMMPS *);
  virtual void compute(int, int);
  double single(int, int, int, int);
};

}    // namespace LAMMPS_NS

#endif
#endif
/* ERROR/WARNING messages:

*/
