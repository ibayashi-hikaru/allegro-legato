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
FixStyle(nph,FixNPH);
// clang-format on
#else

#ifndef LMP_FIX_NPH_H
#define LMP_FIX_NPH_H

#include "fix_nh.h"

namespace LAMMPS_NS {

class FixNPH : public FixNH {
 public:
  FixNPH(class LAMMPS *, int, char **);
  ~FixNPH() {}
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Temperature control can not be used with fix nph

Self-explanatory.

E: Pressure control must be used with fix nph

Self-explanatory.

*/
