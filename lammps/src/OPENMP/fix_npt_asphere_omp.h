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
FixStyle(npt/asphere/omp,FixNPTAsphereOMP);
// clang-format on
#else

#ifndef LMP_FIX_NPT_ASPHERE_OMP_H
#define LMP_FIX_NPT_ASPHERE_OMP_H

#include "fix_nh_asphere_omp.h"

namespace LAMMPS_NS {

class FixNPTAsphereOMP : public FixNHAsphereOMP {
 public:
  FixNPTAsphereOMP(class LAMMPS *, int, char **);
  virtual ~FixNPTAsphereOMP() {}
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Temperature control must be used with fix npt/asphere

Self-explanatory.

E: Pressure control must be used with fix npt/asphere

Self-explanatory.

*/
