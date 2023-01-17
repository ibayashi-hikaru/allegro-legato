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
FixStyle(rigid/nve/omp,FixRigidNVEOMP);
// clang-format on
#else

#ifndef LMP_FIX_RIGID_NVE_OMP_H
#define LMP_FIX_RIGID_NVE_OMP_H

#include "fix_rigid_nh_omp.h"

namespace LAMMPS_NS {

class FixRigidNVEOMP : public FixRigidNHOMP {
 public:
  FixRigidNVEOMP(class LAMMPS *, int, char **);
  ~FixRigidNVEOMP() {}
};

}    // namespace LAMMPS_NS

#endif
#endif
