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

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(ewald/omp,EwaldOMP);
// clang-format on
#else

#ifndef LMP_EWALD_OMP_H
#define LMP_EWALD_OMP_H

#include "ewald.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

class EwaldOMP : public Ewald, public ThrOMP {
 public:
  EwaldOMP(class LAMMPS *);
  virtual ~EwaldOMP(){};
  virtual void allocate();
  virtual void compute(int, int);

 protected:
  virtual void eik_dot_r();
  virtual void eik_dot_r_triclinic();
};

}    // namespace LAMMPS_NS

#endif
#endif
