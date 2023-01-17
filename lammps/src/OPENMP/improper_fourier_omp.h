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
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#ifdef IMPROPER_CLASS
// clang-format off
ImproperStyle(fourier/omp,ImproperFourierOMP);
// clang-format on
#else

#ifndef LMP_IMPROPER_FOURIER_OMP_H
#define LMP_IMPROPER_FOURIER_OMP_H

#include "improper_fourier.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

class ImproperFourierOMP : public ImproperFourier, public ThrOMP {

 public:
  ImproperFourierOMP(class LAMMPS *lmp);
  virtual void compute(int, int);

 private:
  template <int EVFLAG, int EFLAG, int NEWTON_BOND>
  void eval(int ifrom, int ito, ThrData *const thr);

  template <int EVFLAG, int EFLAG, int NEWTON_BOND>
  void add1_thr(const int, const int, const int, const int, const int, const double &,
                const double &, const double &, const double &, const double &, const double &,
                const double &, const double &, const double &, ThrData *const thr);
};

}    // namespace LAMMPS_NS

#endif
#endif
