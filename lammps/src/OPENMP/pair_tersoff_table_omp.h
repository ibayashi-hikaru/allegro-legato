/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(tersoff/table/omp,PairTersoffTableOMP);
// clang-format on
#else

#ifndef LMP_PAIR_TERSOFF_TABLE_OMP_H
#define LMP_PAIR_TERSOFF_TABLE_OMP_H

#include "pair_tersoff_table.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

class PairTersoffTableOMP : public PairTersoffTable, public ThrOMP {

 public:
  PairTersoffTableOMP(class LAMMPS *);
  virtual ~PairTersoffTableOMP();

  virtual void compute(int, int);
  virtual double memory_usage();

 protected:
  double ***thrGtetaFunction, ***thrGtetaFunctionDerived;
  double **thrCutoffFunction, **thrCutoffFunctionDerived;

  void allocatePreLoops(void);
  void deallocatePreLoops(void);

 private:
  template <int EVFLAG, int VFLAG_ATOM> void eval(int ifrom, int ito, ThrData *const thr);
};

}    // namespace LAMMPS_NS

#endif
#endif
