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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/coul/debye/gpu,PairLJCutCoulDebyeGPU);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_COUL_DEBYE_GPU_H
#define LMP_PAIR_LJ_CUT_COUL_DEBYE_GPU_H

#include "pair_lj_cut_coul_debye.h"

namespace LAMMPS_NS {

class PairLJCutCoulDebyeGPU : public PairLJCutCoulDebye {
 public:
  PairLJCutCoulDebyeGPU(LAMMPS *lmp);
  ~PairLJCutCoulDebyeGPU();
  void cpu_compute(int, int, int, int, int *, int *, int **);
  void compute(int, int);
  void init_style();
  double memory_usage();

  enum { GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH };

 private:
  int gpu_mode;
  double cpu_time;
};

}    // namespace LAMMPS_NS
#endif
#endif

/* ERROR/WARNING messages:

E: Insufficient memory on accelerator

There is insufficient memory on one of the devices specified for the gpu
package

E: Pair style lj/cut/coul/debye/gpu requires atom attribute q

The atom style defined does not have this attribute.

E: Cannot use newton pair with lj/cut/coul/debye/gpu pair style

Self-explanatory.

*/
