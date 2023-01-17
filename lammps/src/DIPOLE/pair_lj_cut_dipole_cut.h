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
PairStyle(lj/cut/dipole/cut,PairLJCutDipoleCut);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_DIPOLE_CUT_H
#define LMP_PAIR_LJ_CUT_DIPOLE_CUT_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCutDipoleCut : public Pair {
 public:
  PairLJCutDipoleCut(class LAMMPS *);
  virtual ~PairLJCutDipoleCut();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void *extract(const char *, int &);

 protected:
  double cut_lj_global, cut_coul_global;
  double **cut_lj, **cut_ljsq;
  double **cut_coul, **cut_coulsq;
  double **epsilon, **sigma;
  double **lj1, **lj2, **lj3, **lj4, **offset;

  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args in pair_style command

Self-explanatory.

E: Cannot (yet) use 'electron' units with dipoles

This feature is not yet supported.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair dipole/cut requires atom attributes q, mu, torque

The atom style defined does not have these attributes.

*/
