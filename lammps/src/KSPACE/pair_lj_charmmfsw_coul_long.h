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
PairStyle(lj/charmmfsw/coul/long,PairLJCharmmfswCoulLong);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CHARMMFSW_COUL_LONG_H
#define LMP_PAIR_LJ_CHARMMFSW_COUL_LONG_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCharmmfswCoulLong : public Pair {
 public:
  PairLJCharmmfswCoulLong(class LAMMPS *);
  virtual ~PairLJCharmmfswCoulLong();

  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  virtual double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);

  void compute_inner();
  void compute_middle();
  virtual void compute_outer(int, int);
  virtual void *extract(const char *, int &);

 protected:
  int implicit;
  int dihedflag;

  double cut_lj_inner, cut_lj, cut_ljinv, cut_lj_innerinv;
  double cut_lj_innersq, cut_ljsq;
  double cut_lj3inv, cut_lj_inner3inv, cut_lj3, cut_lj_inner3;
  double cut_lj6inv, cut_lj_inner6inv, cut_lj6, cut_lj_inner6;
  double cut_coul, cut_coulsq;
  double cut_bothsq;
  double denom_lj, denom_lj12, denom_lj6;
  double **epsilon, **sigma, **eps14, **sigma14;
  double **lj1, **lj2, **lj3, **lj4, **offset;
  double **lj14_1, **lj14_2, **lj14_3, **lj14_4;
  double *cut_respa;
  double g_ewald;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style lj/charmmfsw/coul/long requires atom attribute q

The atom style defined does not have these attributes.

E: Pair inner cutoff >= Pair outer cutoff

The specified cutoffs for the pair style are inconsistent.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

E: Pair inner cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

E: Pair style requires a KSpace style

No kspace style is defined.

*/
