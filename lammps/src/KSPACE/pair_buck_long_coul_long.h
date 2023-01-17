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
PairStyle(buck/long/coul/long,PairBuckLongCoulLong);
// clang-format on
#else

#ifndef LMP_PAIR_BUCK_LONG_COUL_LONG_H
#define LMP_PAIR_BUCK_LONG_COUL_LONG_H

#include "pair.h"

namespace LAMMPS_NS {

class PairBuckLongCoulLong : public Pair {
 public:
  double cut_coul;

  PairBuckLongCoulLong(class LAMMPS *);
  ~PairBuckLongCoulLong();
  virtual void compute(int, int);

  virtual void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

  virtual void compute_inner();
  virtual void compute_middle();
  virtual void compute_outer(int, int);

 protected:
  double cut_buck_global;
  double **cut_buck, **cut_buck_read, **cut_bucksq;
  double cut_coulsq;
  double **buck_a_read, **buck_a, **buck_c_read, **buck_c;
  double **buck1, **buck2, **buck_rho_read, **buck_rho, **rhoinv, **offset;
  double *cut_respa;
  double g_ewald;
  double g_ewald_6;
  int ewald_order, ewald_off;

  void options(char **arg, int order);
  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

W: Using largest cutoff for buck/long/coul/long

Self-explanatory.

E: Cutoffs missing in pair_style buck/long/coul/long

Self-explanatory.

E: LJ6 off not supported in pair_style buck/long/coul/long

Self-explanatory.

E: Coulomb cut not supported in pair_style buck/long/coul/coul

Must use long-range Coulombic interactions.

E: Only one cutoff allowed when requesting all long

Self-explanatory.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Invoking coulombic in pair style buck/long/coul/long requires atom attribute q

UNDOCUMENTED

E: Pair style requires a KSpace style

No kspace style is defined.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

U: Pair style buck/long/coul/long requires atom attribute q

The atom style defined does not have this attribute.

*/
