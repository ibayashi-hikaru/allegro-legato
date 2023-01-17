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
PairStyle(lj/cut/soft,PairLJCutSoft);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_SOFT_H
#define LMP_PAIR_LJ_CUT_SOFT_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCutSoft : public Pair {
 public:
  PairLJCutSoft(class LAMMPS *);
  virtual ~PairLJCutSoft();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  virtual double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

  void compute_inner();
  void compute_middle();
  void compute_outer(int, int);

 protected:
  double cut_global;
  double **cut;
  double **epsilon, **sigma, **lambda;
  double nlambda, alphalj;
  double **lj1, **lj2, **lj3, **offset;
  double *cut_respa;

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

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

E: Pair lj/cut/soft different lambda values in mix

The value of lambda has to be the same for I J pairs.

*/
