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
PairStyle(buck6d/coul/gauss/dsf,PairBuck6dCoulGaussDSF);
// clang-format on
#else

#ifndef LMP_PAIR_BUCK6D_COUL_GAUSS_DSF_H
#define LMP_PAIR_BUCK6D_COUL_GAUSS_DSF_H

#include "pair.h"

namespace LAMMPS_NS {

class PairBuck6dCoulGaussDSF : public Pair {
 public:
  PairBuck6dCoulGaussDSF(class LAMMPS *);
  virtual ~PairBuck6dCoulGaussDSF();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  virtual double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

 protected:
  double cut_lj_global;
  double **cut_lj, **cut_ljsq;
  double **alpha_ij;
  double **buck6d1, **buck6d2, **buck6d3, **buck6d4, **offset;
  double **f_shift_ij, **e_shift_ij;
  double cut_coul, cut_coulsq;
  double vdwl_smooth;
  double **c0, **c1, **c2, **c3, **c4, **c5, **rsmooth_sq;

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

E: Pair style lj/cut/coul/dsf requires atom attribute q

The atom style defined does not have these attributes.

*/
