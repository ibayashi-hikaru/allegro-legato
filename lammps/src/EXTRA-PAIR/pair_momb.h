/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
  -------------------------------------------------------------------------
  Contributed by Kristen Fichthorn, Tonnam Balankura, Ya Zhou @ Penn State University
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(momb,PairMomb);
// clang-format on
#else

#ifndef LMP_PAIR_MOMB_H
#define LMP_PAIR_MOMB_H

#include "pair.h"

namespace LAMMPS_NS {

class PairMomb : public Pair {
 public:
  PairMomb(class LAMMPS *);
  virtual ~PairMomb();

  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);

 protected:
  double cut_global;
  double **cut;
  double sscale, dscale;
  double **d0, **alpha, **r0, **c, **rr;
  double **morse1;
  double **offset;

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

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
