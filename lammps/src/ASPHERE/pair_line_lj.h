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
PairStyle(line/lj,PairLineLJ);
// clang-format on
#else

#ifndef LMP_PAIR_LINE_LJ_H
#define LMP_PAIR_LINE_LJ_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLineLJ : public Pair {
 public:
  PairLineLJ(class LAMMPS *);
  virtual ~PairLineLJ();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  double init_one(int, int);

 protected:
  double cut_global;
  double *subsize;
  double **epsilon, **sigma, **cutsub, **cutsubsq;
  double **cut;
  double **lj1, **lj2, **lj3, **lj4;    // for sphere/sphere interactions
  class AtomVecLine *avec;

  double *size;    // per-type size of sub-particles to tile line segment

  struct Discrete {
    double dx, dy;
  };
  Discrete *discrete;    // list of all discretes for all lines
  int ndiscrete;         // number of discretes in list
  int dmax;              // allocated size of discrete list
  int *dnum;             // number of discretes per line, 0 if uninit
  int *dfirst;           // index of first discrete per each line
  int nmax;              // allocated size of dnum,dfirst vectors

  void allocate();
  void discretize(int, double);
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

E: Pair line/lj requires atom style line

Self-explanatory.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
