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
PairStyle(born/coul/wolf,PairBornCoulWolf);
// clang-format on
#else

#ifndef LMP_PAIR_BORN_COUL_WOLF_H
#define LMP_PAIR_BORN_COUL_WOLF_H

#include "pair.h"

namespace LAMMPS_NS {

class PairBornCoulWolf : public Pair {
 public:
  PairBornCoulWolf(class LAMMPS *);
  virtual ~PairBornCoulWolf();
  virtual void compute(int, int);
  void settings(int, char **);
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

 protected:
  double cut_lj_global, alf;
  double **cut_lj, **cut_ljsq;
  double cut_coul, cut_coulsq;
  double **a, **rho, **sigma, **c, **d;
  double **rhoinv, **born1, **born2, **born3, **offset;

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

E: Pair style born/coul/wolf requires atom attribute q

The atom style defined does not have this attribute.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
