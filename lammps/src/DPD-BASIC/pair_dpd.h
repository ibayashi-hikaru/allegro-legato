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
PairStyle(dpd,PairDPD);
// clang-format on
#else

#ifndef LMP_PAIR_DPD_H
#define LMP_PAIR_DPD_H

#include "pair.h"

namespace LAMMPS_NS {

class PairDPD : public Pair {
 public:
  PairDPD(class LAMMPS *);
  virtual ~PairDPD();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  virtual void write_data(FILE *);
  virtual void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);

 protected:
  double cut_global, temperature;
  int seed;
  double **cut;
  double **a0, **gamma;
  double **sigma;
  class RanMars *random;

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

E: Pair dpd requires ghost atoms store velocity

Use the comm_modify vel yes command to enable this.

W: Pair dpd needs newton pair on for momentum conservation

Self-explanatory.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
