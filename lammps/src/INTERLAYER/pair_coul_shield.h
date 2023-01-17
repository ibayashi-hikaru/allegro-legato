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
PairStyle(coul/shield,PairCoulShield);
// clang-format on
#else

#ifndef LMP_PAIR_COUL_SHIELD_H
#define LMP_PAIR_COUL_SHIELD_H

#include "pair.h"

namespace LAMMPS_NS {

class PairCoulShield : public Pair {
 public:
  PairCoulShield(class LAMMPS *);
  virtual ~PairCoulShield();

  virtual void compute(int, int);

  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  virtual void init_style();
  virtual double init_one(int, int);
  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);

  virtual double single(int, int, int, int, double, double, double, double &);

 protected:
  double cut_global;
  double **cut;
  double **sigmae, **offset;
  int tap_flag;

  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
