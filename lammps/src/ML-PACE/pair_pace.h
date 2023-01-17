/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/*
Copyright 2021 Yury Lysogorskiy^1, Cas van der Oord^2, Anton Bochkarev^1,
 Sarath Menon^1, Matteo Rinaldi^1, Thomas Hammerschmidt^1, Matous Mrovec^1,
 Aidan Thompson^3, Gabor Csanyi^2, Christoph Ortner^4, Ralf Drautz^1

^1: Ruhr-University Bochum, Bochum, Germany
^2: University of Cambridge, Cambridge, United Kingdom
^3: Sandia National Laboratories, Albuquerque, New Mexico, USA
^4: University of British Columbia, Vancouver, BC, Canada
*/

//
// Created by Lysogorskiy Yury on 27.02.20.
//

#ifdef PAIR_CLASS
// clang-format off
PairStyle(pace,PairPACE);
// clang-format on
#else

#ifndef LMP_PAIR_PACE_H
#define LMP_PAIR_PACE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairPACE : public Pair {
 public:
  PairPACE(class LAMMPS *);
  virtual ~PairPACE();

  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  double init_one(int, int);

  void *extract(const char *, int &);

 protected:
  struct ACEImpl *aceimpl;

  virtual void allocate();

  double **scale;
  bool recursive;    // "recursive" option for ACERecursiveEvaluator
};
}    // namespace LAMMPS_NS

#endif
#endif
