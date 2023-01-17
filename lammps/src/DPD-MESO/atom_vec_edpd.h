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

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(edpd,AtomVecEDPD);
// clang-format on
#else

#ifndef LMP_ATOM_VEC_EDPD_H
#define LMP_ATOM_VEC_EDPD_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecEDPD : public AtomVec {
 public:
  AtomVecEDPD(class LAMMPS *);
  void init();

  void grow_pointers();
  void force_clear(int, size_t);
  void create_atom_post(int);
  void data_atom_post(int);

 private:
  double *edpd_cv, *edpd_temp, *edpd_flux;
  double **vest;
  double *vest_temp;
};

}    // namespace LAMMPS_NS

#endif
#endif
