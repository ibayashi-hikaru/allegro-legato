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
AtomStyle(tdpd,AtomVecTDPD);
// clang-format on
#else

#ifndef LMP_ATOM_VEC_TDPD_H
#define LMP_ATOM_VEC_TDPD_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecTDPD : public AtomVec {
 public:
  AtomVecTDPD(class LAMMPS *);
  void process_args(int, char **);
  void init();

  void grow_pointers();
  void force_clear(int, size_t);
  void data_atom_post(int);

 protected:
  double **cc_flux;
  double **vest;

  int cc_species;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Per-processor system is too big

The number of owned atoms plus ghost atoms on a single
processor must fit in 32-bit integer.

E: Invalid atom type in Atoms section of data file

Atom types must range from 1 to specified # of types.

*/
