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

#ifdef FIX_CLASS
// clang-format off
FixStyle(deform/kk,FixDeformKokkos);
FixStyle(deform/kk/device,FixDeformKokkos);
FixStyle(deform/kk/host,FixDeformKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_DEFORM_KOKKOS_H
#define LMP_FIX_DEFORM_KOKKOS_H

#include "fix_deform.h"

namespace LAMMPS_NS {

class FixDeformKokkos : public FixDeform {
 public:

  FixDeformKokkos(class LAMMPS *, int, char **);
  virtual ~FixDeformKokkos() {}
  void pre_exchange();
  void end_of_step();

 private:
  class DomainKokkos *domainKK;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot (yet) use rigid bodies with fix deform and Kokkos

Self-explanatory.

*/
