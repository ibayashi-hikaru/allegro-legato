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
FixStyle(npt/kk,FixNPTKokkos<LMPDeviceType>);
FixStyle(npt/kk/device,FixNPTKokkos<LMPDeviceType>);
FixStyle(npt/kk/host,FixNPTKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_NPT_KOKKOS_H
#define LMP_FIX_NPT_KOKKOS_H

#include "fix_nh_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class FixNPTKokkos : public FixNHKokkos<DeviceType> {
 public:
  FixNPTKokkos(class LAMMPS *, int, char **);
  ~FixNPTKokkos() {}
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Temperature control must be used with fix npt

Self-explanatory.

E: Pressure control must be used with fix npt

Self-explanatory.

*/
