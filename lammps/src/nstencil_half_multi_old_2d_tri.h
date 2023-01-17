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

#ifdef NSTENCIL_CLASS
// clang-format off
NStencilStyle(half/multi/old/2d/tri,
              NStencilHalfMultiOld2dTri, NS_HALF | NS_MULTI_OLD | NS_2D | NS_TRI);
// clang-format on
#else

#ifndef LMP_NSTENCIL_HALF_MULTI_OLD_2D_TRI_H
#define LMP_NSTENCIL_HALF_MULTI_OLD_2D_TRI_H

#include "nstencil.h"

namespace LAMMPS_NS {

class NStencilHalfMultiOld2dTri : public NStencil {
 public:
  NStencilHalfMultiOld2dTri(class LAMMPS *);
  ~NStencilHalfMultiOld2dTri() {}
  void create();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
