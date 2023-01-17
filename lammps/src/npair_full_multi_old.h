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

#ifdef NPAIR_CLASS
// clang-format off
NPairStyle(full/multi/old,
           NPairFullMultiOld,
           NP_FULL | NP_MULTI_OLD | NP_NEWTON | NP_NEWTOFF | NP_ORTHO | NP_TRI);
// clang-format on
#else

#ifndef LMP_NPAIR_FULL_MULTI_OLD_H
#define LMP_NPAIR_FULL_MULTI_OLD_H

#include "npair.h"

namespace LAMMPS_NS {

class NPairFullMultiOld : public NPair {
 public:
  NPairFullMultiOld(class LAMMPS *);
  ~NPairFullMultiOld() {}
  void build(class NeighList *);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Neighbor list overflow, boost neigh_modify one

UNDOCUMENTED

*/
