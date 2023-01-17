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
NPairStyle(copy,
           NPairCopy,
           NP_COPY);
// clang-format on
#else

#ifndef LMP_NPAIR_COPY_H
#define LMP_NPAIR_COPY_H

#include "npair.h"

namespace LAMMPS_NS {

class NPairCopy : public NPair {
 public:
  NPairCopy(class LAMMPS *);
  ~NPairCopy() {}
  void build(class NeighList *);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
