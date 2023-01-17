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

#ifdef NTOPO_CLASS
// clang-format off
NTopoStyle(NTOPO_BOND_ALL,NTopoBondAll);
// clang-format on
#else

#ifndef LMP_TOPO_BOND_ALL_H
#define LMP_TOPO_BOND_ALL_H

#include "ntopo.h"

namespace LAMMPS_NS {

class NTopoBondAll : public NTopo {
 public:
  NTopoBondAll(class LAMMPS *);
  ~NTopoBondAll() {}
  void build();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Bond atoms %d %d missing on proc %d at step %ld

UNDOCUMENTED

W: Bond atoms missing at step %ld

UNDOCUMENTED

*/
