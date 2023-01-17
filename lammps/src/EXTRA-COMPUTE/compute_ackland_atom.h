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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(ackland/atom,ComputeAcklandAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_ACKLAND_ATOM_H
#define LMP_COMPUTE_ACKLAND_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeAcklandAtom : public Compute {
 public:
  ComputeAcklandAtom(class LAMMPS *, int, char **);
  ~ComputeAcklandAtom();
  void init();
  void init_list(int, class NeighList *);
  void compute_peratom();
  double memory_usage();

 private:
  int nmax, maxneigh, legacy;
  double *distsq;
  int *nearest, *nearest_n0, *nearest_n1;
  double *structure;
  class NeighList *list;

  void select(int, int, double *);
  void select2(int, int, double *, int *);
};

}    // namespace LAMMPS_NS

#endif
#endif
