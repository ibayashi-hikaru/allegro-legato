/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Contributing author: Maxim Shugaev (UVA), mvs9t@virginia.edu
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(mesont,ComputeMesoNT);
// clang-format on
#else

#ifndef LMP_COMPUTE_MESONT_ATOM_H
#define LMP_COMPUTE_MESONT_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeMesoNT : public Compute {
 public:
  ComputeMesoNT(class LAMMPS *, int, char **);
  ~ComputeMesoNT();
  void init() {}
  void compute_peratom();
  double compute_scalar();
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();

 private:
  int nmax;
  double *energy;

  enum ComputeType { ES, EB, ET };
  ComputeType compute_type;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal compute mesont command

Incorrect argument list in the compute init.

E: Per-atom energy was not tallied on needed timestep

UNSPECIFIED.

E: compute mesont is allowed only with mesont/tpm pair style

Use mesont pair style.

*/
