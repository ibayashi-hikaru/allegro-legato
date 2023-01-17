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
FixStyle(qeq/comb,FixQEQComb);
// clang-format on
#else

#ifndef LMP_FIX_QEQ_COMB_H
#define LMP_FIX_QEQ_COMB_H

#include "fix.h"

namespace LAMMPS_NS {

class FixQEQComb : public Fix {
 public:
  FixQEQComb(class LAMMPS *, int, char **);
  virtual ~FixQEQComb();
  int setmask();
  virtual void init();
  void setup(int);
  virtual void post_force(int);
  void post_force_respa(int, int, int);
  double memory_usage();
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);

  void min_post_force(int);

 protected:
  int me, firstflag;
  double precision;
  int ilevel_respa;
  bigint ngroup;
  FILE *fp;

  class PairComb *comb;
  class PairComb3 *comb3;
  int nmax;
  double *qf, *q1, *q2;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Cannot open fix qeq/comb file %s

The output file for the fix qeq/combs command cannot be opened.
Check that the path and name are correct.

E: Fix qeq/comb requires atom attribute q

An atom style with charge must be used to perform charge equilibration.

E: Must use pair_style comb or comb3 with fix qeq/comb

Self-explanatory.

E: Fix qeq/comb group has no atoms

Self-explanatory.

*/
