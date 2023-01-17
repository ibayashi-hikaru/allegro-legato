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

#ifdef INTEGRATE_CLASS
// clang-format off
IntegrateStyle(respa,Respa);
// clang-format on
#else

#ifndef LMP_RESPA_H
#define LMP_RESPA_H

#include "integrate.h"

namespace LAMMPS_NS {

class Respa : public Integrate {
 public:
  // public so Fixes, Pairs, Neighbor can see them
  int nlevels;         // number of rRESPA levels
                       // 0 = innermost level, nlevels-1 = outermost level
  double *step;        // timestep at each level
  int *loop;           // sub-cycling factor at each level
  double cutoff[4];    // cutoff[0] and cutoff[1] = between inner and middle
                       // cutoff[2] and cutoff[3] = between middle and outer
                       // if no middle then 0,1 = 2,3

  int level_bond, level_angle, level_dihedral;    // level to compute forces at
  int level_improper, level_pair, level_kspace;
  int level_inner, level_middle, level_outer;

  int nhybrid_styles;     // number of hybrid pair styles
  int *hybrid_level;      // level to compute pair hybrid sub-style at
  int *hybrid_compute;    // selects whether to compute sub-style forces
  int tally_global;       // 1 if pair style should tally global accumulators
  int pair_compute;       // 1 if pair force need to be computed

  Respa(class LAMMPS *, int, char **);
  virtual ~Respa();
  virtual void init();
  virtual void setup(int);
  virtual void setup_minimal(int);
  virtual void run(int);
  virtual void cleanup();
  virtual void reset_dt();

  void copy_f_flevel(int);
  void copy_flevel_f(int);

 protected:
  int triclinic;    // 0 if domain is orthog, 1 if triclinic
  int torqueflag, extraflag;

  int *newton;                  // newton flag at each level
  class FixRespa *fix_respa;    // Fix to store the force level array

  virtual void recurse(int);
  void force_clear(int);
  void sum_flevel_f();
  void set_compute_flags(int ilevel);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Respa levels must be >= 1

Self-explanatory.

E: Cannot set both respa pair and inner/middle/outer

In the rRESPA integrator, you must compute pairwise potentials either
all together (pair), or in pieces (inner/middle/outer).  You can't do
both.

E: Must set both respa inner and outer

Cannot use just the inner or outer option with respa without using the
other.

E: Cannot set respa middle without inner/outer

In the rRESPA integrator, you must define both a inner and outer
setting in order to use a middle setting.

E: Cannot set respa hybrid and any of pair/inner/middle/outer

In the rRESPA integrator, you must compute pairwise potentials either
all together (pair), with different cutoff regions (inner/middle/outer),
or per hybrid sub-style (hybrid).  You cannot mix those.

E: Invalid order of forces within respa levels

For respa, ordering of force computations within respa levels must
obey certain rules.  E.g. bonds cannot be compute less frequently than
angles, pairwise forces cannot be computed less frequently than
kspace, etc.

W: One or more respa levels compute no forces

This is computationally inefficient.

E: Respa inner cutoffs are invalid

The first cutoff must be <= the second cutoff.

E: Respa middle cutoffs are invalid

The first cutoff must be <= the second cutoff.

W: No fixes defined, atoms won't move

If you are not using a fix like nve, nvt, npt then atom velocities and
coordinates will not be updated during timestepping.

E: Pair style does not support rRESPA inner/middle/outer

You are attempting to use rRESPA options with a pair style that
does not support them.

*/
