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
FixStyle(wall/region,FixWallRegion);
// clang-format on
#else

#ifndef LMP_FIX_WALL_REGION_H
#define LMP_FIX_WALL_REGION_H

#include "fix.h"

namespace LAMMPS_NS {

class FixWallRegion : public Fix {
 public:
  FixWallRegion(class LAMMPS *, int, char **);
  ~FixWallRegion();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_scalar();
  double compute_vector(int);

 private:
  int style, iregion;
  double epsilon, sigma, cutoff;
  double alpha;
  int eflag;
  double ewall[4], ewall_all[4];
  int ilevel_respa;
  char *idregion;

  double coeff1, coeff2, coeff3, coeff4, offset;
  double coeff5, coeff6, coeff7;
  double eng, fwall;

  void lj93(double);
  void lj126(double);
  void lj1043(double);
  void morse(double);
  void colloid(double, double);
  void harmonic(double);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix wall/region does not exist

Self-explanatory.

E: Fix wall/region cutoff <= 0.0

Self-explanatory.

E: Fix wall/region colloid requires atom style sphere

Self-explanatory.

E: Fix wall/region colloid requires extended particles

One of the particles has radius 0.0.

E: Particle outside surface of region used in fix wall/region

Particles must be inside the region for energy/force to be calculated.
A particle outside the region generates an error.

*/
