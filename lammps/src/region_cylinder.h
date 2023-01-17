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

#ifdef REGION_CLASS
// clang-format off
RegionStyle(cylinder,RegCylinder);
// clang-format on
#else

#ifndef LMP_REGION_CYLINDER_H
#define LMP_REGION_CYLINDER_H

#include "region.h"

namespace LAMMPS_NS {

class RegCylinder : public Region {
  friend class FixPour;

 public:
  RegCylinder(class LAMMPS *, int, char **);
  ~RegCylinder();
  void init();
  int inside(double, double, double);
  int surface_interior(double *, double);
  int surface_exterior(double *, double);
  void shape_update();
  void set_velocity_shape();
  void velocity_contact_shape(double *, double *);

 private:
  char axis;
  double c1, c2;
  double radius;
  double lo, hi;
  int c1style, c1var;
  int c2style, c2var;
  int rstyle, rvar;
  char *c1str, *c2str, *rstr;

  void variable_check();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Invalid region cylinder open setting

UNDOCUMENTED

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Cannot use region INF or EDGE when box does not exist

Regions that extend to the box boundaries can only be used after the
create_box command has been used.

E: Variable evaluation in region gave bad value

Variable returned a radius < 0.0.

E: Variable name for region cylinder does not exist

Self-explanatory.

E: Variable for region cylinder is invalid style

Only equal-style variables are allowed.

*/
