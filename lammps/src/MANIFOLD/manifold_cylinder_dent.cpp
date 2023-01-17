// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   ----------------------------------------------------------------------- */

#include "manifold_cylinder_dent.h"
#include "math_const.h"

#include <cmath>


using namespace LAMMPS_NS;

using namespace user_manifold;

manifold_cylinder_dent::manifold_cylinder_dent( LAMMPS *lmp, int /*argc*/,
                                                char **/*argv*/ ) : manifold(lmp)
{}


double manifold_cylinder_dent::g( const double *x )
{
  double l = params[1], R = params[0], a = params[2];
  double r2 = x[1]*x[1] + x[0]*x[0];
  if (fabs(x[2]) < 0.5*l) {
    double k = MathConst::MY_2PI / l;
    double c = R - 0.5*a*( 1.0 + cos(k*x[2]) );
    return c*c - r2;
  } else {
    return R*R - r2;
  }
}


void manifold_cylinder_dent::n( const double *x, double *n )
{
  double l = params[1], R = params[0], a = params[2];
  if (fabs(x[2]) < 0.5*l) {
    double k = MathConst::MY_2PI / l;
    double c = R - 0.5*a*(1.0 + cos(k*x[2]));
    n[0] = -2*x[0];
    n[1] = -2*x[1];
    n[2] = c*a*k*sin(k*x[2]);
  } else {
    n[0] = -2*x[0];
    n[1] = -2*x[1];
    n[2] = 0.0;
  }
}
