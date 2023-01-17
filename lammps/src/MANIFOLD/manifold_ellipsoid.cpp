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

#include "manifold_ellipsoid.h"

using namespace LAMMPS_NS;

using namespace user_manifold;

manifold_ellipsoid::manifold_ellipsoid( LAMMPS *lmp, int /*narg*/, char **/*argv*/ ) : manifold(lmp)
{}


double manifold_ellipsoid::g( const double *x )
{
  const double ai = 1.0 / params[0];
  const double bi = 1.0 / params[1];
  const double ci = 1.0 / params[2];
  return x[0]*x[0]*ai*ai + x[1]*x[1]*bi*bi + x[2]*x[2]*ci*ci - 1.0;
}

void manifold_ellipsoid::n( const double *x, double * n )
{
  const double ai = 1.0 / params[0];
  const double bi = 1.0 / params[1];
  const double ci = 1.0 / params[2];
  n[0] = 2*x[0]*ai*ai;
  n[1] = 2*x[1]*bi*bi;
  n[2] = 2*x[2]*ci*ci;
}

