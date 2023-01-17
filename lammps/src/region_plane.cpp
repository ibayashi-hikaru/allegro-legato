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
------------------------------------------------------------------------- */

#include "region_plane.h"

#include "error.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

RegPlane::RegPlane(LAMMPS *lmp, int narg, char **arg) :
  Region(lmp, narg, arg)
{
  options(narg-8,&arg[8]);

  xp = xscale*utils::numeric(FLERR,arg[2],false,lmp);
  yp = yscale*utils::numeric(FLERR,arg[3],false,lmp);
  zp = zscale*utils::numeric(FLERR,arg[4],false,lmp);
  normal[0] = xscale*utils::numeric(FLERR,arg[5],false,lmp);
  normal[1] = yscale*utils::numeric(FLERR,arg[6],false,lmp);
  normal[2] = zscale*utils::numeric(FLERR,arg[7],false,lmp);

  // enforce unit normal

  double rsq = normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2];
  if (rsq == 0.0) error->all(FLERR,"Illegal region plane command");
  normal[0] /= sqrt(rsq);
  normal[1] /= sqrt(rsq);
  normal[2] /= sqrt(rsq);

  // plane has no bounding box

  bboxflag = 0;
  cmax = 1;
  contact = new Contact[cmax];
  tmax = 1;
}

/* ---------------------------------------------------------------------- */

RegPlane::~RegPlane()
{
  delete [] contact;
}

/* ----------------------------------------------------------------------
   inside = 1 if x,y,z is on normal side of plane or on plane
   inside = 0 if x,y,z is on non-normal side of plane and not on plane
   x,y,z is inside if (x-xp) dot normal >= 0
------------------------------------------------------------------------- */

int RegPlane::inside(double x, double y, double z)
{
  double dot = (x-xp)*normal[0] + (y-yp)*normal[1] + (z-zp)*normal[2];

  if (dot >= 0.0) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from normal side of plane
   no contact if on other side (possible if called from union/intersect)
   delxyz = vector from nearest projected point on plane to x
------------------------------------------------------------------------- */

int RegPlane::surface_interior(double *x, double cutoff)
{
  double dot = (x[0]-xp)*normal[0] + (x[1]-yp)*normal[1] + (x[2]-zp)*normal[2];
  if (dot < cutoff && dot >= 0.0) {
    contact[0].r = dot;
    contact[0].delx = dot*normal[0];
    contact[0].dely = dot*normal[1];
    contact[0].delz = dot*normal[2];
    contact[0].radius = 0;
    contact[0].iwall = 0;
    return 1;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from non-normal side of plane
   no contact if on other side (possible if called from union/intersect)
   delxyz = vector from nearest projected point on plane to x
------------------------------------------------------------------------- */

int RegPlane::surface_exterior(double *x, double cutoff)
{
  double dot = (x[0]-xp)*normal[0] + (x[1]-yp)*normal[1] + (x[2]-zp)*normal[2];
  dot = -dot;
  if (dot < cutoff && dot >= 0.0) {
    contact[0].r = dot;
    contact[0].delx = -dot*normal[0];
    contact[0].dely = -dot*normal[1];
    contact[0].delz = -dot*normal[2];
    contact[0].radius = 0;
    contact[0].iwall = 0;
    return 1;
  }
  return 0;
}
