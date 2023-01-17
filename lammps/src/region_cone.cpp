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

/* ----------------------------------------------------------------------
   Contributing author: Pim Schravendijk
------------------------------------------------------------------------- */

#include "region_cone.h"

#include "domain.h"
#include "error.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

RegCone::RegCone(LAMMPS *lmp, int narg, char **arg) :
  Region(lmp, narg, arg)
{
  options(narg-9,&arg[9]);

  // check open face settings

  if (openflag && (open_faces[3] || open_faces[4] || open_faces[5]))
    error->all(FLERR,"Invalid region cone open setting");

  if (strcmp(arg[2],"x") && strcmp(arg[2],"y") && strcmp(arg[2],"z"))
    error->all(FLERR,"Illegal region cylinder command");
  axis = arg[2][0];

  if (axis == 'x') {
    c1 = yscale*utils::numeric(FLERR,arg[3],false,lmp);
    c2 = zscale*utils::numeric(FLERR,arg[4],false,lmp);
    radiuslo = yscale*utils::numeric(FLERR,arg[5],false,lmp);
    radiushi = yscale*utils::numeric(FLERR,arg[6],false,lmp);
  } else if (axis == 'y') {
    c1 = xscale*utils::numeric(FLERR,arg[3],false,lmp);
    c2 = zscale*utils::numeric(FLERR,arg[4],false,lmp);
    radiuslo = xscale*utils::numeric(FLERR,arg[5],false,lmp);
    radiushi = xscale*utils::numeric(FLERR,arg[6],false,lmp);
  } else if (axis == 'z') {
    c1 = xscale*utils::numeric(FLERR,arg[3],false,lmp);
    c2 = yscale*utils::numeric(FLERR,arg[4],false,lmp);
    radiuslo = xscale*utils::numeric(FLERR,arg[5],false,lmp);
    radiushi = xscale*utils::numeric(FLERR,arg[6],false,lmp);
  }

  if (strcmp(arg[7],"INF") == 0 || strcmp(arg[7],"EDGE") == 0) {
    if (domain->box_exist == 0)
      error->all(FLERR,"Cannot use region INF or EDGE when box does not exist");
    if (axis == 'x') {
      if (strcmp(arg[7],"INF") == 0) lo = -BIG;
      else if (domain->triclinic == 0) lo = domain->boxlo[0];
      else lo = domain->boxlo_bound[0];
    }
    if (axis == 'y') {
      if (strcmp(arg[7],"INF") == 0) lo = -BIG;
      else if (domain->triclinic == 0) lo = domain->boxlo[1];
      else lo = domain->boxlo_bound[1];
    }
    if (axis == 'z') {
      if (strcmp(arg[7],"INF") == 0) lo = -BIG;
      else if (domain->triclinic == 0) lo = domain->boxlo[2];
      else lo = domain->boxlo_bound[2];
    }
  } else {
    if (axis == 'x') lo = xscale*utils::numeric(FLERR,arg[7],false,lmp);
    if (axis == 'y') lo = yscale*utils::numeric(FLERR,arg[7],false,lmp);
    if (axis == 'z') lo = zscale*utils::numeric(FLERR,arg[7],false,lmp);
  }

  if (strcmp(arg[8],"INF") == 0 || strcmp(arg[7],"EDGE") == 0) {
    if (domain->box_exist == 0)
      error->all(FLERR,"Cannot use region INF or EDGE when box does not exist");
    if (axis == 'x') {
      if (strcmp(arg[8],"INF") == 0) hi = BIG;
      else if (domain->triclinic == 0) hi = domain->boxhi[0];
      else hi = domain->boxhi_bound[0];
    }
    if (axis == 'y') {
      if (strcmp(arg[8],"INF") == 0) hi = BIG;
      if (domain->triclinic == 0) hi = domain->boxhi[1];
      else hi = domain->boxhi_bound[1];
    }
    if (axis == 'z') {
      if (strcmp(arg[8],"INF") == 0) hi = BIG;
      else if (domain->triclinic == 0) hi = domain->boxhi[2];
      else hi = domain->boxhi_bound[2];
    }
  } else {
    if (axis == 'x') hi = xscale*utils::numeric(FLERR,arg[8],false,lmp);
    if (axis == 'y') hi = yscale*utils::numeric(FLERR,arg[8],false,lmp);
    if (axis == 'z') hi = zscale*utils::numeric(FLERR,arg[8],false,lmp);
  }

  // error check

  if (radiuslo < 0.0) error->all(FLERR,"Illegal radius in region cone command");
  if (radiushi < 0.0) error->all(FLERR,"Illegal radius in region cone command");
  if (radiuslo == 0.0 && radiushi == 0.0)
    error->all(FLERR,"Illegal radius in region cone command");
  if (hi == lo) error->all(FLERR,"Illegal cone length in region cone command");

  // extent of cone

  if (radiuslo > radiushi) maxradius = radiuslo;
  else maxradius = radiushi;

  if (interior) {
    bboxflag = 1;
    if (axis == 'x') {
      extent_xlo = lo;
      extent_xhi = hi;
      extent_ylo = c1 - maxradius;
      extent_yhi = c1 + maxradius;
      extent_zlo = c2 - maxradius;
      extent_zhi = c2 + maxradius;
    }
    if (axis == 'y') {
      extent_xlo = c1 - maxradius;
      extent_xhi = c1 + maxradius;
      extent_ylo = lo;
      extent_yhi = hi;
      extent_zlo = c2 - maxradius;
      extent_zhi = c2 + maxradius;
    }
    if (axis == 'z') {
      extent_xlo = c1 - maxradius;
      extent_xhi = c1 + maxradius;
      extent_ylo = c2 - maxradius;
      extent_yhi = c2 + maxradius;
      extent_zlo = lo;
      extent_zhi = hi;
    }
  } else bboxflag = 0;

  // particle could be close to cone surface and 2 ends
  // particle can only touch surface and 1 end

  cmax = 3;
  contact = new Contact[cmax];
  if (interior) tmax = 2;
  else tmax = 1;
}

/* ---------------------------------------------------------------------- */

RegCone::~RegCone()
{
  delete [] contact;
}

/* ----------------------------------------------------------------------
   inside = 1 if x,y,z is inside or on surface
   inside = 0 if x,y,z is outside and not on surface
------------------------------------------------------------------------- */

int RegCone::inside(double x, double y, double z)
{
  double del1,del2,dist;
  double currentradius;

  if (axis == 'x') {
    del1 = y - c1;
    del2 = z - c2;
    dist = sqrt(del1*del1 + del2*del2);
    currentradius = radiuslo + (x-lo)*(radiushi-radiuslo)/(hi-lo);
    if (dist <= currentradius && x >= lo && x <= hi) return 1;
    else return 0;
  } else if (axis == 'y') {
    del1 = x - c1;
    del2 = z - c2;
    dist = sqrt(del1*del1 + del2*del2);
    currentradius = radiuslo + (y-lo)*(radiushi-radiuslo)/(hi-lo);
    if (dist <= currentradius && y >= lo && y <= hi) return 1;
    else return 0;
  } else if (axis == 'z') {
    del1 = x - c1;
    del2 = y - c2;
    dist = sqrt(del1*del1 + del2*del2);
    currentradius = radiuslo + (z-lo)*(radiushi-radiuslo)/(hi-lo);
    if (dist <= currentradius && z >= lo && z <= hi) return 1;
    else return 0;
  }

  return 0;
}

/* ----------------------------------------------------------------------
   contact if 0 <= x < cutoff from one or more inner surfaces of cone
   can be one contact for each of 3 cone surfaces
   no contact if outside (possible if called from union/intersect)
   delxyz = vector from nearest point on cone to x
   special case: no contact with curved surf if x is on center axis
------------------------------------------------------------------------- */

int RegCone::surface_interior(double *x, double cutoff)
{
  double del1,del2,r,currentradius,delx,dely,delz,dist,delta;
  double surflo[3],surfhi[3],xs[3];

  int n = 0;

  if (axis == 'x') {
    del1 = x[1] - c1;
    del2 = x[2] - c2;
    r = sqrt(del1*del1 + del2*del2);
    currentradius = radiuslo + (x[0]-lo)*(radiushi-radiuslo)/(hi-lo);

    // x is exterior to cone

    if (r > currentradius || x[0] < lo || x[0] > hi) return 0;

    // x is interior to cone or on its surface
    // surflo = pt on outer circle of bottom end plane, same dir as x vs axis
    // surfhi = pt on outer circle of top end plane, same dir as x vs axis

    if (r > 0.0 && !open_faces[2]) {
      surflo[0] = lo;
      surflo[1] = c1 + del1*radiuslo/r;
      surflo[2] = c2 + del2*radiuslo/r;
      surfhi[0] = hi;
      surfhi[1] = c1 + del1*radiushi/r;
      surfhi[2] = c2 + del2*radiushi/r;
      point_on_line_segment(surflo,surfhi,x,xs);
      delx = x[0] - xs[0];
      dely = x[1] - xs[1];
      delz = x[2] - xs[2];
      dist = sqrt(delx*delx + dely*dely + delz*delz);
      if (dist < cutoff) {
        contact[n].r = dist;
        contact[n].delx = delx;
        contact[n].dely = dely;
        contact[n].delz = delz;
        contact[n].radius = -2.0*(radiuslo + (xs[0]-lo)*
                                  (radiushi-radiuslo)/(hi-lo));
        contact[n].iwall = 2;
        n++;
      }
    }

    delta = x[0] - lo;
    if (delta < cutoff && !open_faces[0]) {
      contact[n].r = delta;
      contact[n].delx = delta;
      contact[n].dely = contact[n].delz = 0.0;
      contact[n].radius = 0;
      contact[n].iwall = 0;
      n++;
    }
    delta = hi - x[0];
    if (delta < cutoff && !open_faces[1]) {
      contact[n].r = delta;
      contact[n].delx = -delta;
      contact[n].dely = contact[n].delz = 0.0;
      contact[n].radius = 0;
      contact[n].iwall = 1;
      n++;
    }

  } else if (axis == 'y') {
    del1 = x[0] - c1;
    del2 = x[2] - c2;
    r = sqrt(del1*del1 + del2*del2);
    currentradius = radiuslo + (x[1]-lo)*
      (radiushi-radiuslo)/(hi-lo);

    // y is exterior to cone

    if (r > currentradius || x[1] < lo || x[1] > hi) return 0;

    // y is interior to cone or on its surface
    // surflo = pt on outer circle of bottom end plane, same dir as y vs axis
    // surfhi = pt on outer circle of top end plane, same dir as y vs axis

    if (r > 0.0 && !open_faces[2]) {
      surflo[0] = c1 + del1*radiuslo/r;
      surflo[1] = lo;
      surflo[2] = c2 + del2*radiuslo/r;
      surfhi[0] = c1 + del1*radiushi/r;
      surfhi[1] = hi;
      surfhi[2] = c2 + del2*radiushi/r;
      point_on_line_segment(surflo,surfhi,x,xs);
      delx = x[0] - xs[0];
      dely = x[1] - xs[1];
      delz = x[2] - xs[2];
      dist = sqrt(delx*delx + dely*dely + delz*delz);
      if (dist < cutoff) {
        contact[n].r = dist;
        contact[n].delx = delx;
        contact[n].dely = dely;
        contact[n].delz = delz;
        contact[n].iwall = 2;
        contact[n].radius = -2.0*(radiuslo + (xs[1]-lo)*
                                  (radiushi-radiuslo)/(hi-lo));
        n++;
      }
    }

    delta = x[1] - lo;
    if (delta < cutoff && !open_faces[0]) {
      contact[n].r = delta;
      contact[n].delz = delta;
      contact[n].delx = contact[n].dely = 0.0;
      contact[n].iwall = 0;
      contact[n].radius = 0;
      n++;
    }
    delta = hi - x[1];
    if (delta < cutoff && !open_faces[1]) {
      contact[n].r = delta;
      contact[n].delz = -delta;
      contact[n].delx = contact[n].dely = 0.0;
      contact[n].iwall = 1;
      contact[n].radius = 0;
      n++;
    }

  } else {
    del1 = x[0] - c1;
    del2 = x[1] - c2;
    r = sqrt(del1*del1 + del2*del2);
    currentradius = radiuslo + (x[2]-lo)*(radiushi-radiuslo)/(hi-lo);

    // z is exterior to cone

    if (r > currentradius || x[2] < lo || x[2] > hi) return 0;

    // z is interior to cone or on its surface
    // surflo = pt on outer circle of bottom end plane, same dir as z vs axis
    // surfhi = pt on outer circle of top end plane, same dir as z vs axis

    if (r > 0.0 && !open_faces[2]) {
      surflo[0] = c1 + del1*radiuslo/r;
      surflo[1] = c2 + del2*radiuslo/r;
      surflo[2] = lo;
      surfhi[0] = c1 + del1*radiushi/r;
      surfhi[1] = c2 + del2*radiushi/r;
      surfhi[2] = hi;
      point_on_line_segment(surflo,surfhi,x,xs);
      delx = x[0] - xs[0];
      dely = x[1] - xs[1];
      delz = x[2] - xs[2];
      dist = sqrt(delx*delx + dely*dely + delz*delz);
      if (dist < cutoff) {
        contact[n].r = dist;
        contact[n].delx = delx;
        contact[n].dely = dely;
        contact[n].delz = delz;
        contact[n].iwall = 2;
        contact[n].radius = -2.0*(radiuslo + (xs[2]-lo)*
                                  (radiushi-radiuslo)/(hi-lo));
        n++;
      }
    }

    delta = x[2] - lo;
    if (delta < cutoff  && !open_faces[0]) {
      contact[n].r = delta;
      contact[n].delz = delta;
      contact[n].delx = contact[n].dely = 0.0;
      contact[n].iwall = 0;
      contact[n].radius = 0;
      n++;
    }
    delta = hi - x[2];
    if (delta < cutoff  && !open_faces[1]) {
      contact[n].r = delta;
      contact[n].delz = -delta;
      contact[n].delx = contact[n].dely = 0.0;
      contact[n].iwall = 1;
      contact[n].radius = 0;
      n++;
    }
  }

  return n;
}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from outer surface of cone
   no contact if inside (possible if called from union/intersect)
   delxyz = vector from nearest point on cone to x
------------------------------------------------------------------------- */

int RegCone::surface_exterior(double *x, double cutoff)
{
  double del1,del2,r,currentradius,distsq,distsqprev,crad;
  double corner1[3],corner2[3],corner3[3],corner4[3],xp[3],nearest[3];

  if (axis == 'x') {
    del1 = x[1] - c1;
    del2 = x[2] - c2;
    r = sqrt(del1*del1 + del2*del2);
    currentradius = radiuslo + (x[0]-lo)*(radiushi-radiuslo)/(hi-lo);

    // radius of curvature, only used for granular walls

    crad = 0.0;

    // x is far enough from cone that there is no contact
    // x is interior to cone

    if (r >= maxradius+cutoff || x[0] <= lo-cutoff || x[0] >= hi+cutoff)
      return 0;
    if (r < currentradius && x[0] > lo && x[0] < hi) return 0;

    // x is exterior to cone or on its surface
    // corner1234 = 4 corner pts of half trapezoid = cone surf in plane of x
    // project x to 3 line segments in half trapezoid (4th is axis of cone)
    // nearest = point on surface of cone that x is closest to
    //           could be edge of cone
    // do not add contact point if r >= cutoff

    corner1[0] = lo;
    corner1[1] = c1 + del1*radiuslo/r;
    corner1[2] = c2 + del2*radiuslo/r;
    corner2[0] = hi;
    corner2[1] = c1 + del1*radiushi/r;
    corner2[2] = c2 + del2*radiushi/r;
    corner3[0] = lo;
    corner3[1] = c1;
    corner3[2] = c2;
    corner4[0] = hi;
    corner4[1] = c1;
    corner4[2] = c2;

    distsq = BIG;

    if (!open_faces[2]) {
      point_on_line_segment(corner1,corner2,x,xp);
      distsq = closest(x,xp,nearest,distsq);
      crad = -2.0*(radiuslo + (nearest[0]-lo)*(radiushi-radiuslo)/(hi-lo));
    }

    if (!open_faces[0]) {
      point_on_line_segment(corner1,corner3,x,xp);
      distsqprev = distsq;
      distsq = closest(x,xp,nearest,distsq);
      if (distsq < distsqprev) crad = 0.0;
    }

    if (!open_faces[1]) {
      point_on_line_segment(corner2,corner4,x,xp);
      distsqprev = distsq;
      distsq = closest(x,xp,nearest,distsq);
      if (distsq < distsqprev) crad = 0.0;
    }

    if (distsq == BIG) return 0;
    add_contact(0,x,nearest[0],nearest[1],nearest[2]);
    contact[0].radius = crad;
    contact[0].iwall = 0;
    if (contact[0].r < cutoff) return 1;
    return 0;

  } else if (axis == 'y') {
    del1 = x[0] - c1;
    del2 = x[2] - c2;
    r = sqrt(del1*del1 + del2*del2);
    currentradius = radiuslo + (x[1]-lo)*(radiushi-radiuslo)/(hi-lo);

    // radius of curvature, only used for granular walls

    crad = 0.0;

    // y is far enough from cone that there is no contact
    // y is interior to cone

    if (r >= maxradius+cutoff ||
        x[1] <= lo-cutoff || x[1] >= hi+cutoff) return 0;
    if (r < currentradius && x[1] > lo && x[1] < hi) return 0;

    // y is exterior to cone or on its surface
    // corner1234 = 4 corner pts of half trapezoid = cone surf in plane of y
    // project x to 3 line segments in half trapezoid (4th is axis of cone)
    // nearest = point on surface of cone that y is closest to
    //           could be edge of cone
    // do not add contact point if r >= cutoff

    corner1[0] = c1 + del1*radiuslo/r;
    corner1[1] = lo;
    corner1[2] = c2 + del2*radiuslo/r;
    corner2[0] = c1 + del1*radiushi/r;
    corner2[1] = hi;
    corner2[2] = c2 + del2*radiushi/r;
    corner3[0] = c1;
    corner3[1] = lo;
    corner3[2] = c2;
    corner4[0] = c1;
    corner4[1] = hi;
    corner4[2] = c2;

    distsq = BIG;

    if (!open_faces[2]) {
      point_on_line_segment(corner1,corner2,x,xp);
      distsq = closest(x,xp,nearest,distsq);
      crad = -2.0*(radiuslo + (nearest[1]-lo)*(radiushi-radiuslo)/(hi-lo));
    }

    if (!open_faces[0]) {
      point_on_line_segment(corner1,corner3,x,xp);
      distsqprev = distsq;
      distsq = closest(x,xp,nearest,distsq);
      if (distsq < distsqprev) crad = 0;
    }

    if (!open_faces[1]) {
      point_on_line_segment(corner2,corner4,x,xp);
      distsqprev = distsq;
      distsq = closest(x,xp,nearest,distsq);
      if (distsq < distsqprev) crad = 0;
    }

    add_contact(0,x,nearest[0],nearest[1],nearest[2]);
    contact[0].radius = crad;
    contact[0].iwall = 0;
    if (contact[0].r < cutoff) return 1;
    return 0;

  } else {
    del1 = x[0] - c1;
    del2 = x[1] - c2;
    r = sqrt(del1*del1 + del2*del2);
    currentradius = radiuslo + (x[2]-lo)*(radiushi-radiuslo)/(hi-lo);

    // radius of curvature, only used for granular walls

    crad = 0.0;

    // z is far enough from cone that there is no contact
    // z is interior to cone

    if (r >= maxradius+cutoff || x[2] <= lo-cutoff || x[2] >= hi+cutoff)
      return 0;
    if (r < currentradius && x[2] > lo && x[2] < hi) return 0;

    // z is exterior to cone or on its surface
    // corner1234 = 4 corner pts of half trapezoid = cone surf in plane of z
    // project x to 3 line segments in half trapezoid (4th is axis of cone)
    // nearest = point on surface of cone that z is closest to
    //           could be edge of cone
    // do not add contact point if r >= cutoff

    corner1[0] = c1 + del1*radiuslo/r;
    corner1[1] = c2 + del2*radiuslo/r;
    corner1[2] = lo;
    corner2[0] = c1 + del1*radiushi/r;
    corner2[1] = c2 + del2*radiushi/r;
    corner2[2] = hi;
    corner3[0] = c1;
    corner3[1] = c2;
    corner3[2] = lo;
    corner4[0] = c1;
    corner4[1] = c2;
    corner4[2] = hi;

    distsq = BIG;

    if (!open_faces[2]) {
      point_on_line_segment(corner1,corner2,x,xp);
      distsq = closest(x,xp,nearest,distsq);
      crad = -2.0*(radiuslo + (nearest[2]-lo)*(radiushi-radiuslo)/(hi-lo));
    }

    if (!open_faces[0]) {
      point_on_line_segment(corner1,corner3,x,xp);
      distsqprev = distsq;
      distsq = closest(x,xp,nearest,distsq);
      if (distsq < distsqprev) crad = 0;
    }

    if (!open_faces[1]) {
      point_on_line_segment(corner2,corner4,x,xp);
      distsqprev = distsq;
      distsq = closest(x,xp,nearest,distsq);
      if (distsq < distsqprev) crad = 0;
    }

    add_contact(0,x,nearest[0],nearest[1],nearest[2]);
    contact[0].radius = crad;
    contact[0].iwall = 0;
    if (contact[0].r < cutoff) return 1;
    return 0;
  }
}

/* ---------------------------------------------------------------------- */

double RegCone::closest(double *x, double *near, double *nearest, double dsq)
{
  double delx = x[0] - near[0];
  double dely = x[1] - near[1];
  double delz = x[2] - near[2];
  double rsq = delx*delx + dely*dely + delz*delz;
  if (rsq >= dsq) return dsq;

  nearest[0] = near[0];
  nearest[1] = near[1];
  nearest[2] = near[2];
  return rsq;
}
