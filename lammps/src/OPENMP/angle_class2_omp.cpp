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
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "omp_compat.h"
#include "angle_class2_omp.h"
#include <cmath>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"


#include "suffix.h"
using namespace LAMMPS_NS;

#define SMALL 0.001

/* ---------------------------------------------------------------------- */

AngleClass2OMP::AngleClass2OMP(class LAMMPS *lmp)
  : AngleClass2(lmp), ThrOMP(lmp,THR_ANGLE)
{
  suffix_flag |= Suffix::OMP;
}

/* ---------------------------------------------------------------------- */

void AngleClass2OMP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = neighbor->nanglelist;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, cvatom, thr);

    if (inum > 0) {
      if (evflag) {
        if (eflag) {
          if (force->newton_bond) eval<1,1,1>(ifrom, ito, thr);
          else eval<1,1,0>(ifrom, ito, thr);
        } else {
          if (force->newton_bond) eval<1,0,1>(ifrom, ito, thr);
          else eval<1,0,0>(ifrom, ito, thr);
        }
      } else {
        if (force->newton_bond) eval<0,0,1>(ifrom, ito, thr);
        else eval<0,0,0>(ifrom, ito, thr);
      }
    }
    thr->timer(Timer::BOND);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

template <int EVFLAG, int EFLAG, int NEWTON_BOND>
void AngleClass2OMP::eval(int nfrom, int nto, ThrData * const thr)
{
  int i1,i2,i3,n,type;
  double delx1,dely1,delz1,delx2,dely2,delz2;
  double eangle,f1[3],f3[3];
  double dtheta,dtheta2,dtheta3,dtheta4,de_angle;
  double dr1,dr2,tk1,tk2,aa1,aa2,aa11,aa12,aa21,aa22;
  double rsq1,rsq2,r1,r2,c,s,a,a11,a12,a22,b1,b2;
  double vx11,vx12,vy11,vy12,vz11,vz12,vx21,vx22,vy21,vy22,vz21,vz22;

  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  dbl3_t * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const int4_t * _noalias const anglelist = (int4_t *) neighbor->anglelist[0];
  const int nlocal = atom->nlocal;
  eangle = 0.0;

  for (n = nfrom; n < nto; n++) {
    i1 = anglelist[n].a;
    i2 = anglelist[n].b;
    i3 = anglelist[n].c;
    type = anglelist[n].t;

    // 1st bond

    delx1 = x[i1].x - x[i2].x;
    dely1 = x[i1].y - x[i2].y;
    delz1 = x[i1].z - x[i2].z;

    rsq1 = delx1*delx1 + dely1*dely1 + delz1*delz1;
    r1 = sqrt(rsq1);

    // 2nd bond

    delx2 = x[i3].x - x[i2].x;
    dely2 = x[i3].y - x[i2].y;
    delz2 = x[i3].z - x[i2].z;

    rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
    r2 = sqrt(rsq2);

    // angle (cos and sin)

    c = delx1*delx2 + dely1*dely2 + delz1*delz2;
    c /= r1*r2;

    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    s = sqrt(1.0 - c*c);
    if (s < SMALL) s = SMALL;
    s = 1.0/s;

    // force & energy for angle term

    dtheta = acos(c) - theta0[type];
    dtheta2 = dtheta*dtheta;
    dtheta3 = dtheta2*dtheta;
    dtheta4 = dtheta3*dtheta;

    de_angle = 2.0*k2[type]*dtheta + 3.0*k3[type]*dtheta2 +
      4.0*k4[type]*dtheta3;

    a = -de_angle*s;
    a11 = a*c / rsq1;
    a12 = -a / (r1*r2);
    a22 = a*c / rsq2;

    f1[0] = a11*delx1 + a12*delx2;
    f1[1] = a11*dely1 + a12*dely2;
    f1[2] = a11*delz1 + a12*delz2;

    f3[0] = a22*delx2 + a12*delx1;
    f3[1] = a22*dely2 + a12*dely1;
    f3[2] = a22*delz2 + a12*delz1;

    if (EFLAG) eangle = k2[type]*dtheta2 + k3[type]*dtheta3 + k4[type]*dtheta4;

    // force & energy for bond-bond term

    dr1 = r1 - bb_r1[type];
    dr2 = r2 - bb_r2[type];
    tk1 = bb_k[type] * dr1;
    tk2 = bb_k[type] * dr2;

    f1[0] -= delx1*tk2/r1;
    f1[1] -= dely1*tk2/r1;
    f1[2] -= delz1*tk2/r1;

    f3[0] -= delx2*tk1/r2;
    f3[1] -= dely2*tk1/r2;
    f3[2] -= delz2*tk1/r2;

    if (EFLAG) eangle += bb_k[type]*dr1*dr2;

    // force & energy for bond-angle term

    aa1 = s * dr1 * ba_k1[type];
    aa2 = s * dr2 * ba_k2[type];

    aa11 = aa1 * c / rsq1;
    aa12 = -aa1 / (r1 * r2);
    aa21 = aa2 * c / rsq1;
    aa22 = -aa2 / (r1 * r2);

    vx11 = (aa11 * delx1) + (aa12 * delx2);
    vx12 = (aa21 * delx1) + (aa22 * delx2);
    vy11 = (aa11 * dely1) + (aa12 * dely2);
    vy12 = (aa21 * dely1) + (aa22 * dely2);
    vz11 = (aa11 * delz1) + (aa12 * delz2);
    vz12 = (aa21 * delz1) + (aa22 * delz2);

    aa11 = aa1 * c / rsq2;
    aa21 = aa2 * c / rsq2;

    vx21 = (aa11 * delx2) + (aa12 * delx1);
    vx22 = (aa21 * delx2) + (aa22 * delx1);
    vy21 = (aa11 * dely2) + (aa12 * dely1);
    vy22 = (aa21 * dely2) + (aa22 * dely1);
    vz21 = (aa11 * delz2) + (aa12 * delz1);
    vz22 = (aa21 * delz2) + (aa22 * delz1);

    b1 = ba_k1[type] * dtheta / r1;
    b2 = ba_k2[type] * dtheta / r2;

    f1[0] -= vx11 + b1*delx1 + vx12;
    f1[1] -= vy11 + b1*dely1 + vy12;
    f1[2] -= vz11 + b1*delz1 + vz12;

    f3[0] -= vx21 + b2*delx2 + vx22;
    f3[1] -= vy21 + b2*dely2 + vy22;
    f3[2] -= vz21 + b2*delz2 + vz22;

    if (EFLAG) eangle += ba_k1[type]*dr1*dtheta + ba_k2[type]*dr2*dtheta;

    // apply force to each of 3 atoms

    if (NEWTON_BOND || i1 < nlocal) {
      f[i1].x += f1[0];
      f[i1].y += f1[1];
      f[i1].z += f1[2];
    }

    if (NEWTON_BOND || i2 < nlocal) {
      f[i2].x -= f1[0] + f3[0];
      f[i2].y -= f1[1] + f3[1];
      f[i2].z -= f1[2] + f3[2];
    }

    if (NEWTON_BOND || i3 < nlocal) {
      f[i3].x += f3[0];
      f[i3].y += f3[1];
      f[i3].z += f3[2];
    }

    if (EVFLAG) ev_tally_thr(this,i1,i2,i3,nlocal,NEWTON_BOND,eangle,f1,f3,
                             delx1,dely1,delz1,delx2,dely2,delz2,thr);
  }
}
