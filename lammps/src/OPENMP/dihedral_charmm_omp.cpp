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

#include "dihedral_charmm_omp.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "pair.h"

#include <cmath>

#include "omp_compat.h"
#include "suffix.h"
using namespace LAMMPS_NS;

#define TOLERANCE 0.05
#define SMALL     0.001

/* ---------------------------------------------------------------------- */

DihedralCharmmOMP::DihedralCharmmOMP(class LAMMPS *lmp)
  : DihedralCharmm(lmp), ThrOMP(lmp,THR_DIHEDRAL|THR_CHARMM)
{
  suffix_flag |= Suffix::OMP;
}

/* ---------------------------------------------------------------------- */

void DihedralCharmmOMP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  // insure pair->ev_tally() will use 1-4 virial contribution

  if (weightflag && vflag_global == VIRIAL_FDOTR)
    force->pair->vflag_either = force->pair->vflag_global = 1;

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = neighbor->ndihedrallist;

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
void DihedralCharmmOMP::eval(int nfrom, int nto, ThrData * const thr)
{

  int i1,i2,i3,i4,i,m,n,type;
  double vb1x,vb1y,vb1z,vb2x,vb2y,vb2z,vb3x,vb3y,vb3z,vb2xm,vb2ym,vb2zm;
  double edihedral,f1[3],f2[3],f3[3],f4[3];
  double ax,ay,az,bx,by,bz,rasq,rbsq,rgsq,rg,rginv,ra2inv,rb2inv,rabinv;
  double df,df1,ddf1,fg,hg,fga,hgb,gaa,gbb;
  double dtfx,dtfy,dtfz,dtgx,dtgy,dtgz,dthx,dthy,dthz;
  double c,s,p,sx2,sy2,sz2;
  int itype,jtype;
  double delx,dely,delz,rsq,r2inv,r6inv;
  double forcecoul,forcelj,fpair,ecoul,evdwl;

  ecoul = evdwl = edihedral = 0.0;

  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  dbl3_t * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const double * _noalias const q = atom->q;
  const int * const atomtype = atom->type;
  const int5_t * _noalias const dihedrallist = (int5_t *) neighbor->dihedrallist[0];
  const double qqrd2e = force->qqrd2e;
  const int nlocal = atom->nlocal;

  for (n = nfrom; n < nto; n++) {
    i1 = dihedrallist[n].a;
    i2 = dihedrallist[n].b;
    i3 = dihedrallist[n].c;
    i4 = dihedrallist[n].d;
    type = dihedrallist[n].t;

    // 1st bond

    vb1x = x[i1].x - x[i2].x;
    vb1y = x[i1].y - x[i2].y;
    vb1z = x[i1].z - x[i2].z;

    // 2nd bond

    vb2x = x[i3].x - x[i2].x;
    vb2y = x[i3].y - x[i2].y;
    vb2z = x[i3].z - x[i2].z;

    vb2xm = -vb2x;
    vb2ym = -vb2y;
    vb2zm = -vb2z;

    // 3rd bond

    vb3x = x[i4].x - x[i3].x;
    vb3y = x[i4].y - x[i3].y;
    vb3z = x[i4].z - x[i3].z;

    // c,s calculation

    ax = vb1y*vb2zm - vb1z*vb2ym;
    ay = vb1z*vb2xm - vb1x*vb2zm;
    az = vb1x*vb2ym - vb1y*vb2xm;
    bx = vb3y*vb2zm - vb3z*vb2ym;
    by = vb3z*vb2xm - vb3x*vb2zm;
    bz = vb3x*vb2ym - vb3y*vb2xm;

    rasq = ax*ax + ay*ay + az*az;
    rbsq = bx*bx + by*by + bz*bz;
    rgsq = vb2xm*vb2xm + vb2ym*vb2ym + vb2zm*vb2zm;
    rg = sqrt(rgsq);

    rginv = ra2inv = rb2inv = 0.0;
    if (rg > 0) rginv = 1.0/rg;
    if (rasq > 0) ra2inv = 1.0/rasq;
    if (rbsq > 0) rb2inv = 1.0/rbsq;
    rabinv = sqrt(ra2inv*rb2inv);

    c = (ax*bx + ay*by + az*bz)*rabinv;
    s = rg*rabinv*(ax*vb3x + ay*vb3y + az*vb3z);

    // error check

    if (c > 1.0 + TOLERANCE || c < (-1.0 - TOLERANCE))
      problem(FLERR, i1, i2, i3, i4);

    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    m = multiplicity[type];
    p = 1.0;
    ddf1 = df1 = 0.0;

    for (i = 0; i < m; i++) {
      ddf1 = p*c - df1*s;
      df1 = p*s + df1*c;
      p = ddf1;
    }

    p = p*cos_shift[type] + df1*sin_shift[type];
    df1 = df1*cos_shift[type] - ddf1*sin_shift[type];
    df1 *= -m;
    p += 1.0;

    if (m == 0) {
      p = 1.0 + cos_shift[type];
      df1 = 0.0;
    }

    if (EFLAG) edihedral = k[type] * p;

    fg = vb1x*vb2xm + vb1y*vb2ym + vb1z*vb2zm;
    hg = vb3x*vb2xm + vb3y*vb2ym + vb3z*vb2zm;
    fga = fg*ra2inv*rginv;
    hgb = hg*rb2inv*rginv;
    gaa = -ra2inv*rg;
    gbb = rb2inv*rg;

    dtfx = gaa*ax;
    dtfy = gaa*ay;
    dtfz = gaa*az;
    dtgx = fga*ax - hgb*bx;
    dtgy = fga*ay - hgb*by;
    dtgz = fga*az - hgb*bz;
    dthx = gbb*bx;
    dthy = gbb*by;
    dthz = gbb*bz;

    df = -k[type] * df1;

    sx2 = df*dtgx;
    sy2 = df*dtgy;
    sz2 = df*dtgz;

    f1[0] = df*dtfx;
    f1[1] = df*dtfy;
    f1[2] = df*dtfz;

    f2[0] = sx2 - f1[0];
    f2[1] = sy2 - f1[1];
    f2[2] = sz2 - f1[2];

    f4[0] = df*dthx;
    f4[1] = df*dthy;
    f4[2] = df*dthz;

    f3[0] = -sx2 - f4[0];
    f3[1] = -sy2 - f4[1];
    f3[2] = -sz2 - f4[2];

    // apply force to each of 4 atoms

    if (NEWTON_BOND || i1 < nlocal) {
      f[i1].x += f1[0];
      f[i1].y += f1[1];
      f[i1].z += f1[2];
    }

    if (NEWTON_BOND || i2 < nlocal) {
      f[i2].x += f2[0];
      f[i2].y += f2[1];
      f[i2].z += f2[2];
    }

    if (NEWTON_BOND || i3 < nlocal) {
      f[i3].x += f3[0];
      f[i3].y += f3[1];
      f[i3].z += f3[2];
    }

    if (NEWTON_BOND || i4 < nlocal) {
      f[i4].x += f4[0];
      f[i4].y += f4[1];
      f[i4].z += f4[2];
    }

    if (EVFLAG)
      ev_tally_thr(this,i1,i2,i3,i4,nlocal,NEWTON_BOND,edihedral,f1,f3,f4,
                   vb1x,vb1y,vb1z,vb2x,vb2y,vb2z,vb3x,vb3y,vb3z,thr);
    // 1-4 LJ and Coulomb interactions
    // tally energy/virial in pair, using newton_bond as newton flag

    if (weight[type] > 0.0) {
      itype = atomtype[i1];
      jtype = atomtype[i4];

      delx = x[i1].x - x[i4].x;
      dely = x[i1].y - x[i4].y;
      delz = x[i1].z - x[i4].z;
      rsq = delx*delx + dely*dely + delz*delz;
      r2inv = 1.0/rsq;
      r6inv = r2inv*r2inv*r2inv;

      if (implicit) forcecoul = qqrd2e * q[i1]*q[i4]*r2inv;
      else forcecoul = qqrd2e * q[i1]*q[i4]*sqrt(r2inv);
      forcelj = r6inv * (lj14_1[itype][jtype]*r6inv - lj14_2[itype][jtype]);
      fpair = weight[type] * (forcelj+forcecoul)*r2inv;

      if (EFLAG) {
        ecoul = weight[type] * forcecoul;
        evdwl = r6inv * (lj14_3[itype][jtype]*r6inv - lj14_4[itype][jtype]);
        evdwl *= weight[type];
      }

      if (NEWTON_BOND || i1 < nlocal) {
        f[i1].x += delx*fpair;
        f[i1].y += dely*fpair;
        f[i1].z += delz*fpair;
      }
      if (NEWTON_BOND || i4 < nlocal) {
        f[i4].x -= delx*fpair;
        f[i4].y -= dely*fpair;
        f[i4].z -= delz*fpair;
      }

      if (EVFLAG) ev_tally_thr(force->pair,i1,i4,nlocal,NEWTON_BOND,
                               evdwl,ecoul,fpair,delx,dely,delz,thr);
    }
  }
}
