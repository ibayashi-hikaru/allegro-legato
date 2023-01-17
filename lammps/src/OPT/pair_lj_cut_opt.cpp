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
   Contributing authors:
     James Fischer, High Performance Technologies, Inc.
     David Richie, Stone Ridge Technology
     Vincent Natoli, Stone Ridge Technology
------------------------------------------------------------------------- */

#include "pair_lj_cut_opt.h"

#include "atom.h"
#include "force.h"
#include "neigh_list.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJCutOpt::PairLJCutOpt(LAMMPS *lmp) : PairLJCut(lmp) {}

/* ---------------------------------------------------------------------- */

void PairLJCutOpt::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  if (evflag) {
    if (eflag) {
      if (force->newton_pair) return eval<1,1,1>();
      else return eval<1,1,0>();
    } else {
      if (force->newton_pair) return eval<1,0,1>();
      else return eval<1,0,0>();
    }
  } else {
    if (force->newton_pair) return eval<0,0,1>();
    else return eval<0,0,0>();
  }
}

/* ---------------------------------------------------------------------- */

template < int EVFLAG, int EFLAG, int NEWTON_PAIR >
void PairLJCutOpt::eval()
{
  typedef struct { double x,y,z; } vec3_t;

  typedef struct {
    double cutsq,lj1,lj2,lj3,lj4,offset;
    double _pad[2];
  } fast_alpha_t;

  int i,j,ii,jj,inum,jnum,itype,jtype,sbindex;
  double factor_lj;
  double evdwl = 0.0;

  double** _noalias x = atom->x;
  double** _noalias f = atom->f;
  int* _noalias type = atom->type;
  int nlocal = atom->nlocal;
  double* _noalias special_lj = force->special_lj;

  inum = list->inum;
  int* _noalias ilist = list->ilist;
  int** _noalias firstneigh = list->firstneigh;
  int* _noalias numneigh = list->numneigh;

  vec3_t* _noalias xx = (vec3_t*)x[0];
  vec3_t* _noalias ff = (vec3_t*)f[0];

  int ntypes = atom->ntypes;
  int ntypes2 = ntypes*ntypes;

  fast_alpha_t* _noalias fast_alpha =
    (fast_alpha_t*) malloc(ntypes2*sizeof(fast_alpha_t));
  for (i = 0; i < ntypes; i++) for (j = 0; j < ntypes; j++) {
    fast_alpha_t& a = fast_alpha[i*ntypes+j];
    a.cutsq = cutsq[i+1][j+1];
    a.lj1 = lj1[i+1][j+1];
    a.lj2 = lj2[i+1][j+1];
    a.lj3 = lj3[i+1][j+1];
    a.lj4 = lj4[i+1][j+1];
    a.offset = offset[i+1][j+1];
  }
  fast_alpha_t* _noalias tabsix = fast_alpha;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    double xtmp = xx[i].x;
    double ytmp = xx[i].y;
    double ztmp = xx[i].z;
    itype = type[i] - 1;
    int* _noalias jlist = firstneigh[i];
    jnum = numneigh[i];

    double tmpfx = 0.0;
    double tmpfy = 0.0;
    double tmpfz = 0.0;

    fast_alpha_t* _noalias tabsixi = (fast_alpha_t*)&tabsix[itype*ntypes];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      sbindex = sbmask(j);

      if (sbindex == 0) {
        double delx = xtmp - xx[j].x;
        double dely = ytmp - xx[j].y;
        double delz = ztmp - xx[j].z;
        double rsq = delx*delx + dely*dely + delz*delz;

        jtype = type[j] - 1;

        fast_alpha_t& a = tabsixi[jtype];

        if (rsq < a.cutsq) {
          double r2inv = 1.0/rsq;
          double r6inv = r2inv*r2inv*r2inv;
          double forcelj = r6inv * (a.lj1*r6inv - a.lj2);
          double fpair = forcelj*r2inv;

          tmpfx += delx*fpair;
          tmpfy += dely*fpair;
          tmpfz += delz*fpair;
          if (NEWTON_PAIR || j < nlocal) {
            ff[j].x -= delx*fpair;
            ff[j].y -= dely*fpair;
            ff[j].z -= delz*fpair;
          }

          if (EFLAG) evdwl = r6inv*(a.lj3*r6inv-a.lj4) - a.offset;

          if (EVFLAG)
            ev_tally(i,j,nlocal,NEWTON_PAIR,
                     evdwl,0.0,fpair,delx,dely,delz);
        }

      } else {
        factor_lj = special_lj[sbindex];
        j &= NEIGHMASK;

        double delx = xtmp - xx[j].x;
        double dely = ytmp - xx[j].y;
        double delz = ztmp - xx[j].z;
        double rsq = delx*delx + dely*dely + delz*delz;

        int jtype1 = type[j];
        jtype = jtype1 - 1;

        fast_alpha_t& a = tabsixi[jtype];
        if (rsq < a.cutsq) {
          double r2inv = 1.0/rsq;
          double r6inv = r2inv*r2inv*r2inv;
          fast_alpha_t& a = tabsixi[jtype];
          double forcelj = r6inv * (a.lj1*r6inv - a.lj2);
          double fpair = factor_lj*forcelj*r2inv;

          tmpfx += delx*fpair;
          tmpfy += dely*fpair;
          tmpfz += delz*fpair;
          if (NEWTON_PAIR || j < nlocal) {
            ff[j].x -= delx*fpair;
            ff[j].y -= dely*fpair;
            ff[j].z -= delz*fpair;
          }

          if (EFLAG) {
            evdwl = r6inv*(a.lj3*r6inv-a.lj4) - a.offset;
            evdwl *= factor_lj;
          }

          if (EVFLAG) ev_tally(i,j,nlocal,NEWTON_PAIR,
                               evdwl,0.0,fpair,delx,dely,delz);
        }
      }
    }

    ff[i].x += tmpfx;
    ff[i].y += tmpfy;
    ff[i].z += tmpfz;
  }

  free(fast_alpha); fast_alpha = 0;

  if (vflag_fdotr) virial_fdotr_compute();
}
