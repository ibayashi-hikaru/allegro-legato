// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "pair_lj_charmm_coul_long_soft_omp.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "suffix.h"

#include <cmath>

#include "omp_compat.h"
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJCharmmCoulLongSoftOMP::PairLJCharmmCoulLongSoftOMP(LAMMPS *lmp) :
  PairLJCharmmCoulLongSoft(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
  cut_respa = nullptr;
}

/* ---------------------------------------------------------------------- */

void PairLJCharmmCoulLongSoftOMP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (evflag) {
      if (eflag) {
        if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
        else eval<1,1,0>(ifrom, ito, thr);
      } else {
        if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
        else eval<1,0,0>(ifrom, ito, thr);
      }
    } else {
      if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
      else eval<0,0,0>(ifrom, ito, thr);
    }

    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

/* ---------------------------------------------------------------------- */

template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairLJCharmmCoulLongSoftOMP::eval(int iifrom, int iito, ThrData * const thr)
{

  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  dbl3_t * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const double * _noalias const q = atom->q;
  const int * _noalias const type = atom->type;
  const double * _noalias const special_coul = force->special_coul;
  const double * _noalias const special_lj = force->special_lj;
  const double qqrd2e = force->qqrd2e;
  const double inv_denom_lj = 1.0/denom_lj;

  const int * const ilist = list->ilist;
  const int * const numneigh = list->numneigh;
  const int * const * const firstneigh = list->firstneigh;
  const int nlocal = atom->nlocal;

  // loop over neighbors of my atoms

  for (int ii = iifrom; ii < iito; ++ii) {

    const int i = ilist[ii];
    const int itype = type[i];
    const double qtmp = q[i];
    const double xtmp = x[i].x;
    const double ytmp = x[i].y;
    const double ztmp = x[i].z;
    double fxtmp,fytmp,fztmp;
    fxtmp=fytmp=fztmp=0.0;

    const int * const jlist = firstneigh[i];
    const int jnum = numneigh[i];
    const double * _noalias const lj1i = lj1[itype];
    const double * _noalias const lj2i = lj2[itype];
    const double * _noalias const lj3i = lj3[itype];
    const double * _noalias const lj4i = lj4[itype];
    const double * _noalias const epsii = epsilon[itype];

    for (int jj = 0; jj < jnum; jj++) {
      double forcecoul, forcelj, evdwl, ecoul;
      forcecoul = forcelj = evdwl = ecoul = 0.0;

      const int sbindex = sbmask(jlist[jj]);
      const int j = jlist[jj] & NEIGHMASK;

      const double delx = xtmp - x[j].x;
      const double dely = ytmp - x[j].y;
      const double delz = ztmp - x[j].z;
      const double rsq = delx*delx + dely*dely + delz*delz;
      const int jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {

        if (rsq < cut_coulsq) {
          const double A1 =  0.254829592;
          const double A2 = -0.284496736;
          const double A3 =  1.421413741;
          const double A4 = -1.453152027;
          const double A5 =  1.061405429;
          const double EWALD_F = 1.12837917;
          const double INV_EWALD_P = 1.0/0.3275911;

          const double r = sqrt(rsq);
          const double grij = g_ewald * r;
          const double expm2 = exp(-grij*grij);
          const double t = INV_EWALD_P / (INV_EWALD_P + grij);
          const double erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
          const double denc = sqrt(lj4i[jtype] + rsq);
          const double prefactor = qqrd2e * lj1i[jtype] * qtmp*q[j] / (denc*denc*denc);

          forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
          if (EFLAG) ecoul = prefactor*erfc*denc*denc;

          if (sbindex) {
            const double adjust = (1.0-special_coul[sbindex])*prefactor;
            forcecoul -= adjust;
            if (EFLAG) ecoul -= adjust*denc*denc;
          }
        }

        if (rsq < cut_ljsq) {
          const double r4sig6 = rsq*rsq / lj2i[jtype];
          const double denlj = lj3i[jtype] + rsq*r4sig6;
          forcelj = lj1i[jtype] * epsii[jtype] *
            (48.0*r4sig6/(denlj*denlj*denlj) - 24.0*r4sig6/(denlj*denlj));
          const double philj = lj1i[jtype] * 4.0 * epsii[jtype]
            * (1.0/(denlj*denlj) - 1.0/denlj);
          if (EFLAG) evdwl = philj;

          if (rsq > cut_lj_innersq) {
            const double drsq = cut_ljsq - rsq;
            const double cut2 = (rsq - cut_lj_innersq) * drsq;
            const double switch1 = drsq * (drsq*drsq + 3.0*cut2) * inv_denom_lj;
            const double switch2 = 12.0 * cut2 * inv_denom_lj;
            forcelj = forcelj*switch1 + philj*switch2;
            if (EFLAG) evdwl = philj*switch1;
          }

          if (sbindex) {
            const double factor_lj = special_lj[sbindex];
            forcelj *= factor_lj;
            if (EFLAG) evdwl *= factor_lj;
          }
        }
        const double fpair = forcecoul + forcelj;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;
        if (NEWTON_PAIR || j < nlocal) {
          f[j].x -= delx*fpair;
          f[j].y -= dely*fpair;
          f[j].z -= delz*fpair;
        }

        if (EVFLAG) ev_tally_thr(this,i,j,nlocal,NEWTON_PAIR,
                                 evdwl,ecoul,fpair,delx,dely,delz,thr);
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}

/* ---------------------------------------------------------------------- */

double PairLJCharmmCoulLongSoftOMP::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairLJCharmmCoulLongSoft::memory_usage();

  return bytes;
}
