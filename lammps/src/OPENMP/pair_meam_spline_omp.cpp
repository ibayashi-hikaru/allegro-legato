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

#include "pair_meam_spline_omp.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "neigh_list.h"
#include "suffix.h"

#include <cmath>

#include "omp_compat.h"
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairMEAMSplineOMP::PairMEAMSplineOMP(LAMMPS *lmp) :
  PairMEAMSpline(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
}

/* ---------------------------------------------------------------------- */

void PairMEAMSplineOMP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = listfull->inum;

  if (listhalf->inum != inum)
    error->warning(FLERR,"inconsistent half and full neighborlist");

  // Grow per-atom array if necessary.

  if (atom->nmax > nmax) {
    memory->destroy(Uprime_values);
    nmax = atom->nmax;
    memory->create(Uprime_values,nmax*nthreads,"pair:Uprime");
  }

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    thr->init_eam(nall,Uprime_values);

    if (evflag) {
      if (eflag) {
        eval<1,1>(ifrom, ito, thr);
      } else {
        eval<1,0>(ifrom, ito, thr);
      }
    } else {
      eval<0,0>(ifrom, ito, thr);
    }

    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

template <int EVFLAG, int EFLAG>
void PairMEAMSplineOMP::eval(int iifrom, int iito, ThrData * const thr)
{
  const int* const ilist_full = listfull->ilist;
  const int* const numneigh_full = listfull->numneigh;
  const int* const * const firstneigh_full = listfull->firstneigh;

  // Determine the maximum number of neighbors a single atom has.
  int myMaxNeighbors = 0;
  for (int ii = iifrom; ii < iito; ii++) {
    int jnum = numneigh_full[ilist_full[ii]];
    if (jnum > myMaxNeighbors) myMaxNeighbors = jnum;
  }

  // Allocate array for temporary bond info.
  MEAM2Body *myTwoBodyInfo = new MEAM2Body[myMaxNeighbors];

  const double * const * const x = atom->x;
  double * const * const forces = thr->get_f();
  double * const Uprime_thr = thr->get_rho();
  const int tid = thr->get_tid();
  const int nthreads = comm->nthreads;
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const int ntypes = atom->ntypes;

  const double cutforcesq = cutoff*cutoff;

  // Sum three-body contributions to charge density and compute embedding energies.
  for (int ii = iifrom; ii < iito; ii++) {

    const int i = ilist_full[ii];
    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];
    const int* const jlist = firstneigh_full[i];
    const int jnum = numneigh_full[i];
    double rho_value = 0;
    int numBonds = 0;
    MEAM2Body* nextTwoBodyInfo = myTwoBodyInfo;

    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;

      const double jdelx = x[j][0] - xtmp;
      const double jdely = x[j][1] - ytmp;
      const double jdelz = x[j][2] - ztmp;
      const double rij_sq = jdelx*jdelx + jdely*jdely + jdelz*jdelz;

      if (rij_sq < cutforcesq) {
        const int jtype = atom->type[j];
        const double rij = sqrt(rij_sq);
        double partial_sum = 0;

        nextTwoBodyInfo->tag = j;
        nextTwoBodyInfo->r = rij;
        nextTwoBodyInfo->f = fs[i_to_potl(jtype)].eval(rij, nextTwoBodyInfo->fprime);
        nextTwoBodyInfo->del[0] = jdelx / rij;
        nextTwoBodyInfo->del[1] = jdely / rij;
        nextTwoBodyInfo->del[2] = jdelz / rij;

        for (int kk = 0; kk < numBonds; kk++) {
          const MEAM2Body& bondk = myTwoBodyInfo[kk];
          double cos_theta = (nextTwoBodyInfo->del[0]*bondk.del[0] +
                              nextTwoBodyInfo->del[1]*bondk.del[1] +
                              nextTwoBodyInfo->del[2]*bondk.del[2]);
          partial_sum += bondk.f * gs[ij_to_potl(jtype,atom->type[bondk.tag],ntypes)].eval(cos_theta);
        }

        rho_value += nextTwoBodyInfo->f * partial_sum;
        rho_value += rhos[i_to_potl(jtype)].eval(rij);

        numBonds++;
        nextTwoBodyInfo++;
      }
    }

    const int itype = atom->type[i];
    // Compute embedding energy and its derivative.
    double Uprime_i;
    double embeddingEnergy = Us[i_to_potl(itype)].eval(rho_value, Uprime_i)
      - zero_atom_energies[i_to_potl(itype)];
    Uprime_thr[i] = Uprime_i;
    if (EFLAG)
      e_tally_thr(this,i,i,nlocal,1/*newton_pair*/,embeddingEnergy,0.0,thr);

    double forces_i[3] = {0.0, 0.0, 0.0};

    // Compute three-body contributions to force.
    for (int jj = 0; jj < numBonds; jj++) {
      const MEAM2Body bondj = myTwoBodyInfo[jj];
      const double rij = bondj.r;
      const int j = bondj.tag;
      const int jtype = atom->type[j];

      const double f_rij_prime = bondj.fprime;
      const double f_rij = bondj.f;

      double forces_j[3] = {0.0, 0.0, 0.0};

      MEAM2Body const* bondk = myTwoBodyInfo;
      for (int kk = 0; kk < jj; kk++, ++bondk) {
        const double rik = bondk->r;

        const double cos_theta = (bondj.del[0]*bondk->del[0]
                                  + bondj.del[1]*bondk->del[1]
                                  + bondj.del[2]*bondk->del[2]);
        double g_prime;
        double g_value = gs[ij_to_potl(jtype,atom->type[bondk->tag],ntypes)].eval(cos_theta, g_prime);
        const double f_rik_prime = bondk->fprime;
        const double f_rik = bondk->f;

        double fij = -Uprime_i * g_value * f_rik * f_rij_prime;
        double fik = -Uprime_i * g_value * f_rij * f_rik_prime;

        const double prefactor = Uprime_i * f_rij * f_rik * g_prime;
        const double prefactor_ij = prefactor / rij;
        const double prefactor_ik = prefactor / rik;
        fij += prefactor_ij * cos_theta;
        fik += prefactor_ik * cos_theta;

        double fj[3], fk[3];

        fj[0] = bondj.del[0] * fij - bondk->del[0] * prefactor_ij;
        fj[1] = bondj.del[1] * fij - bondk->del[1] * prefactor_ij;
        fj[2] = bondj.del[2] * fij - bondk->del[2] * prefactor_ij;
        forces_j[0] += fj[0];
        forces_j[1] += fj[1];
        forces_j[2] += fj[2];

        fk[0] = bondk->del[0] * fik - bondj.del[0] * prefactor_ik;
        fk[1] = bondk->del[1] * fik - bondj.del[1] * prefactor_ik;
        fk[2] = bondk->del[2] * fik - bondj.del[2] * prefactor_ik;
        forces_i[0] -= fk[0];
        forces_i[1] -= fk[1];
        forces_i[2] -= fk[2];

        const int k = bondk->tag;
        forces[k][0] += fk[0];
        forces[k][1] += fk[1];
        forces[k][2] += fk[2];

        if (EVFLAG) {
          double delta_ij[3];
          double delta_ik[3];
          delta_ij[0] = bondj.del[0] * rij;
          delta_ij[1] = bondj.del[1] * rij;
          delta_ij[2] = bondj.del[2] * rij;
          delta_ik[0] = bondk->del[0] * rik;
          delta_ik[1] = bondk->del[1] * rik;
          delta_ik[2] = bondk->del[2] * rik;
          ev_tally3_thr(this,i,j,k,0.0,0.0,fj,fk,delta_ij,delta_ik,thr);
        }
      }

      forces[i][0] -= forces_j[0];
      forces[i][1] -= forces_j[1];
      forces[i][2] -= forces_j[2];
      forces[j][0] += forces_j[0];
      forces[j][1] += forces_j[1];
      forces[j][2] += forces_j[2];
    }

    forces[i][0] += forces_i[0];
    forces[i][1] += forces_i[1];
    forces[i][2] += forces_i[2];
  }

  delete[] myTwoBodyInfo;

  sync_threads();

  // reduce per thread density
    thr->timer(Timer::PAIR);
  data_reduce_thr(Uprime_values, nall, nthreads, 1, tid);

  // wait until reduction is complete so that master thread
  // can communicate U'(rho) values.
  sync_threads();

#if defined(_OPENMP)
#pragma omp master
#endif
  { comm->forward_comm_pair(this); }

  // wait until master thread is done with communication
  sync_threads();

  const int* const ilist_half = listhalf->ilist;
  const int* const numneigh_half = listhalf->numneigh;
  const int* const * const firstneigh_half = listhalf->firstneigh;

  // Compute two-body pair interactions.
  for (int ii = iifrom; ii < iito; ii++) {
    const int i = ilist_half[ii];
    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];
    const int* const jlist = firstneigh_half[i];
    const int jnum = numneigh_half[i];
    const int itype = atom->type[i];

    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;

      double jdel[3];
      jdel[0] = x[j][0] - xtmp;
      jdel[1] = x[j][1] - ytmp;
      jdel[2] = x[j][2] - ztmp;
      double rij_sq = jdel[0]*jdel[0] + jdel[1]*jdel[1] + jdel[2]*jdel[2];

      if (rij_sq < cutforcesq) {
        double rij = sqrt(rij_sq);
        const int jtype = atom->type[j];

        double rho_prime_i,rho_prime_j;
        rhos[i_to_potl(itype)].eval(rij,rho_prime_i);
        rhos[i_to_potl(jtype)].eval(rij,rho_prime_j);
        double fpair = rho_prime_j * Uprime_values[i] + rho_prime_i*Uprime_values[j];

        double pair_pot_deriv;
        double pair_pot = phis[ij_to_potl(itype,jtype,ntypes)].eval(rij, pair_pot_deriv);

        fpair += pair_pot_deriv;

        // Divide by r_ij to get forces from gradient.
        fpair /= rij;

        forces[i][0] += jdel[0]*fpair;
        forces[i][1] += jdel[1]*fpair;
        forces[i][2] += jdel[2]*fpair;
        forces[j][0] -= jdel[0]*fpair;
        forces[j][1] -= jdel[1]*fpair;
        forces[j][2] -= jdel[2]*fpair;
        if (EVFLAG) ev_tally_thr(this,i,j,nlocal, 1 /* newton_pair */,
                                 pair_pot,0.0,-fpair,jdel[0],jdel[1],jdel[2],thr);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

double PairMEAMSplineOMP::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairMEAMSpline::memory_usage();

  return bytes;
}
