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
   Contributing author: Trung Dac Nguyen (ORNL)
------------------------------------------------------------------------- */

#include "pair_lj_cut_dipole_cut_gpu.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "gpu_extra.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "suffix.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

// External functions from cuda library for atom decomposition

int dpl_gpu_init(const int ntypes, double **cutsq, double **host_lj1,
                 double **host_lj2, double **host_lj3, double **host_lj4,
                 double **offset, double *special_lj, const int nlocal,
                 const int nall, const int max_nbors, const int maxspecial,
                 const double cell_size, int &gpu_mode, FILE *screen,
                 double **host_cut_ljsq, double **host_cut_coulsq,
                 double *host_special_coul, const double qqrd2e);
void dpl_gpu_clear();
int ** dpl_gpu_compute_n(const int ago, const int inum,
                         const int nall, double **host_x, int *host_type,
                         double *sublo, double *subhi, tagint *tag,
                         int **nspecial, tagint **special, const bool eflag,
                         const bool vflag, const bool eatom, const bool vatom,
                         int &host_start, int **ilist, int **jnum,
                         const double cpu_time, bool &success,
                         double *host_q, double **host_mu,
                         double *boxlo, double *prd);
void dpl_gpu_compute(const int ago, const int inum,
                     const int nall, double **host_x, int *host_type,
                     int *ilist, int *numj, int **firstneigh,
                     const bool eflag, const bool vflag, const bool eatom,
                     const bool vatom, int &host_start, const double cpu_time,
                     bool &success, double *host_q, double **host_mu,
                     const int nlocal, double *boxlo, double *prd);
double dpl_gpu_bytes();

/* ---------------------------------------------------------------------- */

PairLJCutDipoleCutGPU::PairLJCutDipoleCutGPU(LAMMPS *lmp) : PairLJCutDipoleCut(lmp),
  gpu_mode(GPU_FORCE)
{
  respa_enable = 0;
  reinitflag = 0;
  cpu_time = 0.0;
  suffix_flag |= Suffix::GPU;
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/* ----------------------------------------------------------------------
   free all arrays
------------------------------------------------------------------------- */

PairLJCutDipoleCutGPU::~PairLJCutDipoleCutGPU()
{
  dpl_gpu_clear();
}

/* ---------------------------------------------------------------------- */

void PairLJCutDipoleCutGPU::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  int nall = atom->nlocal + atom->nghost;
  int inum, host_start;

  bool success = true;
  int *ilist, *numneigh, **firstneigh;
  if (gpu_mode != GPU_FORCE) {
    double sublo[3],subhi[3];
    if (domain->triclinic == 0) {
      sublo[0] = domain->sublo[0];
      sublo[1] = domain->sublo[1];
      sublo[2] = domain->sublo[2];
      subhi[0] = domain->subhi[0];
      subhi[1] = domain->subhi[1];
      subhi[2] = domain->subhi[2];
    } else {
      domain->bbox(domain->sublo_lamda,domain->subhi_lamda,sublo,subhi);
    }
    inum = atom->nlocal;
    firstneigh = dpl_gpu_compute_n(neighbor->ago, inum, nall, atom->x,
                                   atom->type, sublo, subhi,
                                   atom->tag, atom->nspecial, atom->special,
                                   eflag, vflag, eflag_atom, vflag_atom,
                                   host_start, &ilist, &numneigh, cpu_time,
                                   success, atom->q, atom->mu, domain->boxlo,
                                   domain->prd);
  } else {
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    dpl_gpu_compute(neighbor->ago, inum, nall, atom->x, atom->type,
                    ilist, numneigh, firstneigh, eflag, vflag, eflag_atom,
                    vflag_atom, host_start, cpu_time, success, atom->q,
                    atom->mu, atom->nlocal, domain->boxlo, domain->prd);
  }
  if (!success)
    error->one(FLERR,"Insufficient memory on accelerator");

  if (host_start<inum) {
    cpu_time = MPI_Wtime();
    cpu_compute(host_start, inum, eflag, vflag, ilist, numneigh, firstneigh);
    cpu_time = MPI_Wtime() - cpu_time;
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutDipoleCutGPU::init_style()
{
  if (!atom->q_flag || !atom->mu_flag || !atom->torque_flag)
    error->all(FLERR,"Pair dipole/cut/gpu requires atom attributes q, mu, torque");

  if (force->newton_pair)
    error->all(FLERR,"Pair style dipole/cut/gpu requires newton pair off");

  if (strcmp(update->unit_style,"electron") == 0)
    error->all(FLERR,"Cannot (yet) use 'electron' units with dipoles");

  // Repeat cutsq calculation because done after call to init_style
  double maxcut = -1.0;
  double cut;
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
        cut = init_one(i,j);
        cut *= cut;
        if (cut > maxcut)
          maxcut = cut;
        cutsq[i][j] = cutsq[j][i] = cut;
      } else
        cutsq[i][j] = cutsq[j][i] = 0.0;
    }
  }
  double cell_size = sqrt(maxcut) + neighbor->skin;

  int maxspecial=0;
  if (atom->molecular != Atom::ATOMIC)
    maxspecial=atom->maxspecial;
  int mnf = 5e-2 * neighbor->oneatom;
  int success = dpl_gpu_init(atom->ntypes+1, cutsq, lj1, lj2, lj3, lj4,
                             offset, force->special_lj, atom->nlocal,
                             atom->nlocal+atom->nghost, mnf, maxspecial,
                             cell_size, gpu_mode, screen, cut_ljsq, cut_coulsq,
                             force->special_coul, force->qqrd2e);
  GPU_EXTRA::check_flag(success,error,world);

  if (gpu_mode == GPU_FORCE) {
    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
  }
}

/* ---------------------------------------------------------------------- */

double PairLJCutDipoleCutGPU::memory_usage()
{
  double bytes = Pair::memory_usage();
  return bytes + dpl_gpu_bytes();
}

/* ---------------------------------------------------------------------- */

void PairLJCutDipoleCutGPU::cpu_compute(int start, int inum, int eflag, int vflag,
                                   int *ilist, int *numneigh,
                                   int **firstneigh)
{
  int i,j,ii,jj,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fx,fy,fz;
  double rsq,rinv,r2inv,r6inv,r3inv,r5inv,r7inv;
  double forcecoulx,forcecouly,forcecoulz,crossx,crossy,crossz;
  double tixcoul,tiycoul,tizcoul,tjxcoul,tjycoul,tjzcoul;
  double fq,pdotp,pidotr,pjdotr,pre1,pre2,pre3,pre4;
  double forcelj,factor_coul,factor_lj;
  int *jlist;

  evdwl = ecoul = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  double **mu = atom->mu;
  double **torque = atom->torque;
  int *type = atom->type;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  double qqrd2e = force->qqrd2e;


  // loop over neighbors of my atoms

  for (ii = start; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        rinv = sqrt(r2inv);

        // atom can have both a charge and dipole
        // i,j = charge-charge, dipole-dipole, dipole-charge, or charge-dipole

        forcecoulx = forcecouly = forcecoulz = 0.0;
        tixcoul = tiycoul = tizcoul = 0.0;
        tjxcoul = tjycoul = tjzcoul = 0.0;

        if (rsq < cut_coulsq[itype][jtype]) {

          if (qtmp != 0.0 && q[j] != 0.0) {
            r3inv = r2inv*rinv;
            pre1 = qtmp*q[j]*r3inv;

            forcecoulx += pre1*delx;
            forcecouly += pre1*dely;
            forcecoulz += pre1*delz;
          }

          if (mu[i][3] > 0.0 && mu[j][3] > 0.0) {
            r3inv = r2inv*rinv;
            r5inv = r3inv*r2inv;
            r7inv = r5inv*r2inv;

            pdotp = mu[i][0]*mu[j][0] + mu[i][1]*mu[j][1] + mu[i][2]*mu[j][2];
            pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
            pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;

            pre1 = 3.0*r5inv*pdotp - 15.0*r7inv*pidotr*pjdotr;
            pre2 = 3.0*r5inv*pjdotr;
            pre3 = 3.0*r5inv*pidotr;
            pre4 = -1.0*r3inv;

            forcecoulx += pre1*delx + pre2*mu[i][0] + pre3*mu[j][0];
            forcecouly += pre1*dely + pre2*mu[i][1] + pre3*mu[j][1];
            forcecoulz += pre1*delz + pre2*mu[i][2] + pre3*mu[j][2];

            crossx = pre4 * (mu[i][1]*mu[j][2] - mu[i][2]*mu[j][1]);
            crossy = pre4 * (mu[i][2]*mu[j][0] - mu[i][0]*mu[j][2]);
            crossz = pre4 * (mu[i][0]*mu[j][1] - mu[i][1]*mu[j][0]);

            tixcoul += crossx + pre2 * (mu[i][1]*delz - mu[i][2]*dely);
            tiycoul += crossy + pre2 * (mu[i][2]*delx - mu[i][0]*delz);
            tizcoul += crossz + pre2 * (mu[i][0]*dely - mu[i][1]*delx);
            tjxcoul += -crossx + pre3 * (mu[j][1]*delz - mu[j][2]*dely);
            tjycoul += -crossy + pre3 * (mu[j][2]*delx - mu[j][0]*delz);
            tjzcoul += -crossz + pre3 * (mu[j][0]*dely - mu[j][1]*delx);
          }

          if (mu[i][3] > 0.0 && q[j] != 0.0) {
            r3inv = r2inv*rinv;
            r5inv = r3inv*r2inv;
            pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
            pre1 = 3.0*q[j]*r5inv * pidotr;
            pre2 = q[j]*r3inv;

            forcecoulx += pre2*mu[i][0] - pre1*delx;
            forcecouly += pre2*mu[i][1] - pre1*dely;
            forcecoulz += pre2*mu[i][2] - pre1*delz;
            tixcoul += pre2 * (mu[i][1]*delz - mu[i][2]*dely);
            tiycoul += pre2 * (mu[i][2]*delx - mu[i][0]*delz);
            tizcoul += pre2 * (mu[i][0]*dely - mu[i][1]*delx);
          }

          if (mu[j][3] > 0.0 && qtmp != 0.0) {
            r3inv = r2inv*rinv;
            r5inv = r3inv*r2inv;
            pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;
            pre1 = 3.0*qtmp*r5inv * pjdotr;
            pre2 = qtmp*r3inv;

            forcecoulx += pre1*delx - pre2*mu[j][0];
            forcecouly += pre1*dely - pre2*mu[j][1];
            forcecoulz += pre1*delz - pre2*mu[j][2];
            tjxcoul += -pre2 * (mu[j][1]*delz - mu[j][2]*dely);
            tjycoul += -pre2 * (mu[j][2]*delx - mu[j][0]*delz);
            tjzcoul += -pre2 * (mu[j][0]*dely - mu[j][1]*delx);
          }
        }

        // LJ interaction

        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
          forcelj *= factor_lj * r2inv;
        } else forcelj = 0.0;

        // total force

        fq = factor_coul*qqrd2e;
        fx = fq*forcecoulx + delx*forcelj;
        fy = fq*forcecouly + dely*forcelj;
        fz = fq*forcecoulz + delz*forcelj;

        // force & torque accumulation

        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;
        torque[i][0] += fq*tixcoul;
        torque[i][1] += fq*tiycoul;
        torque[i][2] += fq*tizcoul;

        if (eflag) {
          if (rsq < cut_coulsq[itype][jtype]) {
            ecoul = qtmp*q[j]*rinv;
            if (mu[i][3] > 0.0 && mu[j][3] > 0.0)
              ecoul += r3inv*pdotp - 3.0*r5inv*pidotr*pjdotr;
            if (mu[i][3] > 0.0 && q[j] != 0.0)
              ecoul += -q[j]*r3inv*pidotr;
            if (mu[j][3] > 0.0 && qtmp != 0.0)
              ecoul += qtmp*r3inv*pjdotr;
            ecoul *= factor_coul*qqrd2e;
          } else ecoul = 0.0;

          if (rsq < cut_ljsq[itype][jtype]) {
            evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
              offset[itype][jtype];
            evdwl *= factor_lj;
          } else evdwl = 0.0;
        }

        if (evflag) ev_tally_xyz_full(i,evdwl,ecoul,fx,fy,fz,delx,dely,delz);
      }
    }
  }
}
