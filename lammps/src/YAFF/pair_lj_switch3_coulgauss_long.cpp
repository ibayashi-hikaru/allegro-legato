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
   Contributing author: Steven Vandenbrande
------------------------------------------------------------------------- */

#include "pair_lj_switch3_coulgauss_long.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

/* ---------------------------------------------------------------------- */

PairLJSwitch3CoulGaussLong::PairLJSwitch3CoulGaussLong(LAMMPS *lmp) : Pair(lmp)
{
  ewaldflag = pppmflag = 1;
  writedata = 1;
  ftable = nullptr;
  qdist = 0.0;
}

/* ---------------------------------------------------------------------- */

PairLJSwitch3CoulGaussLong::~PairLJSwitch3CoulGaussLong()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(gamma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
  if (ftable) free_tables();
}

/* ---------------------------------------------------------------------- */

void PairLJSwitch3CoulGaussLong::compute(int eflag, int vflag)
{
  int i,ii,j,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcecoul2,forcelj,factor_coul,factor_lj,tr,ftr,trx;
  double grij,expm2,prefactor,prefactor2,t,erfc1,erfc2,rrij,expn2;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

  evdwl = ecoul = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;
  union_int_float_t rsq_lookup;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
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

        if (rsq < cut_coulsq) {
          if (!ncoultablebits || rsq <= tabinnersq) {
            r = sqrt(rsq);
            grij = g_ewald * r;
            expm2 = exp(-grij*grij);
            t = 1.0 / (1.0 + EWALD_P*grij);
            erfc1 = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
            prefactor = qqrd2e * qtmp*q[j]/r;
            forcecoul = prefactor * (erfc1 + EWALD_F*grij*expm2);
            if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
          } else {
            rsq_lookup.f = rsq;
            itable = rsq_lookup.i & ncoulmask;
            itable >>= ncoulshiftbits;
            fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
            table = ftable[itable] + fraction*dftable[itable];
            forcecoul = qtmp*q[j] * table;
            if (factor_coul < 1.0) {
              table = ctable[itable] + fraction*dctable[itable];
              prefactor = qtmp*q[j] * table;
              forcecoul -= (1.0-factor_coul)*prefactor;
            }
          }
        } else forcecoul = 0.0;

        if (rsq < cut_ljsq[itype][jtype]) {
          // Lennard-Jones potential
          r = sqrt(rsq);
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv*(12.0*lj3[itype][jtype]*r6inv - 6.0*lj4[itype][jtype]);
          // Correction for Gaussian radii
          if (lj2[itype][jtype]==0.0) {
            // This means a point charge is considered, so the correction is zero
            expn2 = 0.0;
            erfc2 = 0.0;
            forcecoul2 = 0.0;
            prefactor2 = 0.0;
          } else {
            rrij = lj2[itype][jtype]*r;
            expn2 = exp(-rrij*rrij);
            erfc2 = erfc(rrij);
            prefactor2 = -qqrd2e*qtmp*q[j]/r;
            forcecoul2 = prefactor2*(erfc2+EWALD_F*rrij*expn2);
          }
        } else forcelj = 0.0;

        if (rsq < cut_coulsq) {
          if (!ncoultablebits || rsq <= tabinnersq)
            ecoul = prefactor*erfc1;
          else {
            table = etable[itable] + fraction*detable[itable];
            ecoul = qtmp*q[j] * table;
          }
          if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
        } else ecoul = 0.0;

        if (rsq < cut_ljsq[itype][jtype]) {
          ecoul += prefactor2*erfc2*factor_coul;
          evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
              offset[itype][jtype];
        } else evdwl = 0.0;

        // Truncation, see Yaff Switch3
        if (truncw>0) {
          if (rsq < cut_ljsq[itype][jtype]) {
            if (r>cut_lj[itype][jtype]-truncw) {
              trx = (cut_lj[itype][jtype]-r)*truncwi;
              tr = trx*trx*(3.0-2.0*trx);
              ftr = 6.0*trx*(1.0-trx)*r*truncwi;
              forcelj = forcelj*tr + evdwl*ftr;
              evdwl *= tr;
            }
          }
        }

        fpair = (forcecoul + factor_coul*forcecoul2 + factor_lj*forcelj) * r2inv;
        evdwl *= factor_lj;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,ecoul,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJSwitch3CoulGaussLong::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut_lj,n+1,n+1,"pair:cut_lj");
  memory->create(cut_ljsq,n+1,n+1,"pair:cut_ljsq");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(gamma,n+1,n+1,"pair:gamma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJSwitch3CoulGaussLong::settings(int narg, char **arg)
{
  if (narg < 2 || narg > 3) error->all(FLERR,"Illegal pair_style command");

  cut_lj_global = utils::numeric(FLERR,arg[0],false,lmp);

  if (narg == 2) {
    cut_coul = cut_lj_global;
    truncw = utils::numeric(FLERR,arg[1],false,lmp);
  } else {
    cut_coul = utils::numeric(FLERR,arg[1],false,lmp);
    truncw = utils::numeric(FLERR,arg[2],false,lmp);
  }

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJSwitch3CoulGaussLong::coeff(int narg, char **arg)
{
  if (narg < 5 || narg > 6)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double epsilon_one = utils::numeric(FLERR,arg[2],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[3],false,lmp);
  double gamma_one = utils::numeric(FLERR,arg[4],false,lmp);

  double cut_lj_one = cut_lj_global;
  if (narg == 6) cut_lj_one = utils::numeric(FLERR,arg[5],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      gamma[i][j] = gamma_one;
      cut_lj[i][j] = cut_lj_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJSwitch3CoulGaussLong::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair style lj/switch3/coulgauss/long requires atom attribute q");

  cut_coulsq = cut_coul * cut_coul;

  if (truncw>0.0) truncwi = 1.0/truncw;
  else truncwi = 0.0;

  // insure use of KSpace long-range solver, set g_ewald

  if (force->kspace == nullptr)
    error->all(FLERR,"Pair style requires a KSpace style");
  g_ewald = force->kspace->g_ewald;

  neighbor->request(this,instance_me);

  // setup force tables

  if (ncoultablebits) init_tables(cut_coul,nullptr);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJSwitch3CoulGaussLong::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    gamma[i][j] = 1.0/sqrt(gamma[i][i]*gamma[i][i]+gamma[j][j]*gamma[j][j]);
    cut_lj[i][j] = mix_distance(cut_lj[i][i],cut_lj[j][j]);
  }

  double cut = MAX(cut_lj[i][j],cut_coul+2.0*qdist);
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];
  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  if (gamma[i][i]==0.0 && gamma[j][j]==0.0) lj2[i][j] = 0.0;
  else lj2[i][j] = 1.0/sqrt(gamma[i][i]*gamma[i][i]+gamma[j][j]*gamma[j][j]);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag && (cut_lj[i][j] > 0.0)) {
    // Truncation is active, so offset is zero, except if truncw==0.0
    if (truncw==0.0) {
      double r = cut_lj[i][j];
      double r2inv = 1.0/(r*r);
      double r6inv = r2inv*r2inv*r2inv;
      double r12inv = r6inv*r6inv;
      offset[i][j] = lj3[i][j]*r12inv-lj4[i][j]*r6inv;
    } else {offset[i][j] = 0.0;}
  } else offset[i][j] = 0.0;

  cut_ljsq[j][i] = cut_ljsq[i][j];
  cut_lj[j][i] = cut_lj[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double cg = epsilon[i][j];
    double cg1 = cut_lj[i][j];
    double cg3 = sigma[i][j];
    if (truncw > 0.0) {
      double cg5 = truncw;
      double t1 = pow(cg1, 0.2e1);
      double t2 = t1 * cg1;
      double t3 = t1 * t1;
      double t4 = t3 * t2;
      double t5 = -cg1 + cg5;
      double t6 = t5 * t5;
      double t8 = t6 * t6;
      double t9 = t8 * t6 * t5;
      double t10 = t4 * t9;
      double t11 = log(-t5);
      double t14 = log(cg1);
      double t19 = pow(cg5, 0.2e1);
      double t20 = t19 * t19;
      double t21 = t20 * t19;
      double t24 = pow(cg3, 0.2e1);
      double t25 = t24 * t24;
      double t26 = t25 * t24;
      double t29 = t20 * cg5;
      double t35 = t3 * t3;
      double t41 = t19 * cg5;
      double t58 = t21 * t3 * t1 - t21 * t26 / 0.84e2 - 0.6e1 * t29 * t4 + t29 * cg1 * t26 / 0.18e2 + 0.15e2 * t20 * t35 - t20 * t1 * t26 / 0.9e1 - 0.20e2 * t41 * t35 * cg1 + t41 * t2 * t26 / 0.9e1 + 0.15e2 * t19 * t35 * t1 - t19 * t3 * t26 / 0.18e2 - 0.6e1 * t35 * t2 * cg5 + t35 * t3;
      double t71 = -0.4e1 * cg * (0.2e1 * t10 * t11 - 0.2e1 * t10 * t14 + (cg5 - 0.2e1 * cg1) * t58 * cg5) * t26 / t4 / t41 / t9;
      etail_ij = 2.0*MY_PI*all[0]*all[1]*t71;
      ptail_ij = 2.0*MY_PI*all[0]*all[1]*t71;
    } else {
      double t1 = pow(cg3, 0.2e1);
      double t2 = t1 * t1;
      double t3 = t2 * t1;
      double t5 = pow(cg1, 0.2e1);
      double t6 = t5 * t5;
      double t10 = t6 * t6;
      double t16 = -0.4e1 / 0.9e1 * t3 * cg * (0.3e1 * t6 * t5 - t3) / t10 / cg1;
      t1 = pow(cg3, 0.2e1);
      t2 = t1 * t1;
      t3 = t2 * t1;
      t5 = pow(cg1, 0.2e1);
      t6 = t5 * t5;
      double t11 = t6 * t6;
      double t17 = 0.8e1 / 0.3e1 * t3 * cg * (0.3e1 * t6 * t5 - 0.2e1 * t3) / t11 / cg1;
      etail_ij = 2.0*MY_PI*all[0]*all[1]*t16;
      ptail_ij = -2.0/3.0*MY_PI*all[0]*all[1]*t17;
    }
  }

  return cut;
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJSwitch3CoulGaussLong::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&gamma[i][j],sizeof(double),1,fp);
        fwrite(&cut_lj[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJSwitch3CoulGaussLong::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&epsilon[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigma[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&gamma[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut_lj[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJSwitch3CoulGaussLong::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&truncw,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
  fwrite(&ncoultablebits,sizeof(int),1,fp);
  fwrite(&tabinner,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJSwitch3CoulGaussLong::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_lj_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_coul,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&truncw,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&tail_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&ncoultablebits,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&tabinner,sizeof(double),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&truncw,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
  MPI_Bcast(&ncoultablebits,1,MPI_INT,0,world);
  MPI_Bcast(&tabinner,1,MPI_DOUBLE,0,world);
}


/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJSwitch3CoulGaussLong::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g\n",i,epsilon[i][i],sigma[i][i],gamma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJSwitch3CoulGaussLong::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g\n",i,j,epsilon[i][j],sigma[i][j],gamma[i][j],cut_lj[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairLJSwitch3CoulGaussLong::single(int i, int j, int itype, int jtype,
                                          double rsq,
                                          double factor_coul, double factor_lj,
                                          double &fforce)
{
  double r2inv,r6inv,r,grij,expm2,t,erfc1,prefactor,prefactor2;
  double fraction,table,forcecoul,forcecoul2,forcelj;
  double rrij,expn2,erfc2,ecoul,evdwl,trx,tr,ftr;

  int itable;

  r2inv = 1.0/rsq;
  r = sqrt(rsq);
  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq) {
      grij = g_ewald * r;
      expm2 = exp(-grij*grij);
      t = 1.0 / (1.0 + EWALD_P*grij);
      erfc1 = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
      prefactor = force->qqrd2e * atom->q[i]*atom->q[j]/r;
      forcecoul = prefactor * (erfc1 + EWALD_F*grij*expm2);
      if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
    } else {
      union_int_float_t rsq_lookup_single;
      rsq_lookup_single.f = rsq;
      itable = rsq_lookup_single.i & ncoulmask;
      itable >>= ncoulshiftbits;
      fraction = (rsq_lookup_single.f - rtable[itable]) * drtable[itable];
      table = ftable[itable] + fraction*dftable[itable];
      forcecoul = atom->q[i]*atom->q[j] * table;
      if (factor_coul < 1.0) {
        table = ctable[itable] + fraction*dctable[itable];
        prefactor = atom->q[i]*atom->q[j] * table;
        forcecoul -= (1.0-factor_coul)*prefactor;
      }
    }
  } else forcecoul = 0.0;

  if (rsq < cut_ljsq[itype][jtype]) {
    r6inv = r2inv*r2inv*r2inv;
    forcelj = r6inv*(12.0*lj3[itype][jtype]*r6inv-6.0*lj4[itype][jtype]);
    if (lj2[itype][jtype] == 0.0) {
      expn2 = 0.0;
      erfc2 = 0.0;
      forcecoul2 = 0.0;
      prefactor2 = 0.0;
    } else {
      rrij = lj2[itype][jtype]*r;
      expn2 = exp(-rrij*rrij);
      erfc2 = erfc(rrij);
      prefactor2 = -force->qqrd2e*atom->q[i]*atom->q[j]/r;
      forcecoul2 = prefactor2*(erfc2+EWALD_F*rrij*expn2);
    }
  } else forcelj = 0.0;

  evdwl = ecoul = 0.0;
  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq)
      ecoul = prefactor*erfc1;
    else {
      table = etable[itable] + fraction*detable[itable];
      ecoul = atom->q[i]*atom->q[j] * table;
    }
    if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
  }

  if (rsq < cut_ljsq[itype][jtype]) {
    ecoul += prefactor2*erfc2*factor_coul;
    evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
        offset[itype][jtype];
  } else evdwl = 0.0;

  // Truncation, see Yaff Switch3
  if (truncw > 0) {
    if (rsq < cut_ljsq[itype][jtype]) {
      if (r>cut_lj[itype][jtype]-truncw) {
        trx = (cut_lj[itype][jtype]-r)*truncwi;
        tr = trx*trx*(3.0-2.0*trx);
        ftr = 6.0*trx*(1.0-trx)*r*truncwi;
        forcelj = forcelj*tr + evdwl*ftr;
        evdwl *= tr;
      }
    }
  }
  fforce = (forcecoul + factor_coul*forcecoul2 + factor_lj*forcelj) * r2inv;

  return evdwl*factor_lj + ecoul;
}

/* ---------------------------------------------------------------------- */

void *PairLJSwitch3CoulGaussLong::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"cut_coul") == 0) return (void *) &cut_coul;
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  if (strcmp(str,"gamma") == 0) return (void *) gamma;
  return nullptr;
}
