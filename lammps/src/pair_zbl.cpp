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
   Contributing authors: Stephen Foiles, Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include "pair_zbl.h"

#include <cmath>

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"

#include "pair_zbl_const.h"

// From J.F. Zeigler, J. P. Biersack and U. Littmark,
// "The Stopping and Range of Ions in Matter" volume 1, Pergamon, 1985.

using namespace LAMMPS_NS;
using namespace PairZBLConstants;

/* ---------------------------------------------------------------------- */

PairZBL::PairZBL(LAMMPS *lmp) : Pair(lmp) {
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairZBL::~PairZBL()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(z);
    memory->destroy(d1a);
    memory->destroy(d2a);
    memory->destroy(d3a);
    memory->destroy(d4a);
    memory->destroy(zze);
    memory->destroy(sw1);
    memory->destroy(sw2);
    memory->destroy(sw3);
    memory->destroy(sw4);
    memory->destroy(sw5);
  }
}

/* ---------------------------------------------------------------------- */

void PairZBL::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r,t,fswitch,eswitch;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cut_globalsq) {
        r = sqrt(rsq);
        fpair = dzbldr(r, itype, jtype);

        if (rsq > cut_innersq) {
          t = r - cut_inner;
          fswitch = t*t *
            (sw1[itype][jtype] + sw2[itype][jtype]*t);
          fpair += fswitch;
        }

        fpair *= -1.0/r;
        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = e_zbl(r, itype, jtype);
          evdwl += sw5[itype][jtype];
          if (rsq > cut_innersq) {
            eswitch = t*t*t *
              (sw3[itype][jtype] + sw4[itype][jtype]*t);
            evdwl += eswitch;
          }
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairZBL::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(z,n+1,"pair:z");
  memory->create(d1a,n+1,n+1,"pair:d1a");
  memory->create(d2a,n+1,n+1,"pair:d2a");
  memory->create(d3a,n+1,n+1,"pair:d3a");
  memory->create(d4a,n+1,n+1,"pair:d4a");
  memory->create(zze,n+1,n+1,"pair:zze");
  memory->create(sw1,n+1,n+1,"pair:sw1");
  memory->create(sw2,n+1,n+1,"pair:sw2");
  memory->create(sw3,n+1,n+1,"pair:sw3");
  memory->create(sw4,n+1,n+1,"pair:sw4");
  memory->create(sw5,n+1,n+1,"pair:sw5");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairZBL::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Illegal pair_style command");

  cut_inner = utils::numeric(FLERR,arg[0],false,lmp);
  cut_global = utils::numeric(FLERR,arg[1],false,lmp);

  if (cut_inner <= 0.0 )
    error->all(FLERR,"Illegal pair_style command");
  if (cut_inner > cut_global)
    error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs

------------------------------------------------------------------------- */

void PairZBL::coeff(int narg, char **arg)
{
  double z_one, z_two;

  if (narg != 4)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);

  int jlo,jhi;
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  z_one = utils::numeric(FLERR,arg[2],false,lmp);
  z_two = utils::numeric(FLERR,arg[3],false,lmp);

  // set flag for each i-j pair
  // set z-parameter only for i-i pairs

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      if (i == j) {
        if (z_one != z_two)
          error->all(FLERR,"Incorrect args for pair coefficients");
        z[i] = z_one;
      }
      setflag[i][j] = 1;
      set_coeff(i, j, z_one, z_two);
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairZBL::init_style()
{
  neighbor->request(this,instance_me);

  cut_innersq = cut_inner * cut_inner;
  cut_globalsq = cut_global * cut_global;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairZBL::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    set_coeff(i, j, z[i], z[j]);

  return cut_global;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairZBL::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i;
  for (i = 1; i <= atom->ntypes; i++) {
    fwrite(&setflag[i][i],sizeof(int),1,fp);
    if (setflag[i][i]) fwrite(&z[i],sizeof(double),1,fp);
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairZBL::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++) {
    if (me == 0) utils::sfread(FLERR,&setflag[i][i],sizeof(int),1,fp,nullptr,error);
    MPI_Bcast(&setflag[i][i],1,MPI_INT,0,world);
    if (setflag[i][i]) {
      if (me == 0) utils::sfread(FLERR,&z[i],sizeof(double),1,fp,nullptr,error);
      MPI_Bcast(&z[i],1,MPI_DOUBLE,0,world);
    }
  }

  for (i = 1; i <= atom->ntypes; i++)
    for (j = 1; j <= atom->ntypes; j++)
      set_coeff(i,j,z[i],z[j]);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairZBL::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&cut_inner,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairZBL::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_inner,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&tail_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_inner,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairZBL::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,z[i],z[i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairZBL::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g\n",i,j,z[i],z[j]);
}

/* ---------------------------------------------------------------------- */

double PairZBL::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                         double /*dummy1*/, double /*dummy2*/,
                         double &fforce)
{
  double phi,r,t,eswitch,fswitch;

  r = sqrt(rsq);
  fforce = dzbldr(r, itype, jtype);
  if (rsq > cut_innersq) {
    t = r - cut_inner;
    fswitch = t*t *
      (sw1[itype][jtype] + sw2[itype][jtype]*t);
    fforce += fswitch;
  }
  fforce *= -1.0/r;

  phi = e_zbl(r, itype, jtype);
  phi += sw5[itype][jtype];
  if (rsq > cut_innersq) {
    eswitch = t*t*t *
      (sw3[itype][jtype] + sw4[itype][jtype]*t);
    phi += eswitch;
  }

  return phi;
}

/* ----------------------------------------------------------------------
   compute ZBL pair energy
------------------------------------------------------------------------- */

double PairZBL::e_zbl(double r, int i, int j) {

  double d1aij = d1a[i][j];
  double d2aij = d2a[i][j];
  double d3aij = d3a[i][j];
  double d4aij = d4a[i][j];
  double zzeij = zze[i][j];
  double rinv = 1.0/r;

  double sum = c1*exp(-d1aij*r);
  sum += c2*exp(-d2aij*r);
  sum += c3*exp(-d3aij*r);
  sum += c4*exp(-d4aij*r);

  double result = zzeij*sum*rinv;

  return result;
}


/* ----------------------------------------------------------------------
   compute ZBL first derivative
------------------------------------------------------------------------- */

double PairZBL::dzbldr(double r, int i, int j) {

  double d1aij = d1a[i][j];
  double d2aij = d2a[i][j];
  double d3aij = d3a[i][j];
  double d4aij = d4a[i][j];
  double zzeij = zze[i][j];
  double rinv = 1.0/r;

  double e1 = exp(-d1aij*r);
  double e2 = exp(-d2aij*r);
  double e3 = exp(-d3aij*r);
  double e4 = exp(-d4aij*r);

  double sum = c1*e1;
  sum += c2*e2;
  sum += c3*e3;
  sum += c4*e4;

  double sum_p = -c1*d1aij*e1;
  sum_p -= c2*d2aij*e2;
  sum_p -= c3*d3aij*e3;
  sum_p -= c4*d4aij*e4;

  double result = zzeij*(sum_p - sum*rinv)*rinv;

  return result;
}

/* ----------------------------------------------------------------------
   compute ZBL second derivative
------------------------------------------------------------------------- */

double PairZBL::d2zbldr2(double r, int i, int j) {

  double d1aij = d1a[i][j];
  double d2aij = d2a[i][j];
  double d3aij = d3a[i][j];
  double d4aij = d4a[i][j];
  double zzeij = zze[i][j];
  double rinv = 1.0/r;

  double e1 = exp(-d1aij*r);
  double e2 = exp(-d2aij*r);
  double e3 = exp(-d3aij*r);
  double e4 = exp(-d4aij*r);

  double sum = c1*e1;
  sum += c2*e2;
  sum += c3*e3;
  sum += c4*e4;

  double sum_p = c1*e1*d1aij;
  sum_p += c2*e2*d2aij;
  sum_p += c3*e3*d3aij;
  sum_p += c4*e4*d4aij;

  double sum_pp = c1*e1*d1aij*d1aij;
  sum_pp += c2*e2*d2aij*d2aij;
  sum_pp += c3*e3*d3aij*d3aij;
  sum_pp += c4*e4*d4aij*d4aij;

  double result = zzeij*(sum_pp + 2.0*sum_p*rinv +
                         2.0*sum*rinv*rinv)*rinv;

  return result;
}

/* ----------------------------------------------------------------------
   calculate the i,j entries in the various coeff arrays
------------------------------------------------------------------------- */

void PairZBL::set_coeff(int i, int j, double zi, double zj)
{
  double ainv = (pow(zi,pzbl) + pow(zj,pzbl))/(a0*force->angstrom);
  d1a[i][j] = d1*ainv;
  d2a[i][j] = d2*ainv;
  d3a[i][j] = d3*ainv;
  d4a[i][j] = d4*ainv;
  zze[i][j] = zi*zj*force->qqr2e*force->qelectron*force->qelectron;

  d1a[j][i] = d1a[i][j];
  d2a[j][i] = d2a[i][j];
  d3a[j][i] = d3a[i][j];
  d4a[j][i] = d4a[i][j];
  zze[j][i] = zze[i][j];

  // e =  t^3 (sw3 + sw4*t) + sw5
  //   = A/3*t^3 + B/4*t^4 + C
  // sw3 = A/3
  // sw4 = B/4
  // sw5 = C

  // dedr = t^2 (sw1 + sw2*t)
  //      = A*t^2 + B*t^3
  // sw1 = A
  // sw2 = B

  // de2dr2 = 2*A*t + 3*B*t^2

  // Require that at t = tc:
  // e = -Fc
  // dedr = -Fc'
  // d2edr2 = -Fc''

  // Hence:
  // A = (-3Fc' + tc*Fc'')/tc^2
  // B = ( 2Fc' - tc*Fc'')/tc^3
  // C = -Fc + tc/2*Fc' - tc^2/12*Fc''

  double tc = cut_global - cut_inner;
  double fc = e_zbl(cut_global, i, j);
  double fcp = dzbldr(cut_global, i, j);
  double fcpp = d2zbldr2(cut_global, i, j);

  double swa = (-3.0*fcp + tc*fcpp)/(tc*tc);
  double swb = ( 2.0*fcp - tc*fcpp)/(tc*tc*tc);
  double swc = -fc + (tc/2.0)*fcp - (tc*tc/12.0)*fcpp;

  sw1[i][j] = swa;
  sw2[i][j] = swb;
  sw3[i][j] = swa/3.0;
  sw4[i][j] = swb/4.0;
  sw5[i][j] = swc;

  sw1[j][i] = sw1[i][j];
  sw2[j][i] = sw2[i][j];
  sw3[j][i] = sw3[i][j];
  sw4[j][i] = sw4[i][j];
  sw5[j][i] = sw5[i][j];
}

