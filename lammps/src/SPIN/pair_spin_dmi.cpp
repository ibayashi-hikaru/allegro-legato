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

/* ------------------------------------------------------------------------
   Contributing authors: Julien Tranchida (SNL)
                         Aidan Thompson (SNL)

   Please cite the related publication:
   Tranchida, J., Plimpton, S. J., Thibaudeau, P., & Thompson, A. P. (2018).
   Massively parallel symplectic algorithm for coupled magnetic spin dynamics
   and molecular dynamics. Journal of Computational Physics.
------------------------------------------------------------------------- */

#include "pair_spin_dmi.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSpinDmi::~PairSpinDmi()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cut_spin_dmi);
    memory->destroy(DM);
    memory->destroy(v_dmx);
    memory->destroy(v_dmy);
    memory->destroy(v_dmz);
    memory->destroy(vmech_dmx);
    memory->destroy(vmech_dmy);
    memory->destroy(vmech_dmz);
    memory->destroy(cutsq);
    memory->destroy(emag);
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSpinDmi::settings(int narg, char **arg)
{
  PairSpin::settings(narg,arg);

  cut_spin_dmi_global = utils::numeric(FLERR,arg[0],false,lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++) {
      for (j = i+1; j <= atom->ntypes; j++) {
        if (setflag[i][j]) {
          cut_spin_dmi[i][j] = cut_spin_dmi_global;
        }
      }
    }
  }

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type spin pairs (only one for now)
------------------------------------------------------------------------- */

void PairSpinDmi::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  // check if args correct

  if (strcmp(arg[2],"dmi") != 0)
    error->all(FLERR,"Incorrect args in pair_style command");
  if (narg != 8)
    error->all(FLERR,"Incorrect args in pair_style command");

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  const double rij = utils::numeric(FLERR,arg[3],false,lmp);
  const double dm = utils::numeric(FLERR,arg[4],false,lmp);
  double dmx = utils::numeric(FLERR,arg[5],false,lmp);
  double dmy = utils::numeric(FLERR,arg[6],false,lmp);
  double dmz = utils::numeric(FLERR,arg[7],false,lmp);

  double inorm = 1.0/(dmx*dmx+dmy*dmy+dmz*dmz);
  dmx *= inorm;
  dmy *= inorm;
  dmz *= inorm;

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cut_spin_dmi[i][j] = rij;
      DM[i][j] = dm;
      v_dmx[i][j] = dmx * dm / hbar;
      v_dmy[i][j] = dmy * dm / hbar;
      v_dmz[i][j] = dmz * dm / hbar;
      vmech_dmx[i][j] = dmx * dm;
      vmech_dmy[i][j] = dmy * dm;
      vmech_dmz[i][j] = dmz * dm;
      setflag[i][j] = 1;
      count++;
    }
  }
  if (count == 0)
    error->all(FLERR,"Incorrect args in pair_style command");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSpinDmi::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  DM[j][i] = DM[i][j];
  v_dmx[j][i] = v_dmx[i][j];
  v_dmy[j][i] = v_dmy[i][j];
  v_dmz[j][i] = v_dmz[i][j];
  vmech_dmx[j][i] = vmech_dmx[i][j];
  vmech_dmy[j][i] = vmech_dmy[i][j];
  vmech_dmz[j][i] = vmech_dmz[i][j];
  cut_spin_dmi[j][i] = cut_spin_dmi[i][j];

  return cut_spin_dmi_global;
}

/* ----------------------------------------------------------------------
   extract the larger cutoff
------------------------------------------------------------------------- */

void *PairSpinDmi::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"cut") == 0) return (void *) &cut_spin_dmi_global;
  return nullptr;
}

/* ---------------------------------------------------------------------- */

void PairSpinDmi::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double evdwl, ecoul;
  double xi[3], eij[3];
  double delx,dely,delz;
  double spi[3], spj[3];
  double fi[3], fmi[3];
  double local_cut2;
  double rsq, inorm;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = ecoul = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **fm = atom->fm;
  double **sp = atom->sp;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // checking size of emag

  if (nlocal_max < nlocal) {                    // grow emag lists if necessary
    nlocal_max = nlocal;
    memory->grow(emag,nlocal_max,"pair/spin:emag");
  }

  // dmi computation
  // loop over all atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];

    jlist = firstneigh[i];
    jnum = numneigh[i];
    xi[0] = x[i][0];
    xi[1] = x[i][1];
    xi[2] = x[i][2];
    spi[0] = sp[i][0];
    spi[1] = sp[i][1];
    spi[2] = sp[i][2];
    emag[i] = 0.0;

    // loop on neighbors

    for (jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      spj[0] = sp[j][0];
      spj[1] = sp[j][1];
      spj[2] = sp[j][2];

      evdwl = 0.0;
      fi[0] = fi[1] = fi[2] = 0.0;
      fmi[0] = fmi[1] = fmi[2] = 0.0;

      delx = xi[0] - x[j][0];
      dely = xi[1] - x[j][1];
      delz = xi[2] - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      inorm = 1.0/sqrt(rsq);
      eij[0] = -inorm*delx;
      eij[1] = -inorm*dely;
      eij[2] = -inorm*delz;

      local_cut2 = cut_spin_dmi[itype][jtype]*cut_spin_dmi[itype][jtype];

      // compute dmi interaction

      if (rsq <= local_cut2) {
        compute_dmi(i,j,eij,fmi,spj);

        if (lattice_flag)
          compute_dmi_mech(i,j,rsq,eij,fi,spi,spj);

        if (eflag) {
          evdwl -= (spi[0]*fmi[0] + spi[1]*fmi[1] + spi[2]*fmi[2]);
          evdwl *= 0.5*hbar;
          emag[i] += evdwl;
        } else evdwl = 0.0;

        f[i][0] += fi[0];
        f[i][1] += fi[1];
        f[i][2] += fi[2];
        if (newton_pair || j < nlocal) {
          f[j][0] -= fi[0];
          f[j][1] -= fi[1];
          f[j][2] -= fi[2];
        }
        fm[i][0] += fmi[0];
        fm[i][1] += fmi[1];
        fm[i][2] += fmi[2];

        if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
            evdwl,ecoul,fi[0],fi[1],fi[2],delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   update the pair interactions fmi acting on the spin ii
------------------------------------------------------------------------- */

void PairSpinDmi::compute_single_pair(int ii, double fmi[3])
{
  int *type = atom->type;
  double **x = atom->x;
  double **sp = atom->sp;
  double local_cut2;
  double xi[3], eij[3];
  double delx,dely,delz;
  double spj[3];

  int j,jnum,itype,jtype,ntypes;
  int k,locflag;
  int *jlist,*numneigh,**firstneigh;

  double rsq, inorm;

  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // check if interaction applies to type of ii

  itype = type[ii];
  ntypes = atom->ntypes;
  locflag = 0;
  k = 1;
  while (k <= ntypes) {
    if (k <= itype) {
      if (setflag[k][itype] == 1) {
        locflag =1;
        break;
      }
      k++;
    } else if (k > itype) {
      if (setflag[itype][k] == 1) {
        locflag =1;
        break;
      }
      k++;
    } else error->all(FLERR,"Wrong type number");
  }

  // if interaction applies to type ii,
  // locflag = 1 and compute pair interaction

  if (locflag == 1) {

    xi[0] = x[ii][0];
    xi[1] = x[ii][1];
    xi[2] = x[ii][2];

    jlist = firstneigh[ii];
    jnum = numneigh[ii];

    for (int jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      spj[0] = sp[j][0];
      spj[1] = sp[j][1];
      spj[2] = sp[j][2];

      delx = xi[0] - x[j][0];
      dely = xi[1] - x[j][1];
      delz = xi[2] - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      inorm = 1.0/sqrt(rsq);
      eij[0] = -inorm*delx;
      eij[1] = -inorm*dely;
      eij[2] = -inorm*delz;

      local_cut2 = cut_spin_dmi[itype][jtype]*cut_spin_dmi[itype][jtype];

      if (rsq <= local_cut2) {
        compute_dmi(ii,j,eij,fmi,spj);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   compute the dmi interaction between spin i and spin j
------------------------------------------------------------------------- */

void PairSpinDmi::compute_dmi(int i, int j, double eij[3], double fmi[3], double spj[3])
{
  int *type = atom->type;
  int itype, jtype;
  double dmix, dmiy, dmiz;
  itype = type[i];
  jtype = type[j];

  dmix = eij[1]*v_dmz[itype][jtype] - eij[2]*v_dmy[itype][jtype];
  dmiy = eij[2]*v_dmx[itype][jtype] - eij[0]*v_dmz[itype][jtype];
  dmiz = eij[0]*v_dmy[itype][jtype] - eij[1]*v_dmx[itype][jtype];

  fmi[0] -= (dmiy*spj[2] - dmiz*spj[1]);
  fmi[1] -= (dmiz*spj[0] - dmix*spj[2]);
  fmi[2] -= (dmix*spj[1] - dmiy*spj[0]);
}

/* ----------------------------------------------------------------------
   compute the mechanical force due to the dmi interaction between atom i and atom j
------------------------------------------------------------------------- */

void PairSpinDmi::compute_dmi_mech(int i, int j, double rsq, double /*eij*/[3],
    double fi[3],  double spi[3], double spj[3])
{
  int *type = atom->type;
  int itype, jtype;
  double dmix,dmiy,dmiz;
  itype = type[i];
  jtype = type[j];
  double csx,csy,csz,cdmx,cdmy,cdmz,irij;

  irij = 1.0/sqrt(rsq);

  dmix = vmech_dmx[itype][jtype];
  dmiy = vmech_dmy[itype][jtype];
  dmiz = vmech_dmz[itype][jtype];

  csx = (spi[1]*spj[2] - spi[2]*spj[1]);
  csy = (spi[2]*spj[0] - spi[0]*spj[2]);
  csz = (spi[0]*spj[1] - spi[1]*spj[0]);

  cdmx = (dmiy*csz - dmiz*csy);
  cdmy = (dmiz*csx - dmix*csz);
  cdmz = (dmix*csy - dmiy*csz);

  fi[0] += 0.5*irij*cdmx;
  fi[1] += 0.5*irij*cdmy;
  fi[2] += 0.5*irij*cdmz;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSpinDmi::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cut_spin_dmi,n+1,n+1,"pair:cut_spin_dmi");
  memory->create(DM,n+1,n+1,"pair:DM");
  memory->create(v_dmx,n+1,n+1,"pair:DM_vector_x");
  memory->create(v_dmy,n+1,n+1,"pair:DM_vector_y");
  memory->create(v_dmz,n+1,n+1,"pair:DM_vector_z");
  memory->create(vmech_dmx,n+1,n+1,"pair:DMmech_vector_x");
  memory->create(vmech_dmy,n+1,n+1,"pair:DMmech_vector_y");
  memory->create(vmech_dmz,n+1,n+1,"pair:DMmech_vector_z");

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

}


/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSpinDmi::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&DM[i][j],sizeof(double),1,fp);
        fwrite(&v_dmx[i][j],sizeof(double),1,fp);
        fwrite(&v_dmy[i][j],sizeof(double),1,fp);
        fwrite(&v_dmz[i][j],sizeof(double),1,fp);
        fwrite(&vmech_dmx[i][j],sizeof(double),1,fp);
        fwrite(&vmech_dmy[i][j],sizeof(double),1,fp);
        fwrite(&vmech_dmz[i][j],sizeof(double),1,fp);
        fwrite(&cut_spin_dmi[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSpinDmi::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&DM[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&v_dmx[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&v_dmy[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&v_dmz[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&vmech_dmx[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&vmech_dmy[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&vmech_dmz[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut_spin_dmi[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&DM[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&v_dmx[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&v_dmy[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&v_dmz[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&vmech_dmx[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&vmech_dmy[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&vmech_dmz[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_spin_dmi[i][j],1,MPI_DOUBLE,0,world);
      }
    }
  }
}


/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSpinDmi::write_restart_settings(FILE *fp)
{
  fwrite(&cut_spin_dmi_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSpinDmi::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_spin_dmi_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_spin_dmi_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}
