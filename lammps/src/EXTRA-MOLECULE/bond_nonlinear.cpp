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

#include "bond_nonlinear.h"

#include <cmath>
#include <cstring>
#include "atom.h"
#include "neighbor.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondNonlinear::BondNonlinear(LAMMPS *lmp) : Bond(lmp) {}

/* ---------------------------------------------------------------------- */

BondNonlinear::~BondNonlinear()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(epsilon);
    memory->destroy(r0);
    memory->destroy(lamda);
  }
}

/* ---------------------------------------------------------------------- */

void BondNonlinear::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double delx,dely,delz,ebond,fbond;
  double rsq,r,dr,drsq,lamdasq,denom,denomsq;

  ebond = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);
    dr = r - r0[type];
    drsq = dr*dr;
    lamdasq = lamda[type]*lamda[type];
    denom = lamdasq - drsq;
    denomsq = denom*denom;

    // force & energy

    fbond = -epsilon[type]/r * 2.0*dr*lamdasq/denomsq;
    if (eflag) ebond = epsilon[type] * drsq / denom;

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond;
      f[i1][1] += dely*fbond;
      f[i1][2] += delz*fbond;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond;
      f[i2][1] -= dely*fbond;
      f[i2][2] -= delz*fbond;
    }

    if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondNonlinear::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(epsilon,n+1,"bond:epsilon");
  memory->create(r0,n+1,"bond:r0");
  memory->create(lamda,n+1,"bond:lamda");
  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void BondNonlinear::coeff(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nbondtypes,ilo,ihi,error);

  double epsilon_one = utils::numeric(FLERR,arg[1],false,lmp);
  double r0_one = utils::numeric(FLERR,arg[2],false,lmp);
  double lamda_one = utils::numeric(FLERR,arg[3],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    epsilon[i] = epsilon_one;
    r0[i] = r0_one;
    lamda[i] = lamda_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ---------------------------------------------------------------------- */

double BondNonlinear::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void BondNonlinear::write_restart(FILE *fp)
{
  fwrite(&epsilon[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r0[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&lamda[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void BondNonlinear::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR,&epsilon[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&r0[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&lamda[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
  }
  MPI_Bcast(&epsilon[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&lamda[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondNonlinear::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g\n",i,epsilon[i],r0[i],lamda[i]);
}

/* ---------------------------------------------------------------------- */

double BondNonlinear::single(int type, double rsq, int /*i*/, int /*j*/,
                             double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  double drsq = dr*dr;
  double lamdasq = lamda[type]*lamda[type];
  double denom = lamdasq - drsq;
  double denomsq = denom*denom;
  fforce = -epsilon[type]/r * 2.0*dr*lamdasq/denomsq;
  return epsilon[type] * drsq / denom;
}

/* ---------------------------------------------------------------------- */

void *BondNonlinear::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str,"epsilon")==0) return (void*) epsilon;
  if (strcmp(str,"r0")==0) return (void*) r0;
  return nullptr;
}
