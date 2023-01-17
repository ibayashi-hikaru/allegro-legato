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
   Contributing author: Carsten Svaneborg, science@zqex.dk
------------------------------------------------------------------------- */

#include "bond_harmonic_shift_cut.h"

#include <cmath>
#include "atom.h"
#include "neighbor.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondHarmonicShiftCut::BondHarmonicShiftCut(LAMMPS *lmp) : Bond(lmp) {}

/* ---------------------------------------------------------------------- */

BondHarmonicShiftCut::~BondHarmonicShiftCut()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(r0);
    memory->destroy(r1);
  }
}

/* ---------------------------------------------------------------------- */

void BondHarmonicShiftCut::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double delx,dely,delz,ebond,fbond;
  double rsq,r,dr,rk;

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

    if (r>r1[type]) continue;

    dr = r - r0[type];
    rk = k[type] * dr;

    // force & energy

    if (r > 0.0) fbond = -2.0*rk/r;
    else fbond = 0.0;

    if (eflag)
      ebond = k[type]*(dr*dr -(r0[type]-r1[type])*(r0[type]-r1[type]));

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

void BondHarmonicShiftCut::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(k ,    n+1,"bond:k");
  memory->create(r0,    n+1,"bond:r0");
  memory->create(r1,    n+1,"bond:r1");
  memory->create(setflag,n+1,"bond:setflag");

  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondHarmonicShiftCut::coeff(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nbondtypes,ilo,ihi,error);

  double Umin = utils::numeric(FLERR,arg[1],false,lmp);   // energy at minimum
  double r0_one = utils::numeric(FLERR,arg[2],false,lmp); // position of minimum
  double r1_one = utils::numeric(FLERR,arg[3],false,lmp);  // position where energy = 0 = cutoff
  if (r0_one == r1_one)
    error->all(FLERR,"Bond harmonic/shift/cut r0 and r1 must be different");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = Umin/((r0_one-r1_one)*(r0_one-r1_one));
    r0[i] = r0_one;
    r1[i] = r1_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondHarmonicShiftCut::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondHarmonicShiftCut::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r0[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r1[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondHarmonicShiftCut::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR,&k[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&r0[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&r1[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
  }
  MPI_Bcast(&k[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r1[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondHarmonicShiftCut::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++) {
    double d2 = (r0[i]-r1[i])*(r0[i]-r1[i]);
    fprintf(fp,"%d %g %g %g\n",i,k[i]*d2,r0[i],r1[i]);
  }
}

/* ---------------------------------------------------------------------- */

double BondHarmonicShiftCut::single(int type, double rsq, int /*i*/, int /*j*/,
                                    double &fforce)
{
  fforce = 0.0;
  double r = sqrt(rsq);

  if (r>r1[type]) return 0.0;

  double dr = r - r0[type];
  double dr2=r0[type]-r1[type];

  fforce = -2.0*k[type]*dr/r;
  return k[type]*(dr*dr - dr2*dr2);
}
