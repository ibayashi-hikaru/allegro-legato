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
   Contributing author: Tod A Pascal (Caltech)
------------------------------------------------------------------------- */

#include "angle_cosine_periodic.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "neighbor.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#define SMALL 0.001

/* ---------------------------------------------------------------------- */

AngleCosinePeriodic::AngleCosinePeriodic(LAMMPS *lmp) : Angle(lmp) {}

/* ---------------------------------------------------------------------- */

AngleCosinePeriodic::~AngleCosinePeriodic()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(b);
    memory->destroy(multiplicity);
  }
}

/* ---------------------------------------------------------------------- */

void AngleCosinePeriodic::compute(int eflag, int vflag)
{
  int i,i1,i2,i3,n,m,type,b_factor;
  double delx1,dely1,delz1,delx2,dely2,delz2;
  double eangle,f1[3],f3[3];
  double rsq1,rsq2,r1,r2,c,a,a11,a12,a22;
  double tn,tn_1,tn_2,un,un_1,un_2;

  eangle = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];

    // 1st bond

    delx1 = x[i1][0] - x[i2][0];
    dely1 = x[i1][1] - x[i2][1];
    delz1 = x[i1][2] - x[i2][2];

    rsq1 = delx1*delx1 + dely1*dely1 + delz1*delz1;
    r1 = sqrt(rsq1);

    // 2nd bond

    delx2 = x[i3][0] - x[i2][0];
    dely2 = x[i3][1] - x[i2][1];
    delz2 = x[i3][2] - x[i2][2];

    rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
    r2 = sqrt(rsq2);

    // c = cosine of angle

    c = delx1*delx2 + dely1*dely2 + delz1*delz2;
    c /= r1*r2;
    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    m = multiplicity[type];
    b_factor = b[type];

    // cos(n*x) = Tn(cos(x))
    // Tn(x) = Chebyshev polynomials of the first kind: T_0 = 1, T_1 = x, ...
    // recurrence relationship:
    // Tn(x) = 2*x*T[n-1](x) - T[n-2](x) where T[-1](x) = 0
    // also, dTn(x)/dx = n*U[n-1](x)
    // where Un(x) = 2*x*U[n-1](x) - U[n-2](x) and U[-1](x) = 0
    // finally need to handle special case for n = 1

    tn = 1.0;
    tn_1 = 1.0;
    tn_2 = 0.0;
    un = 1.0;
    un_1 = 2.0;
    un_2 = 0.0;

    // force & energy

    tn_2 = c;
    for (i = 1; i <= m; i++) {
      tn = 2*c*tn_1 - tn_2;
      tn_2 = tn_1;
      tn_1 = tn;
    }

    for (i = 2; i <= m; i++) {
      un = 2*c*un_1 - un_2;
      un_2 = un_1;
      un_1 = un;
    }
    tn = b_factor*powsign(m)*tn;
    un = b_factor*powsign(m)*m*un;

    if (eflag) eangle = 2*k[type]*(1.0 - tn);

    a = -k[type]*un;
    a11 = a*c / rsq1;
    a12 = -a / (r1*r2);
    a22 = a*c / rsq2;

    f1[0] = a11*delx1 + a12*delx2;
    f1[1] = a11*dely1 + a12*dely2;
    f1[2] = a11*delz1 + a12*delz2;
    f3[0] = a22*delx2 + a12*delx1;
    f3[1] = a22*dely2 + a12*dely1;
    f3[2] = a22*delz2 + a12*delz1;

    // apply force to each of 3 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= f1[0] + f3[0];
      f[i2][1] -= f1[1] + f3[1];
      f[i2][2] -= f1[2] + f3[2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }

    if (evflag) ev_tally(i1,i2,i3,nlocal,newton_bond,eangle,f1,f3,
                         delx1,dely1,delz1,delx2,dely2,delz2);
  }
}

/* ---------------------------------------------------------------------- */

void AngleCosinePeriodic::allocate()
{
  allocated = 1;
  int n = atom->nangletypes;

  memory->create(k,n+1,"angle:k");
  memory->create(multiplicity,n+1,"angle:multiplicity");
  memory->create(b,n+1,"angle:b");

  memory->create(setflag,n+1,"angle:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void AngleCosinePeriodic::coeff(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR,"Incorrect args for angle coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nangletypes,ilo,ihi,error);

  double c_one = utils::numeric(FLERR,arg[1],false,lmp);
  int b_one = utils::inumeric(FLERR,arg[2],false,lmp);
  int n_one = utils::inumeric(FLERR,arg[3],false,lmp);
  if (n_one <= 0) error->all(FLERR,"Incorrect args for angle coefficients");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = c_one/(n_one*n_one);
    b[i] = b_one;
    multiplicity[i] = n_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for angle coefficients");
}

/* ---------------------------------------------------------------------- */

double AngleCosinePeriodic::equilibrium_angle(int i)
{
  return MY_PI*(1.0 - ((b[i]>0) ? 0.0 : (1.0/static_cast<double>(multiplicity[i]))));
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void AngleCosinePeriodic::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&b[1],sizeof(int),atom->nangletypes,fp);
  fwrite(&multiplicity[1],sizeof(int),atom->nangletypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void AngleCosinePeriodic::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR,&k[1],sizeof(double),atom->nangletypes,fp,nullptr,error);
    utils::sfread(FLERR,&b[1],sizeof(int),atom->nangletypes,fp,nullptr,error);
    utils::sfread(FLERR,&multiplicity[1],sizeof(int),atom->nangletypes,fp,nullptr,error);
  }

  MPI_Bcast(&k[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&b[1],atom->nangletypes,MPI_INT,0,world);
  MPI_Bcast(&multiplicity[1],atom->nangletypes,MPI_INT,0,world);
  for (int i = 1; i <= atom->nangletypes; i++) setflag[i] = 1;
}


/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void AngleCosinePeriodic::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nangletypes; i++) {
    int m = multiplicity[i];
    fprintf(fp,"%d %g %d %d\n",i,k[i]*m*m,b[i],m);
  }
}

/* ---------------------------------------------------------------------- */

double AngleCosinePeriodic::single(int type, int i1, int i2, int i3)
{
  double **x = atom->x;

  double delx1 = x[i1][0] - x[i2][0];
  double dely1 = x[i1][1] - x[i2][1];
  double delz1 = x[i1][2] - x[i2][2];
  domain->minimum_image(delx1,dely1,delz1);
  double r1 = sqrt(delx1*delx1 + dely1*dely1 + delz1*delz1);

  double delx2 = x[i3][0] - x[i2][0];
  double dely2 = x[i3][1] - x[i2][1];
  double delz2 = x[i3][2] - x[i2][2];
  domain->minimum_image(delx2,dely2,delz2);
  double r2 = sqrt(delx2*delx2 + dely2*dely2 + delz2*delz2);

  double c = delx1*delx2 + dely1*dely2 + delz1*delz2;
  c /= r1*r2;
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;

  c = cos(acos(c)*multiplicity[type]);
  return 2.0*k[type]*(1.0-b[type]*powsign(multiplicity[type])*c);
}
