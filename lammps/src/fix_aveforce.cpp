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

#include "fix_aveforce.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "modify.h"
#include "region.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,CONSTANT,EQUAL};

/* ---------------------------------------------------------------------- */

FixAveForce::FixAveForce(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  xstr(nullptr), ystr(nullptr), zstr(nullptr), idregion(nullptr)
{
  if (narg < 6) error->all(FLERR,"Illegal fix aveforce command");

  dynamic_group_allow = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extvector = 1;
  respa_level_support = 1;
  ilevel_respa = nlevels_respa = 0;

  xstr = ystr = zstr = nullptr;

  if (utils::strmatch(arg[3],"^v_")) {
    xstr = utils::strdup(arg[3]+2);
  } else if (strcmp(arg[3],"NULL") == 0) {
    xstyle = NONE;
  } else {
    xvalue = utils::numeric(FLERR,arg[3],false,lmp);
    xstyle = CONSTANT;
  }
  if (utils::strmatch(arg[4],"^v_")) {
    ystr = utils::strdup(arg[4]+2);
  } else if (strcmp(arg[4],"NULL") == 0) {
    ystyle = NONE;
  } else {
    yvalue = utils::numeric(FLERR,arg[4],false,lmp);
    ystyle = CONSTANT;
  }
  if (utils::strmatch(arg[5],"^v_")) {
    zstr = utils::strdup(arg[5]+2);
  } else if (strcmp(arg[5],"NULL") == 0) {
    zstyle = NONE;
  } else {
    zvalue = utils::numeric(FLERR,arg[5],false,lmp);
    zstyle = CONSTANT;
  }

  // optional args

  iregion = -1;
  idregion = nullptr;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix aveforce command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix aveforce does not exist");
      idregion = utils::strdup(arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix aveforce command");

  }

  foriginal_all[0] = foriginal_all[1] =
    foriginal_all[2] = foriginal_all[3] = 0.0;
}

/* ---------------------------------------------------------------------- */

FixAveForce::~FixAveForce()
{
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
  delete [] idregion;
}

/* ---------------------------------------------------------------------- */

int FixAveForce::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAveForce::init()
{
  // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix aveforce does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else error->all(FLERR,"Variable for fix aveforce is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix aveforce does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else error->all(FLERR,"Variable for fix aveforce is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR,"Variable name for fix aveforce does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else error->all(FLERR,"Variable for fix aveforce is invalid style");
  }

  // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix aveforce does not exist");
  }

  if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL) varflag = EQUAL;
  else varflag = CONSTANT;

  if (utils::strmatch(update->integrate_style,"^respa")) {
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,nlevels_respa-1);
    else ilevel_respa = nlevels_respa-1;
  }
}

/* ---------------------------------------------------------------------- */

void FixAveForce::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style,"^verlet"))
    post_force(vflag);
  else
    for (int ilevel = 0; ilevel < nlevels_respa; ilevel++) {
      ((Respa *) update->integrate)->copy_flevel_f(ilevel);
      post_force_respa(vflag,ilevel,0);
      ((Respa *) update->integrate)->copy_f_flevel(ilevel);
    }
}

/* ---------------------------------------------------------------------- */

void FixAveForce::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAveForce::post_force(int /*vflag*/)
{
  // update region if necessary

  Region *region = nullptr;
  if (iregion >= 0) {
    region = domain->regions[iregion];
    region->prematch();
  }

  // sum forces on participating atoms

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double foriginal[4];
  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
      foriginal[0] += f[i][0];
      foriginal[1] += f[i][1];
      foriginal[2] += f[i][2];
      foriginal[3] += 1.0;
    }

  // average the force on participating atoms
  // add in requested amount, computed via variable evaluation if necessary
  // wrap variable evaluation with clear/add

  MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);

  int ncount = static_cast<int> (foriginal_all[3]);
  if (ncount == 0) return;

  if (varflag == EQUAL) {
    modify->clearstep_compute();
    if (xstyle == EQUAL) xvalue = input->variable->compute_equal(xvar);
    if (ystyle == EQUAL) yvalue = input->variable->compute_equal(yvar);
    if (zstyle == EQUAL) zvalue = input->variable->compute_equal(zvar);
    modify->addstep_compute(update->ntimestep + 1);
  }

  double fave[3];
  fave[0] = foriginal_all[0]/ncount + xvalue;
  fave[1] = foriginal_all[1]/ncount + yvalue;
  fave[2] = foriginal_all[2]/ncount + zvalue;

  // set force of all participating atoms to same value
  // only for active dimensions

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
      if (xstyle) f[i][0] = fave[0];
      if (ystyle) f[i][1] = fave[1];
      if (zstyle) f[i][2] = fave[2];
    }
}

/* ---------------------------------------------------------------------- */

void FixAveForce::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  // ave + extra force on selected RESPA level
  // just ave on all other levels

  if (ilevel == ilevel_respa) post_force(vflag);
  else {
    Region *region = nullptr;
    if (iregion >= 0) {
      region = domain->regions[iregion];
      region->prematch();
    }

    double **x = atom->x;
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    double foriginal[4];
    foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
        foriginal[0] += f[i][0];
        foriginal[1] += f[i][1];
        foriginal[2] += f[i][2];
        foriginal[3] += 1.0;
      }

    MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);

    int ncount = static_cast<int> (foriginal_all[3]);
    if (ncount == 0) return;

    double fave[3];
    fave[0] = foriginal_all[0]/ncount;
    fave[1] = foriginal_all[1]/ncount;
    fave[2] = foriginal_all[2]/ncount;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
        if (xstyle) f[i][0] = fave[0];
        if (ystyle) f[i][1] = fave[1];
        if (zstyle) f[i][2] = fave[2];
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixAveForce::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixAveForce::compute_vector(int n)
{
  return foriginal_all[n];
}
