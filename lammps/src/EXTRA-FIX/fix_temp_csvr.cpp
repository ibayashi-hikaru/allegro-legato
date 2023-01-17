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
   Contributing author: Axel Kohlmeyer (Temple U)
   Based on code by Paolo Raiteri (Curtin U) and Giovanni Bussi (SISSA)
------------------------------------------------------------------------- */

#include "fix_temp_csvr.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "modify.h"
#include "random_mars.h"
#include "update.h"
#include "variable.h"

#include <cstring>
#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
enum{CONSTANT,EQUAL};

static constexpr int PRNGSIZE = 98+2+3;
/* ---------------------------------------------------------------------- */

FixTempCSVR::FixTempCSVR(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  tstr(nullptr), id_temp(nullptr), random(nullptr)
{
  if (narg != 7) error->all(FLERR,"Illegal fix temp/csvr command");

  // CSVR thermostat should be applied every step

  restart_global = 1;
  nevery = 1;
  scalar_flag = 1;
  ecouple_flag = 1;
  global_freq = nevery;
  dynamic_group_allow = 1;
  extscalar = 1;

  tstr = nullptr;
  if (utils::strmatch(arg[3],"^v_")) {
    tstr = utils::strdup(arg[3]+2);
    tstyle = EQUAL;
  } else {
    t_start = utils::numeric(FLERR,arg[3],false,lmp);
    t_target = t_start;
    tstyle = CONSTANT;
  }

  t_stop = utils::numeric(FLERR,arg[4],false,lmp);
  t_period = utils::numeric(FLERR,arg[5],false,lmp);
  int seed = utils::inumeric(FLERR,arg[6],false,lmp);

  // error checks

  if (t_period <= 0.0) error->all(FLERR,"Illegal fix temp/csvr command");
  if (seed <= 0) error->all(FLERR,"Illegal fix temp/csvr command");

  random = new RanMars(lmp,seed + comm->me);

  // create a new compute temp style
  // id = fix-ID + temp, compute group = fix group

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} {} temp",id_temp,group->names[igroup]));
  tflag = 1;

  nmax = -1;
  energy = 0.0;
}

/* ---------------------------------------------------------------------- */

FixTempCSVR::~FixTempCSVR()
{
  delete [] tstr;

  // delete temperature if fix created it

  if (tflag) modify->delete_compute(id_temp);
  delete [] id_temp;

  delete random;
  nmax = -1;
}

/* ---------------------------------------------------------------------- */

int FixTempCSVR::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTempCSVR::init()
{
  // check variable

  if (tstr) {
    tvar = input->variable->find(tstr);
    if (tvar < 0)
      error->all(FLERR,"Variable name for fix temp/csvr does not exist");
    if (input->variable->equalstyle(tvar)) tstyle = EQUAL;
    else error->all(FLERR,"Variable for fix temp/csvr is invalid style");
  }

  int icompute = modify->find_compute(id_temp);
  if (icompute < 0)
    error->all(FLERR,"Temperature ID for fix temp/csvr does not exist");
  temperature = modify->compute[icompute];

  if (temperature->tempbias) which = BIAS;
  else which = NOBIAS;
}

/* ---------------------------------------------------------------------- */

void FixTempCSVR::end_of_step()
{
  // set current t_target
  // if variable temp, evaluate variable, wrap with clear/add

  double delta = update->ntimestep - update->beginstep;

  if (delta != 0.0) delta /= update->endstep - update->beginstep;
  if (tstyle == CONSTANT)
    t_target = t_start + delta * (t_stop-t_start);
  else {
    modify->clearstep_compute();
    t_target = input->variable->compute_equal(tvar);
    if (t_target < 0.0)
      error->one(FLERR,
                 "Fix temp/csvr variable returned negative temperature");
    modify->addstep_compute(update->ntimestep + nevery);
  }

  const double t_current = temperature->compute_scalar();
  const double efactor = 0.5 * temperature->dof * force->boltz;
  const double ekin_old = t_current * efactor;
  const double ekin_new = t_target * efactor;

  // there is nothing to do, if there are no degrees of freedom

  if (temperature->dof < 1) return;

  // compute velocity scaling factor on root node and broadcast

  double lamda;
  if (comm->me == 0) {
    lamda = resamplekin(ekin_old, ekin_new);
  }
  MPI_Bcast(&lamda,1,MPI_DOUBLE,0,world);

  double * const * const v = atom->v;
  const int * const mask = atom->mask;
  const int nlocal = atom->nlocal;

  if (which == NOBIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        v[i][0] *= lamda;
        v[i][1] *= lamda;
        v[i][2] *= lamda;
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        temperature->remove_bias(i,v[i]);
        v[i][0] *= lamda;
        v[i][1] *= lamda;
        v[i][2] *= lamda;
        temperature->restore_bias(i,v[i]);
      }
    }
  }

  // tally the kinetic energy transferred between heat bath and system

  energy += ekin_old * (1.0 - lamda*lamda);
}

/* ---------------------------------------------------------------------- */

int FixTempCSVR::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (tflag) {
      modify->delete_compute(id_temp);
      tflag = 0;
    }
    delete [] id_temp;
    id_temp = utils::strdup(arg[1]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find fix_modify temperature ID");
    temperature = modify->compute[icompute];

    if (temperature->tempflag == 0)
      error->all(FLERR,
                 "Fix_modify temperature ID does not compute temperature");
    if (temperature->igroup != igroup && comm->me == 0)
      error->warning(FLERR,"Group for fix_modify temp != fix group");
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

double FixTempCSVR::gamdev(const int ia)
{
  int j;
  double am,e,s,v1,v2,x,y;

  if (ia < 1) return 0.0;
  if (ia < 6) {
    x=1.0;
    for (j=1; j<=ia; j++)
      x *= random->uniform();

    // make certain, that -log() doesn't overflow.
    if (x < 2.2250759805e-308)
      x = 708.4;
    else
      x = -log(x);
  } else {
  restart:
    do {
      do {
        do {
          v1 = random->uniform();
          v2 = 2.0*random->uniform() - 1.0;
        } while (v1*v1 + v2*v2 > 1.0);

        y=v2/v1;
        am=ia-1;
        s=sqrt(2.0*am+1.0);
        x=s*y+am;
      } while (x <= 0.0);

      if (am*log(x/am)-s*y < -700 || v1<0.00001) {
        goto restart;
      }

      e=(1.0+y*y)*exp(am*log(x/am)-s*y);
    } while (random->uniform() > e);
  }
  return x;
}

/* -------------------------------------------------------------------
  returns the sum of n independent gaussian noises squared
  (i.e. equivalent to summing the square of the return values of nn
   calls to gasdev)
---------------------------------------------------------------------- */

double FixTempCSVR::sumnoises(int nn) {
  if (nn == 0) {
    return 0.0;
  } else if (nn == 1) {
    const double rr = random->gaussian();
    return rr*rr;
  } else if (nn % 2 == 0) {
    return 2.0 * gamdev(nn / 2);
  } else {
    const double rr = random->gaussian();
    return  2.0 * gamdev((nn-1) / 2) + rr*rr;
  }
}

/* -------------------------------------------------------------------
  returns the scaling factor for velocities to thermalize
  the system so it samples the canonical ensemble
---------------------------------------------------------------------- */

double FixTempCSVR::resamplekin(double ekin_old, double ekin_new) {
  const double tdof = temperature->dof;
  const double c1 = exp(-update->dt/t_period);
  const double c2 = (1.0-c1)*ekin_new/ekin_old/tdof;
  const double r1 = random->gaussian();
  const double r2 = sumnoises(tdof - 1);

  const double scale = c1 + c2*(r1*r1+r2) + 2.0*r1*sqrt(c1*c2);
  return sqrt(scale);
}

/* ---------------------------------------------------------------------- */

void FixTempCSVR::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

double FixTempCSVR::compute_scalar()
{
  return energy;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixTempCSVR::write_restart(FILE *fp)
{
  int nsize = PRNGSIZE*comm->nprocs+2; // pRNG state per proc + nprocs + energy
  double *list = nullptr;
  if (comm->me == 0) {
    list = new double[nsize];
    list[0] = energy;
    list[1] = comm->nprocs;
  }
  double state[PRNGSIZE];
  random->get_state(state);
  MPI_Gather(state,PRNGSIZE,MPI_DOUBLE,list+2,PRNGSIZE,MPI_DOUBLE,0,world);

  if (comm->me == 0) {
    int size = nsize * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),nsize,fp);
    delete[] list;
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixTempCSVR::restart(char *buf)
{
  double *list = (double *) buf;

  energy = list[0];
  int nprocs = (int) list[1];
  if (nprocs != comm->nprocs) {
    if (comm->me == 0)
      error->warning(FLERR,"Different number of procs. Cannot restore RNG state.");
  } else random->set_state(list+2+comm->me*PRNGSIZE);
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixTempCSVR::extract(const char *str, int &dim)
{
  dim=0;
  if (strcmp(str,"t_target") == 0) {
    return &t_target;
  }
  return nullptr;
}
