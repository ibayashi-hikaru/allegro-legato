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

#include "fix_controller.h"

#include "arg_info.h"
#include "compute.h"
#include "error.h"
#include "input.h"
#include "modify.h"
#include "update.h"
#include "variable.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixController::FixController(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  pvID(nullptr), cvID(nullptr)
{
  if (narg != 11) error->all(FLERR,"Illegal fix controller command");

  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extvector = 0;

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix controller command");

  alpha = utils::numeric(FLERR,arg[4],false,lmp);
  kp = utils::numeric(FLERR,arg[5],false,lmp);
  ki = utils::numeric(FLERR,arg[6],false,lmp);
  kd = utils::numeric(FLERR,arg[7],false,lmp);

  // process variable arg

  ArgInfo argi(arg[8]);
  if ((argi.get_type() == ArgInfo::UNKNOWN)
      || (argi.get_type() == ArgInfo::NONE)
      || (argi.get_dim() != 0))
    error->all(FLERR,"Illegal fix controller command");

  pvwhich = argi.get_type();
  pvindex = argi.get_index1();
  pvID = argi.copy_name();

  // setpoint arg

  int iarg=9;
  setpoint = utils::numeric(FLERR,arg[iarg],false,lmp);
  iarg++;

  // control variable arg

  cvID = utils::strdup(arg[iarg]);

  // error check

  if (pvwhich == ArgInfo::COMPUTE) {
    int icompute = modify->find_compute(pvID);
    if (icompute < 0)
      error->all(FLERR,"Compute ID for fix controller does not exist");
    Compute *c = modify->compute[icompute];
    int flag = 0;
    if (c->scalar_flag && pvindex == 0) flag = 1;
    else if (c->vector_flag && pvindex > 0) flag = 1;
    if (!flag) error->all(FLERR,"Fix controller compute does not "
                          "calculate a global scalar or vector");
    if (pvindex && pvindex > c->size_vector)
      error->all(FLERR,"Fix controller compute vector is "
                 "accessed out-of-range");
  } else if (pvwhich == ArgInfo::FIX) {
    int ifix = modify->find_fix(pvID);
    if (ifix < 0)
      error->all(FLERR,"Fix ID for fix controller does not exist");
    Fix *f = modify->fix[ifix];
    int flag = 0;
    if (f->scalar_flag && pvindex == 0) flag = 1;
    else if (f->vector_flag && pvindex > 0) flag = 1;
    if (!flag) error->all(FLERR,"Fix controller fix does not "
                          "calculate a global scalar or vector");
    if (pvindex && pvindex > f->size_vector)
      error->all(FLERR,"Fix controller fix vector is accessed out-of-range");
  } else if (pvwhich == ArgInfo::VARIABLE) {
    int ivariable = input->variable->find(pvID);
    if (ivariable < 0)
      error->all(FLERR,"Variable name for fix controller does not exist");
    if (input->variable->equalstyle(ivariable) == 0)
      error->all(FLERR,"Fix controller variable is not equal-style variable");
  }

  int ivariable = input->variable->find(cvID);
  if (ivariable < 0)
    error->all(FLERR,"Variable name for fix controller does not exist");
  if (input->variable->internalstyle(ivariable) == 0)
    error->all(FLERR,"Fix controller variable is not internal-style variable");
  control = input->variable->compute_equal(ivariable);

  firsttime = 1;
}

/* ---------------------------------------------------------------------- */

FixController::~FixController()
{
  delete [] pvID;
  delete [] cvID;
}

/* ---------------------------------------------------------------------- */

int FixController::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixController::init()
{
  if (pvwhich == ArgInfo::COMPUTE) {
    int icompute = modify->find_compute(pvID);
    if (icompute < 0)
      error->all(FLERR,"Compute ID for fix controller does not exist");
    pcompute = modify->compute[icompute];

  } else if (pvwhich == ArgInfo::FIX) {
    int ifix = modify->find_fix(pvID);
    if (ifix < 0) error->all(FLERR,"Fix ID for fix controller does not exist");
    pfix = modify->fix[ifix];

  } else if (pvwhich == ArgInfo::VARIABLE) {
    pvar = input->variable->find(pvID);
    if (pvar < 0)
      error->all(FLERR,"Variable name for fix controller does not exist");
  }

  cvar = input->variable->find(cvID);
  if (cvar < 0)
    error->all(FLERR,"Variable name for fix controller does not exist");

  // set sampling time

  tau = nevery * update->dt;
}

/* ---------------------------------------------------------------------- */

void FixController::end_of_step()
{
  // current value of pv = invocation of compute,fix,variable
  // compute/fix/variable may invoke computes so wrap with clear/add

  modify->clearstep_compute();

  // invoke compute if not previously invoked

  double current = 0.0;

  if (pvwhich == ArgInfo::COMPUTE) {
    if (pvindex == 0) {
      if (!(pcompute->invoked_flag & Compute::INVOKED_SCALAR)) {
        pcompute->compute_scalar();
        pcompute->invoked_flag |= Compute::INVOKED_SCALAR;
      }
      current = pcompute->scalar;
    } else {
      if (!(pcompute->invoked_flag & Compute::INVOKED_VECTOR)) {
        pcompute->compute_vector();
        pcompute->invoked_flag |= Compute::INVOKED_VECTOR;
      }
      current = pcompute->vector[pvindex-1];
    }

  // access fix field, guaranteed to be ready

  } else if (pvwhich == ArgInfo::FIX) {
    if (pvindex == 0) current = pfix->compute_scalar();
    else current = pfix->compute_vector(pvindex-1);

  // evaluate equal-style variable

  } else if (pvwhich == ArgInfo::VARIABLE) {
    current = input->variable->compute_equal(pvar);
  }

  modify->addstep_compute(update->ntimestep + nevery);

  // new control var = f(old value, current process var, setpoint)
  // cv = cvold -kp*err -ki*sumerr -kd*deltaerr
  // note: this deviates from standard notation, which is
  // cv = kp*err +ki*sumerr +kd*deltaerr
  // the difference is in the sign and the time integral

  err = current - setpoint;

  if (firsttime) {
    firsttime = 0;
    deltaerr = sumerr = 0.0;
  } else {
    deltaerr = err - olderr;
    sumerr += err;
  }

  // 3 terms of PID equation

  control += -kp * alpha * tau * err;
  control += -ki * alpha * tau * tau * sumerr;
  control += -kd * alpha * deltaerr;
  olderr = err;

  // reset control variable

  input->variable->internal_set(cvar,control);
}

/* ---------------------------------------------------------------------- */

void FixController::reset_dt()
{
  tau = nevery * update->dt;
}

/* ----------------------------------------------------------------------
   return 3 terms of PID controller at last invocation of end_of_step()
------------------------------------------------------------------------- */

double FixController::compute_vector(int n)
{
  if (n == 0) return (-kp * alpha * tau * err);
  else if (n == 1) return (-ki * alpha * tau * tau * sumerr);
  else return (-kd * alpha * deltaerr);
}
