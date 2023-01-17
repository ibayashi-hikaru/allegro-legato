// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "fix_nvt_sllod_omp.h"

#include "atom.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fix_deform.h"
#include "group.h"
#include "math_extra.h"
#include "modify.h"

#include <cstring>

#include "omp_compat.h"

using namespace LAMMPS_NS;
using namespace FixConst;

typedef struct { double x,y,z; } dbl3_t;

/* ---------------------------------------------------------------------- */

FixNVTSllodOMP::FixNVTSllodOMP(LAMMPS *lmp, int narg, char **arg) :
  FixNHOMP(lmp, narg, arg)
{
  if (!tstat_flag)
    error->all(FLERR,"Temperature control must be used with fix nvt/sllod");
  if (pstat_flag)
    error->all(FLERR,"Pressure control can not be used with fix nvt/sllod");

  // default values

  if (mtchain_default_flag) mtchain = 1;


  // create a new compute temp style
  // id = fix-ID + temp

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} {} temp/deform",
                                  id_temp,group->names[igroup]));
  tcomputeflag = 1;
}

/* ---------------------------------------------------------------------- */

void FixNVTSllodOMP::init()
{
  FixNHOMP::init();

  if (!temperature->tempbias)
    error->all(FLERR,"Temperature for fix nvt/sllod/omp does not have a bias");

  nondeformbias = 0;
  if (strcmp(temperature->style,"temp/deform") != 0) nondeformbias = 1;

  // check fix deform remap settings

  int i;
  for (i = 0; i < modify->nfix; i++)
    if (utils::strmatch(modify->fix[i]->style,"^deform")) {
      if (((FixDeform *) modify->fix[i])->remapflag != Domain::V_REMAP)
        error->all(FLERR,"Using fix nvt/sllod/omp with inconsistent fix "
                   "deform remap option");
      break;
    }
  if (i == modify->nfix)
    error->all(FLERR,"Using fix nvt/sllod/omp with no fix deform defined");
}

/* ----------------------------------------------------------------------
   perform half-step scaling of velocities
-----------------------------------------------------------------------*/

void FixNVTSllodOMP::nh_v_temp()
{
  // remove and restore bias = streaming velocity = Hrate*lamda + Hratelo
  // thermostat thermal velocity only
  // vdelu = SLLOD correction = Hrate*Hinv*vthermal
  // for non temp/deform BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias

  dbl3_t * _noalias const v = (dbl3_t *) atom->v[0];
  const int * _noalias const mask = atom->mask;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;

  if (nondeformbias) temperature->compute_scalar();

  double h_two[6];
  MathExtra::multiply_shape_shape(domain->h_rate,domain->h_inv,h_two);

#if defined(_OPENMP)
#pragma omp parallel for LMP_DEFAULT_NONE LMP_SHARED(h_two) schedule(static)
#endif
  for (int i = 0; i < nlocal; i++) {
    double vdelu0,vdelu1,vdelu2,buf[3];
    if (mask[i] & groupbit) {
      vdelu0 = h_two[0]*v[i].x + h_two[5]*v[i].y + h_two[4]*v[i].z;
      vdelu1 = h_two[1]*v[i].y + h_two[3]*v[i].z;
      vdelu2 = h_two[2]*v[i].z;
      temperature->remove_bias_thr(i,&v[i].x,buf);
      v[i].x = v[i].x*factor_eta - dthalf*vdelu0;
      v[i].y = v[i].y*factor_eta - dthalf*vdelu1;
      v[i].z = v[i].z*factor_eta - dthalf*vdelu2;
      temperature->restore_bias_thr(i,&v[i].x,buf);
    }
  }
}
