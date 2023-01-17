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

#include "fix_langevin_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "memory_kokkos.h"
#include "modify.h"
#include "update.h"
#include "variable.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
enum{CONSTANT,EQUAL,ATOM};
#define SINERTIA 0.4          // moment of inertia prefactor for sphere
#define EINERTIA 0.2          // moment of inertia prefactor for ellipsoid

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixLangevinKokkos<DeviceType>::FixLangevinKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixLangevin(lmp, narg, arg),rand_pool(seed + comm->me)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  int ntypes = atomKK->ntypes;

  // allocate per-type arrays for force prefactors
  memoryKK->create_kokkos(k_gfactor1,gfactor1,ntypes+1,"langevin:gfactor1");
  memoryKK->create_kokkos(k_gfactor2,gfactor2,ntypes+1,"langevin:gfactor2");
  memoryKK->create_kokkos(k_ratio,ratio,ntypes+1,"langevin:ratio");
  d_gfactor1 = k_gfactor1.template view<DeviceType>();
  h_gfactor1 = k_gfactor1.template view<LMPHostType>();
  d_gfactor2 = k_gfactor2.template view<DeviceType>();
  h_gfactor2 = k_gfactor2.template view<LMPHostType>();
  d_ratio = k_ratio.template view<DeviceType>();
  h_ratio = k_ratio.template view<LMPHostType>();

  // optional args
  for (int i = 1; i <= ntypes; i++) ratio[i] = 1.0;
  k_ratio.template modify<LMPHostType>();

  if (gjfflag) {
    grow_arrays(atomKK->nmax);
    atom->add_callback(Atom::GROW);
    // initialize franprev to zero
    for (int i = 0; i < atomKK->nlocal; i++) {
      franprev[i][0] = 0.0;
      franprev[i][1] = 0.0;
      franprev[i][2] = 0.0;
      lv[i][0] = 0.0;
      lv[i][1] = 0.0;
      lv[i][2] = 0.0;
    }
    k_franprev.template modify<LMPHostType>();
    k_lv.template modify<LMPHostType>();
  }
  if (zeroflag) {
    k_fsumall = tdual_double_1d_3n("langevin:fsumall");
    h_fsumall = k_fsumall.template view<LMPHostType>();
    d_fsumall = k_fsumall.template view<DeviceType>();
  }

  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read =  V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK;
  datamask_modify = F_MASK;

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixLangevinKokkos<DeviceType>::~FixLangevinKokkos()
{
  memoryKK->destroy_kokkos(k_gfactor1,gfactor1);
  memoryKK->destroy_kokkos(k_gfactor2,gfactor2);
  memoryKK->destroy_kokkos(k_ratio,ratio);
  memoryKK->destroy_kokkos(k_flangevin,flangevin);
  if (gjfflag) memoryKK->destroy_kokkos(k_franprev,franprev);
  if (gjfflag) memoryKK->destroy_kokkos(k_lv,lv);
  memoryKK->destroy_kokkos(k_tforce,tforce);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixLangevinKokkos<DeviceType>::init()
{
  FixLangevin::init();
  if (oflag)
    error->all(FLERR,"Fix langevin omega is not yet implemented with kokkos");
  if (ascale)
    error->all(FLERR,"Fix langevin angmom is not yet implemented with kokkos");
  if (gjfflag && tbiasflag)
    error->all(FLERR,"Fix langevin gjf + tbias is not yet implemented with kokkos");
  if (gjfflag && tbiasflag)
    error->warning(FLERR,"Fix langevin gjf + kokkos is not implemented with random gaussians");

  // prefactors are modified in the init
  k_gfactor1.template modify<LMPHostType>();
  k_gfactor2.template modify<LMPHostType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixLangevinKokkos<DeviceType>::grow_arrays(int nmax)
{
  memoryKK->grow_kokkos(k_franprev,franprev,nmax,3,"langevin:franprev");
  d_franprev = k_franprev.template view<DeviceType>();
  h_franprev = k_franprev.template view<LMPHostType>();
  memoryKK->grow_kokkos(k_lv,lv,nmax,3,"langevin:lv");
  d_lv = k_lv.template view<DeviceType>();
  h_lv = k_lv.template view<LMPHostType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixLangevinKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  atomKK->sync(execution_space,datamask_read);
  atomKK->modified(execution_space,datamask_modify);

  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  FixLangevinKokkosInitialIntegrateFunctor<DeviceType> functor(this);
  Kokkos::parallel_for(nlocal,functor);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixLangevinKokkos<DeviceType>::initial_integrate_item(int i) const
{
  if (mask[i] & groupbit) {
    f(i,0) /= gjfa;
    f(i,1) /= gjfa;
    f(i,2) /= gjfa;
    v(i,0) = d_lv(i,0);
    v(i,1) = d_lv(i,1);
    v(i,2) = d_lv(i,2);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixLangevinKokkos<DeviceType>::post_force(int /*vflag*/)
{
  // sync the device views which might have been modified on host
  atomKK->sync(execution_space,datamask_read);
  rmass = atomKK->k_rmass.view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  v = atomKK->k_v.template view<DeviceType>();
  type = atomKK->k_type.template view<DeviceType>();
  mask = atomKK->k_mask.template view<DeviceType>();

  k_gfactor1.template sync<DeviceType>();
  k_gfactor2.template sync<DeviceType>();
  k_ratio.template sync<DeviceType>();
  if (gjfflag) k_franprev.template sync<DeviceType>();
  if (gjfflag) k_lv.template sync<DeviceType>();

  boltz = force->boltz;
  dt = update->dt;
  mvv2e = force->mvv2e;
  ftm2v = force->ftm2v;
  fran_prop_const = sqrt(24.0*boltz/t_period/dt/mvv2e);

  compute_target(); // modifies tforce vector, hence sync here
  k_tforce.template sync<DeviceType>();

  double fsum[3],fsumall[3];
  bigint count;
  int nlocal = atomKK->nlocal;

  if (zeroflag) {
    fsum[0] = fsum[1] = fsum[2] = 0.0;
    count = group->count(igroup);
    if (count == 0)
      error->all(FLERR,"Cannot zero Langevin force of 0 atoms");
  }

  // reallocate flangevin if necessary
  if (tallyflag || osflag) {
    if (nlocal > maxatom1) {
      memoryKK->destroy_kokkos(k_flangevin,flangevin);
      maxatom1 = atomKK->nmax;
      memoryKK->create_kokkos(k_flangevin,flangevin,maxatom1,3,"langevin:flangevin");
      d_flangevin = k_flangevin.template view<DeviceType>();
      h_flangevin = k_flangevin.template view<LMPHostType>();
    }
  }

  // account for bias velocity
  if (tbiasflag == BIAS) {
    atomKK->sync(temperature->execution_space,temperature->datamask_read);
    temperature->compute_scalar();
    temperature->remove_bias_all(); // modifies velocities
    // if temeprature compute is kokkosized host-device comm won't be needed
    atomKK->modified(temperature->execution_space,temperature->datamask_modify);
    atomKK->sync(execution_space,temperature->datamask_modify);
  }

  // compute langevin force in parallel on the device
  FSUM s_fsum;
  if (tstyle == ATOM)
    if (gjfflag)
      if (tallyflag || osflag)
        if (tbiasflag == BIAS)
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,1,1,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,1,1,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,1,1,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,1,1,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
        else
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,1,0,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,1,0,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,1,0,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,1,0,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
      else
        if (tbiasflag == BIAS)
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,0,1,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,0,1,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,0,1,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,0,1,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
        else
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,0,0,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,0,0,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,0,0,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,1,0,0,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
    else
      if (tallyflag || osflag)
        if (tbiasflag == BIAS)
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,1,1,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,1,1,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,1,1,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,1,1,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
        else
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,1,0,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,1,0,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,1,0,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,1,0,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
      else
        if (tbiasflag == BIAS)
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,0,1,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,0,1,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,0,1,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,0,1,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
        else
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,0,0,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,0,0,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,0,0,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,1,0,0,0,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
  else
    if (gjfflag)
      if (tallyflag || osflag)
        if (tbiasflag == BIAS)
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,1,1,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,1,1,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,1,1,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,1,1,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
        else
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,1,0,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,1,0,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,1,0,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,1,0,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
      else
        if (tbiasflag == BIAS)
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,0,1,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,0,1,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,0,1,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,0,1,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
        else
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,0,0,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,0,0,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,0,0,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,1,0,0,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
    else
      if (tallyflag || osflag)
        if (tbiasflag == BIAS)
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,1,1,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,1,1,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,1,1,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,1,1,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
        else
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,1,0,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,1,0,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,1,0,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,1,0,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
      else
        if (tbiasflag == BIAS)
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,0,1,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,0,1,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,0,1,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,0,1,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
        else
          if (rmass.data())
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,0,0,1,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,0,0,1,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }
          else
            if (zeroflag) {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,0,0,0,1> post_functor(this);
              Kokkos::parallel_reduce(nlocal,post_functor,s_fsum);
            } else {
              FixLangevinKokkosPostForceFunctor<DeviceType,0,0,0,0,0,0> post_functor(this);
              Kokkos::parallel_for(nlocal,post_functor);
            }


  if (tbiasflag == BIAS) {
    atomKK->sync(temperature->execution_space,temperature->datamask_read);
    temperature->restore_bias_all(); // modifies velocities
    atomKK->modified(temperature->execution_space,temperature->datamask_modify);
    atomKK->sync(execution_space,temperature->datamask_modify);
  }

  // set modify flags for the views modified in post_force functor
  if (gjfflag) k_franprev.template modify<DeviceType>();
  if (gjfflag) k_lv.template modify<DeviceType>();
  if (tallyflag || osflag) k_flangevin.template modify<DeviceType>();

  // set total force to zero
  if (zeroflag) {
    fsum[0] = s_fsum.fx; fsum[1] = s_fsum.fy; fsum[2] = s_fsum.fz;
    MPI_Allreduce(fsum,fsumall,3,MPI_DOUBLE,MPI_SUM,world);
    h_fsumall(0) = fsumall[0]/count;
    h_fsumall(1) = fsumall[1]/count;
    h_fsumall(2) = fsumall[2]/count;
    k_fsumall.template modify<LMPHostType>();
    k_fsumall.template sync<DeviceType>();
    // set total force zero in parallel on the device
    FixLangevinKokkosZeroForceFunctor<DeviceType> zero_functor(this);
    Kokkos::parallel_for(nlocal,zero_functor);
  }
  // f is modified by both post_force and zero_force functors
  atomKK->modified(execution_space,datamask_modify);

  // thermostat omega and angmom
  //  if (oflag) omega_thermostat();
  //  if (ascale) angmom_thermostat();

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int Tp_TSTYLEATOM, int Tp_GJF, int Tp_TALLY,
         int Tp_BIAS, int Tp_RMASS, int Tp_ZERO>
KOKKOS_INLINE_FUNCTION
FSUM FixLangevinKokkos<DeviceType>::post_force_item(int i) const
{
  FSUM fsum;
  double fdrag[3],fran[3];
  double gamma1,gamma2;
  double fswap;
  double tsqrt_t = tsqrt;

  if (mask[i] & groupbit) {
    rand_type rand_gen = rand_pool.get_state();
    if (Tp_TSTYLEATOM) tsqrt_t = sqrt(d_tforce[i]);
    if (Tp_RMASS) {
      gamma1 = -rmass[i] / t_period / ftm2v;
      gamma2 = sqrt(rmass[i]) * fran_prop_const / ftm2v;
      gamma1 *= 1.0/d_ratio[type[i]];
      gamma2 *= 1.0/sqrt(d_ratio[type[i]]) * tsqrt_t;
    } else {
      gamma1 = d_gfactor1[type[i]];
      gamma2 = d_gfactor2[type[i]] * tsqrt_t;
    }

    fran[0] = gamma2 * (rand_gen.drand() - 0.5); //(random->uniform()-0.5);
    fran[1] = gamma2 * (rand_gen.drand() - 0.5); //(random->uniform()-0.5);
    fran[2] = gamma2 * (rand_gen.drand() - 0.5); //(random->uniform()-0.5);

    if (Tp_BIAS) {
      fdrag[0] = gamma1*v(i,0);
      fdrag[1] = gamma1*v(i,1);
      fdrag[2] = gamma1*v(i,2);
      if (v(i,0) == 0.0) fran[0] = 0.0;
      if (v(i,1) == 0.0) fran[1] = 0.0;
      if (v(i,2) == 0.0) fran[2] = 0.0;
    } else {
      fdrag[0] = gamma1*v(i,0);
      fdrag[1] = gamma1*v(i,1);
      fdrag[2] = gamma1*v(i,2);
    }

    if (Tp_GJF) {
      d_lv(i,0) = gjfsib*v(i,0);
      d_lv(i,1) = gjfsib*v(i,1);
      d_lv(i,2) = gjfsib*v(i,2);

      fswap = 0.5*(fran[0]+d_franprev(i,0));
      d_franprev(i,0) = fran[0];
      fran[0] = fswap;
      fswap = 0.5*(fran[1]+d_franprev(i,1));
      d_franprev(i,1) = fran[1];
      fran[1] = fswap;
      fswap = 0.5*(fran[2]+d_franprev(i,2));
      d_franprev(i,2) = fran[2];
      fran[2] = fswap;

      fdrag[0] *= gjfa;
      fdrag[1] *= gjfa;
      fdrag[2] *= gjfa;
      fran[0] *= gjfa;
      fran[1] *= gjfa;
      fran[2] *= gjfa;
      f(i,0) *= gjfa;
      f(i,1) *= gjfa;
      f(i,2) *= gjfa;
    }

    f(i,0) += fdrag[0] + fran[0];
    f(i,1) += fdrag[1] + fran[1];
    f(i,2) += fdrag[2] + fran[2];

    if (Tp_TALLY) {
      if (Tp_GJF) {
        fdrag[0] = gamma1*d_lv(i,0)/gjfsib/gjfsib;
        fdrag[1] = gamma1*d_lv(i,1)/gjfsib/gjfsib;
        fdrag[2] = gamma1*d_lv(i,2)/gjfsib/gjfsib;
        fswap = (2*fran[0]/gjfa - d_franprev(i,0))/gjfsib;
        fran[0] = fswap;
        fswap = (2*fran[1]/gjfa - d_franprev(i,1))/gjfsib;
        fran[1] = fswap;
        fswap = (2*fran[2]/gjfa - d_franprev(i,2))/gjfsib;
        fran[2] = fswap;
      }
      d_flangevin(i,0) = fdrag[0] + fran[0];
      d_flangevin(i,1) = fdrag[1] + fran[1];
      d_flangevin(i,2) = fdrag[2] + fran[2];
    }

    if (Tp_ZERO) {
      fsum.fx = fran[0];
      fsum.fy = fran[1];
      fsum.fz = fran[2];
    }
    rand_pool.free_state(rand_gen);
  }

  return fsum;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixLangevinKokkos<DeviceType>::zero_force_item(int i) const
{
  if (mask[i] & groupbit) {
    f(i,0) -= d_fsumall[0];
    f(i,1) -= d_fsumall[1];
    f(i,2) -= d_fsumall[2];
  }

}

/* ----------------------------------------------------------------------
   set current t_target and t_sqrt
   ------------------------------------------------------------------------- */

template<class DeviceType>
void FixLangevinKokkos<DeviceType>::compute_target()
{
  atomKK->sync(Host, MASK_MASK);
  mask = atomKK->k_mask.template view<DeviceType>();
  int nlocal = atomKK->nlocal;

  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  // if variable temp, evaluate variable, wrap with clear/add
  // reallocate tforce array if necessary

  if (tstyle == CONSTANT) {
    t_target = t_start + delta * (t_stop-t_start);
    tsqrt = sqrt(t_target);
  } else {
    modify->clearstep_compute();
    if (tstyle == EQUAL) {
      t_target = input->variable->compute_equal(tvar);
      if (t_target < 0.0)
        error->one(FLERR,"Fix langevin variable returned negative temperature");
      tsqrt = sqrt(t_target);
    } else {
      if (atom->nmax > maxatom2) {
        maxatom2 = atom->nmax;
        memoryKK->destroy_kokkos(k_tforce,tforce);
        memoryKK->create_kokkos(k_tforce,tforce,maxatom2,"langevin:tforce");
        d_tforce = k_tforce.template view<DeviceType>();
        h_tforce = k_tforce.template view<LMPHostType>();
      }
      input->variable->compute_atom(tvar,igroup,tforce,1,0); // tforce is modified on host
      k_tforce.template modify<LMPHostType>();
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
          if (h_tforce[i] < 0.0)
            error->one(FLERR,
                       "Fix langevin variable returned negative temperature");
    }
    modify->addstep_compute(update->ntimestep + 1);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixLangevinKokkos<DeviceType>::reset_dt()
{
  if (atomKK->mass) {
    for (int i = 1; i <= atomKK->ntypes; i++) {
      h_gfactor2[i] = sqrt(atomKK->mass[i]) *
        sqrt(24.0*force->boltz/t_period/update->dt/force->mvv2e) /
        force->ftm2v;
      h_gfactor2[i] *= 1.0/sqrt(h_ratio[i]);
    }
    k_gfactor2.template modify<LMPHostType>();
  }

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double FixLangevinKokkos<DeviceType>::compute_scalar()
{
  if (!tallyflag || flangevin == nullptr) return 0.0;

  v = atomKK->k_v.template view<DeviceType>();
  mask = atomKK->k_mask.template view<DeviceType>();

  // capture the very first energy transfer to thermal reservoir

  if (update->ntimestep == update->beginstep) {
    energy_onestep = 0.0;
    atomKK->sync(execution_space,V_MASK | MASK_MASK);
    int nlocal = atomKK->nlocal;
    k_flangevin.template sync<DeviceType>();
    FixLangevinKokkosTallyEnergyFunctor<DeviceType> scalar_functor(this);
    Kokkos::parallel_reduce(nlocal,scalar_functor,energy_onestep);
    energy = 0.5*energy_onestep*update->dt;
  }

  // convert midstep energy back to previous fullstep energy
  double energy_me = energy - 0.5*energy_onestep*update->dt;
  double energy_all;
  MPI_Allreduce(&energy_me,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
  return -energy_all;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double FixLangevinKokkos<DeviceType>::compute_energy_item(int i) const
{
  double my_energy = 0.0;
  if (mask[i] & groupbit)
    my_energy = d_flangevin(i,0)*v(i,0) + d_flangevin(i,1)*v(i,1) +
      d_flangevin(i,2)*v(i,2);
  return my_energy;
}

/* ----------------------------------------------------------------------
   tally energy transfer to thermal reservoir
   ------------------------------------------------------------------------- */

template<class DeviceType>
void FixLangevinKokkos<DeviceType>::end_of_step()
{
  if (!tallyflag && !gjfflag) return;

  v = atomKK->k_v.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  mask = atomKK->k_mask.template view<DeviceType>();

  atomKK->sync(execution_space,V_MASK | MASK_MASK);
  int nlocal = atomKK->nlocal;

  energy_onestep = 0.0;

  k_flangevin.template sync<DeviceType>();
  FixLangevinKokkosTallyEnergyFunctor<DeviceType> tally_functor(this);
  Kokkos::parallel_reduce(nlocal,tally_functor,energy_onestep);

  if (gjfflag) {
    if (rmass.data()) {
      FixLangevinKokkosEndOfStepFunctor<DeviceType,1> functor(this);
      Kokkos::parallel_for(nlocal,functor);
    } else {
      mass = atomKK->k_mass.view<DeviceType>();
      FixLangevinKokkosEndOfStepFunctor<DeviceType,0> functor(this);
      Kokkos::parallel_for(nlocal,functor);
    }
  }

  energy += energy_onestep*update->dt;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixLangevinKokkos<DeviceType>::end_of_step_item(int i) const {
  double tmp[3];
  if (mask[i] & groupbit) {
    const double dtfm = force->ftm2v * 0.5 * dt / mass[type[i]];
    tmp[0] = v(i,0);
    tmp[1] = v(i,1);
    tmp[2] = v(i,2);
    if (!osflag) {
      v(i,0) = d_lv(i,0);
      v(i,1) = d_lv(i,1);
      v(i,2) = d_lv(i,2);
    } else {
      v(i,0) = 0.5 * gjfsib * gjfsib * (v(i,0) + dtfm * f(i,0) / gjfa) +
                dtfm * 0.5 * (gjfsib * d_flangevin(i,0) - d_franprev(i,0)) +
                (gjfsib * gjfa * 0.5 + dt * 0.25 / t_period / gjfsib) * d_lv(i,0);
      v(i,1) = 0.5 * gjfsib * gjfsib * (v(i,1) + dtfm * f(i,1) / gjfa) +
                dtfm * 0.5 * (gjfsib * d_flangevin(i,0) - d_franprev(i,1)) +
                (gjfsib * gjfa * 0.5 + dt * 0.25 / t_period / gjfsib) * d_lv(i,1);
      v(i,2) = 0.5 * gjfsib * gjfsib * (v(i,2) + dtfm * f(i,2) / gjfa) +
                dtfm * 0.5 * (gjfsib * d_flangevin(i,0) - d_franprev(i,2)) +
                (gjfsib * gjfa * 0.5 + dt * 0.25 / t_period / gjfsib) * d_lv(i,2);
    }
    d_lv(i,0) = tmp[0];
    d_lv(i,1) = tmp[1];
    d_lv(i,2) = tmp[2];
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixLangevinKokkos<DeviceType>::end_of_step_rmass_item(int i) const
{
  double tmp[3];
  if (mask[i] & groupbit) {
    const double dtfm = force->ftm2v * 0.5 * dt / rmass[i];
    tmp[0] = v(i,0);
    tmp[1] = v(i,1);
    tmp[2] = v(i,2);
    if (!osflag) {
      v(i,0) = d_lv(i,0);
      v(i,1) = d_lv(i,1);
      v(i,2) = d_lv(i,2);
    } else {
      v(i,0) = 0.5 * gjfsib * gjfsib * (v(i,0) + dtfm * f(i,0) / gjfa) +
                dtfm * 0.5 * (gjfsib * d_flangevin(i,0) - d_franprev(i,0)) +
                (gjfsib * gjfa * 0.5 + dt * 0.25 / t_period / gjfsib) * d_lv(i,0);
      v(i,1) = 0.5 * gjfsib * gjfsib * (v(i,1) + dtfm * f(i,1) / gjfa) +
                dtfm * 0.5 * (gjfsib * d_flangevin(i,1) - d_franprev(i,1)) +
                (gjfsib * gjfa * 0.5 + dt * 0.25 / t_period / gjfsib) * d_lv(i,1);
      v(i,2) = 0.5 * gjfsib * gjfsib * (v(i,2) + dtfm * f(i,2) / gjfa) +
                dtfm * 0.5 * (gjfsib * d_flangevin(i,2) - d_franprev(i,2)) +
                (gjfsib * gjfa * 0.5 + dt * 0.25 / t_period / gjfsib) * d_lv(i,2);
    }
    d_lv(i,0) = tmp[0];
    d_lv(i,1) = tmp[1];
    d_lv(i,2) = tmp[2];
  }
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
   ------------------------------------------------------------------------- */

template<class DeviceType>
void FixLangevinKokkos<DeviceType>::copy_arrays(int i, int j, int /*delflag*/)
{
  h_franprev(j,0) = h_franprev(i,0);
  h_franprev(j,1) = h_franprev(i,1);
  h_franprev(j,2) = h_franprev(i,2);
  h_lv(j,0) = h_lv(i,0);
  h_lv(j,1) = h_lv(i,1);
  h_lv(j,2) = h_lv(i,2);

  k_franprev.template modify<LMPHostType>();
  k_lv.template modify<LMPHostType>();

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixLangevinKokkos<DeviceType>::cleanup_copy()
{
  random = nullptr;
  tstr = nullptr;
  gfactor1 = nullptr;
  gfactor2 = nullptr;
  ratio = nullptr;
  id_temp = nullptr;
  flangevin = nullptr;
  tforce = nullptr;
  gjfflag = 0;
  franprev = nullptr;
  lv = nullptr;
  id = style = nullptr;
  vatom = nullptr;
}

namespace LAMMPS_NS {
template class FixLangevinKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixLangevinKokkos<LMPHostType>;
#endif
}

