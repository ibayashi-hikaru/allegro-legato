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
   Contributing authors: Stefan Paquay (Brandeis University)
------------------------------------------------------------------------- */

#include "pair_yukawa_kokkos.h"
#include <cmath>
#include "kokkos.h"
#include "atom_kokkos.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "respa.h"
#include "memory_kokkos.h"
#include "error.h"
#include "atom_masks.h"

using namespace LAMMPS_NS;

#define KOKKOS_CUDA_MAX_THREADS 256
#define KOKKOS_CUDA_MIN_BLOCKS 8

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairYukawaKokkos<DeviceType>::PairYukawaKokkos(LAMMPS *lmp) : PairYukawa(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairYukawaKokkos<DeviceType>::~PairYukawaKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq,cutsq);
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairYukawaKokkos<DeviceType>::allocate()
{
  PairYukawa::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();
  k_params = Kokkos::DualView<params_yukawa**,
                              Kokkos::LayoutRight,DeviceType>(
                              "PairYukawa::params",n+1,n+1);

  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairYukawaKokkos<DeviceType>::init_style()
{
  PairYukawa::init_style();

  // error if rRESPA with inner levels

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style,"^respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
    if (respa)
      error->all(FLERR,"Cannot use Kokkos pair style with rRESPA inner/middle");
  }

  // irequest = neigh request made by parent class

  neighflag = lmp->kokkos->neighflag;
  int irequest = neighbor->nrequest - 1;

  neighbor->requests[irequest]->
    kokkos_host = std::is_same<DeviceType,LMPHostType>::value &&
    !std::is_same<DeviceType,LMPDeviceType>::value;
  neighbor->requests[irequest]->
    kokkos_device = std::is_same<DeviceType,LMPDeviceType>::value;

  if (neighflag == FULL) {
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->half = 0;
  } else if (neighflag == HALF || neighflag == HALFTHREAD) {
    neighbor->requests[irequest]->full = 0;
    neighbor->requests[irequest]->half = 1;
  } else {
    error->all(FLERR,"Cannot use chosen neighbor list style with yukawa/kk");
  }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
// Rewrite this.
template<class DeviceType>
double PairYukawaKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairYukawa::init_one(i,j);

  k_params.h_view(i,j).a      = a[i][j];
  k_params.h_view(i,j).offset = offset[i][j];
  k_params.h_view(i,j).cutsq  = cutone*cutone;
  k_params.h_view(j,i)        = k_params.h_view(i,j);

  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_params[i][j] = m_params[j][i] = k_params.h_view(i,j);
    m_cutsq[j][i] = m_cutsq[i][j] = cutone*cutone;
  }

  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();
  k_params.template modify<LMPHostType>();

  return cutone;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairYukawaKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;


  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  k_cutsq.template sync<DeviceType>();
  k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  c_x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];

  // loop over neighbors of my atoms

  EV_FLOAT ev = pair_compute<PairYukawaKokkos<DeviceType>,void >(
    this,(NeighListKokkos<DeviceType>*)list);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }
}



template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairYukawaKokkos<DeviceType>::
compute_fpair(const F_FLOAT& rsq, const int& i, const int&j,
              const int& itype, const int& jtype) const {
  (void) i;
  (void) j;
  const F_FLOAT rr     = sqrt(rsq);
  // Fetch the params either off the stack or from some mapped memory?
  const F_FLOAT aa     = STACKPARAMS ? m_params[itype][jtype].a
                                     : params(itype,jtype).a;

  // U   = a * exp(-kappa*r) / r
  // f   = (kappa * a * exp(-kappa*r) / r + a*exp(-kappa*r)/r^2)*grad(r)
  //     = (kappa + 1/r) * (a * exp(-kappa*r) / r)
  // f/r = (kappa + 1/r) * (a * exp(-kappa*r) / r^2)
  const F_FLOAT rinv = 1.0 / rr;
  const F_FLOAT rinv2 = rinv*rinv;
  const F_FLOAT screening = exp(-kappa*rr);
  const F_FLOAT forceyukawa = aa * screening * (kappa + rinv);
  const F_FLOAT fpair = forceyukawa * rinv2;

  return fpair;
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairYukawaKokkos<DeviceType>::
compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j,
              const int& itype, const int& jtype) const {
  (void) i;
  (void) j;
  const F_FLOAT rr     = sqrt(rsq);
  const F_FLOAT aa     = STACKPARAMS ? m_params[itype][jtype].a
                                     : params(itype,jtype).a;
  const F_FLOAT offset = STACKPARAMS ? m_params[itype][jtype].offset
                                     : params(itype,jtype).offset;

  // U   = a * exp(-kappa*r) / r
  // f   = (kappa * a * exp(-kappa*r) / r + a*exp(-kappa*r)/r^2)*grad(r)
  //     = (kappa + 1/r) * (a * exp(-kappa*r) / r)
  // f/r = (kappa + 1/r) * (a * exp(-kappa*r) / r^2)
  const F_FLOAT rinv = 1.0 / rr;
  const F_FLOAT screening = exp(-kappa*rr);

  return aa * screening * rinv - offset;
}


namespace LAMMPS_NS {
template class PairYukawaKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairYukawaKokkos<LMPHostType>;
#endif
}
