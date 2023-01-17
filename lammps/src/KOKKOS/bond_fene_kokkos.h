/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef BOND_CLASS
// clang-format off
BondStyle(fene/kk,BondFENEKokkos<LMPDeviceType>);
BondStyle(fene/kk/device,BondFENEKokkos<LMPDeviceType>);
BondStyle(fene/kk/host,BondFENEKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_BOND_FENE_KOKKOS_H
#define LMP_BOND_FENE_KOKKOS_H

#include "bond_fene.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<int NEWTON_BOND, int EVFLAG>
struct TagBondFENECompute{};

template<class DeviceType>
class BondFENEKokkos : public BondFENE {
 public:
  typedef DeviceType device_type;
  typedef EV_FLOAT value_type;
  typedef ArrayTypes<DeviceType> AT;

  BondFENEKokkos(class LAMMPS *);
  virtual ~BondFENEKokkos();
  void compute(int, int);
  void coeff(int, char **);
  void read_restart(FILE *);

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagBondFENECompute<NEWTON_BOND,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagBondFENECompute<NEWTON_BOND,EVFLAG>, const int&) const;

  //template<int NEWTON_BOND>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &ebond, const F_FLOAT &fbond, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

 protected:

  class NeighborKokkos *neighborKK;

  typename ArrayTypes<DeviceType>::t_x_array_randomread x;
  typename ArrayTypes<DeviceType>::t_f_array f;
  typename ArrayTypes<DeviceType>::t_int_2d bondlist;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename ArrayTypes<DeviceType>::t_efloat_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_virial_array d_vatom;

  DAT::tdual_int_scalar k_warning_flag;
  typename AT::t_int_scalar d_warning_flag;
  HAT::t_int_scalar h_warning_flag;

  DAT::tdual_int_scalar k_error_flag;
  typename AT::t_int_scalar d_error_flag;
  HAT::t_int_scalar h_error_flag;

  int nlocal,newton_bond;
  int eflag,vflag;

  DAT::tdual_ffloat_1d k_k;
  DAT::tdual_ffloat_1d k_r0;
  DAT::tdual_ffloat_1d k_epsilon;
  DAT::tdual_ffloat_1d k_sigma;

  typename AT::t_ffloat_1d d_k;
  typename AT::t_ffloat_1d d_r0;
  typename AT::t_ffloat_1d d_epsilon;
  typename AT::t_ffloat_1d d_sigma;

  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

W: FENE bond too long

A FENE bond has stretched dangerously far.  It's interaction strength
will be truncated to attempt to prevent the bond from blowing up.

E: Bad FENE bond

Two atoms in a FENE bond have become so far apart that the bond cannot
be computed.

*/
