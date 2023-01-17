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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(yukawa/kk,PairYukawaKokkos<LMPDeviceType>);
PairStyle(yukawa/kk/device,PairYukawaKokkos<LMPDeviceType>);
PairStyle(yukawa/kk/host,PairYukawaKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_YUKAWA_KOKKOS_H
#define LMP_PAIR_YUKAWA_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_yukawa.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairYukawaKokkos : public PairYukawa {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairYukawaKokkos(class LAMMPS *);
  virtual ~PairYukawaKokkos();

  void compute(int, int);
  void init_style();
  double init_one(int,int);

  struct params_yukawa {
    KOKKOS_INLINE_FUNCTION
    params_yukawa() { cutsq=0, a = 0; offset = 0; }
    KOKKOS_INLINE_FUNCTION
    params_yukawa(int /*i*/) { cutsq=0, a = 0; offset = 0; }
    F_FLOAT cutsq, a, offset;
  };


 protected:
  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fpair(const F_FLOAT& rsq, const int& i, const int&j,
                        const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j,
                        const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_ecoul(const F_FLOAT& /*rsq*/, const int& /*i*/, const int& /*j*/,
                        const int& /*itype*/, const int& /*jtype*/) const { return 0; }


  Kokkos::DualView<params_yukawa**,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_yukawa**,Kokkos::LayoutRight,DeviceType>::t_dev_const_um params;
  params_yukawa m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  F_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  typename AT::t_x_array_randomread x;
  typename AT::t_x_array c_x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;
  typename AT::t_tagint_1d tag;

  int newton_pair;
  double special_lj[4];

  typename AT::tdual_ffloat_2d k_cutsq;
  typename AT::t_ffloat_2d d_cutsq;


  int neighflag;
  int nlocal,nall,eflag,vflag;

  void allocate();
  friend struct PairComputeFunctor<PairYukawaKokkos,FULL,true>;
  friend struct PairComputeFunctor<PairYukawaKokkos,HALF,true>;
  friend struct PairComputeFunctor<PairYukawaKokkos,HALFTHREAD,true>;
  friend struct PairComputeFunctor<PairYukawaKokkos,FULL,false>;
  friend struct PairComputeFunctor<PairYukawaKokkos,HALF,false>;
  friend struct PairComputeFunctor<PairYukawaKokkos,HALFTHREAD,false>;
  friend EV_FLOAT pair_compute_neighlist<PairYukawaKokkos,FULL,void>(
    PairYukawaKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairYukawaKokkos,HALF,void>(
    PairYukawaKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairYukawaKokkos,HALFTHREAD,void>(
    PairYukawaKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairYukawaKokkos,void>(
    PairYukawaKokkos*,NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairYukawaKokkos>(PairYukawaKokkos*);

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot use Kokkos pair style with rRESPA inner/middle

UNDOCUMENTED

E: Cannot use chosen neighbor list style with yukawa/kk

That style is not supported by Kokkos.

U: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

U: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

*/
