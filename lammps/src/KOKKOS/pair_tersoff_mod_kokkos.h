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
PairStyle(tersoff/mod/kk,PairTersoffMODKokkos<LMPDeviceType>);
PairStyle(tersoff/mod/kk/device,PairTersoffMODKokkos<LMPDeviceType>);
PairStyle(tersoff/mod/kk/host,PairTersoffMODKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_TERSOFF_MOD_KOKKOS_H
#define LMP_PAIR_TERSOFF_MOD_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_tersoff_mod.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<int NEIGHFLAG, int EVFLAG>
struct TagPairTersoffMODComputeHalf{};

template<int NEIGHFLAG, int EVFLAG>
struct TagPairTersoffMODComputeFullA{};

template<int NEIGHFLAG, int EVFLAG>
struct TagPairTersoffMODComputeFullB{};

struct TagPairTersoffMODComputeShortNeigh{};

template<class DeviceType>
class PairTersoffMODKokkos : public PairTersoffMOD {
 public:
  enum {EnabledNeighFlags=FULL};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairTersoffMODKokkos(class LAMMPS *);
  virtual ~PairTersoffMODKokkos();
  virtual void compute(int, int);
  void init_style();

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODComputeHalf<NEIGHFLAG,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODComputeHalf<NEIGHFLAG,EVFLAG>, const int&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODComputeFullA<NEIGHFLAG,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODComputeFullA<NEIGHFLAG,EVFLAG>, const int&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODComputeFullB<NEIGHFLAG,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODComputeFullB<NEIGHFLAG,EVFLAG>, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODComputeShortNeigh, const int&) const;

  KOKKOS_INLINE_FUNCTION
  double ters_fc_k(const int &i, const int &j, const int &k, const F_FLOAT &r) const;

  KOKKOS_INLINE_FUNCTION
  double ters_dfc(const int &i, const int &j, const int &k, const F_FLOAT &r) const;

  KOKKOS_INLINE_FUNCTION
  double ters_fa_k(const int &i, const int &j, const int &k, const F_FLOAT &r) const;

  KOKKOS_INLINE_FUNCTION
  double ters_dfa(const int &i, const int &j, const int &k, const F_FLOAT &r) const;

  KOKKOS_INLINE_FUNCTION
  double ters_bij_k(const int &i, const int &j, const int &k, const F_FLOAT &bo) const;

  KOKKOS_INLINE_FUNCTION
  double ters_dbij(const int &i, const int &j, const int &k, const F_FLOAT &bo) const;

  KOKKOS_INLINE_FUNCTION
  double bondorder(const int &i, const int &j, const int &k,
        const F_FLOAT &rij, const F_FLOAT &dx1, const F_FLOAT &dy1, const F_FLOAT &dz1,
        const F_FLOAT &rik, const F_FLOAT &dx2, const F_FLOAT &dy2, const F_FLOAT &dz2) const;

  KOKKOS_INLINE_FUNCTION
  double ters_gijk(const int &i, const int &j, const int &k, const F_FLOAT &cos) const;

  KOKKOS_INLINE_FUNCTION
  double ters_dgijk(const int &i, const int &j, const int &k, const F_FLOAT &cos) const;

  KOKKOS_INLINE_FUNCTION
  void ters_dthb(const int &i, const int &j, const int &k, const F_FLOAT &prefactor,
        const F_FLOAT &rij, const F_FLOAT &dx1, const F_FLOAT &dy1, const F_FLOAT &dz1,
        const F_FLOAT &rik, const F_FLOAT &dx2, const F_FLOAT &dy2, const F_FLOAT &dz2,
        F_FLOAT *fi, F_FLOAT *fj, F_FLOAT *fk) const;

  KOKKOS_INLINE_FUNCTION
  void ters_dthbj(const int &i, const int &j, const int &k, const F_FLOAT &prefactor,
        const F_FLOAT &rij, const F_FLOAT &dx1, const F_FLOAT &dy1, const F_FLOAT &dz1,
        const F_FLOAT &rik, const F_FLOAT &dx2, const F_FLOAT &dy2, const F_FLOAT &dz2,
        F_FLOAT *fj, F_FLOAT *fk) const;

  KOKKOS_INLINE_FUNCTION
  void ters_dthbk(const int &i, const int &j, const int &k, const F_FLOAT &prefactor,
        const F_FLOAT &rij, const F_FLOAT &dx1, const F_FLOAT &dy1, const F_FLOAT &dz1,
        const F_FLOAT &rik, const F_FLOAT &dx2, const F_FLOAT &dy2, const F_FLOAT &dz2,
        F_FLOAT *fk) const;

  KOKKOS_INLINE_FUNCTION
  double vec3_dot(const F_FLOAT x[3], const double y[3]) const {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
  }

  KOKKOS_INLINE_FUNCTION
  void vec3_add(const F_FLOAT x[3], const double y[3], double * const z) const {
    z[0] = x[0]+y[0]; z[1] = x[1]+y[1]; z[2] = x[2]+y[2];
  }

  KOKKOS_INLINE_FUNCTION
  void vec3_scale(const F_FLOAT k, const double x[3], double y[3]) const {
    y[0] = k*x[0]; y[1] = k*x[1]; y[2] = k*x[2];
  }

  KOKKOS_INLINE_FUNCTION
  void vec3_scaleadd(const F_FLOAT k, const double x[3], const double y[3], double * const z) const {
    z[0] = k*x[0]+y[0]; z[1] = k*x[1]+y[1]; z[2] = k*x[2]+y[2];
  }

  KOKKOS_INLINE_FUNCTION
  int sbmask(const int& j) const;

  struct params_ters{
    KOKKOS_INLINE_FUNCTION
    params_ters() {powerm=0;lam3=0;h=0;powern=0;beta=0;lam2=0;bigb=0;bigr=0;bigd=0;
      lam1=0;biga=0;powern_del=0;cutsq=0;c1=0;c2=0;c3=0;c4=0;c5=0;ca1=0;ca4=0;};
    KOKKOS_INLINE_FUNCTION
    params_ters(int /*i*/) {powerm=0;lam3=0;h=0;powern=0;beta=0;lam2=0;bigb=0;bigr=0;bigd=0;
      lam1=0;biga=0;powern_del=0;cutsq=0;c1=0;c2=0;c3=0;c4=0;c5=0;ca1=0;ca4=0;};
    F_FLOAT powerm, lam3, h, powern, beta, lam2, bigb, bigr, bigd,
      lam1, biga, powern_del, cutsq, c1, c2, c3, c4, c5, ca1, ca4;
  };

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void v_tally3(EV_FLOAT &ev, const int &i, const int &j, const int &k,
    F_FLOAT *fj, F_FLOAT *fk, F_FLOAT *drij, F_FLOAT *drik) const;

  KOKKOS_INLINE_FUNCTION
  void v_tally3_atom(EV_FLOAT &ev, const int &i, const int &j, const int &k,
    F_FLOAT *fj, F_FLOAT *fk, F_FLOAT *drji, F_FLOAT *drjk) const;

  void allocate();
  void setup_params();

 protected:
  typedef Kokkos::DualView<int***,DeviceType> tdual_int_3d;
  Kokkos::DualView<params_ters***,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_ters***,
    Kokkos::LayoutRight,DeviceType>::t_dev_const_um paramskk;
  // hardwired to space for 12 atom types
  //params_ters m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];

  int inum;
  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;
  typename AT::t_tagint_1d tag;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename ArrayTypes<DeviceType>::t_efloat_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_virial_array d_vatom;

  int need_dup;
  Kokkos::Experimental::ScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_f;
  Kokkos::Experimental::ScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_eatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_vatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_f;
  Kokkos::Experimental::ScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_eatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_vatom;

  typedef Kokkos::DualView<F_FLOAT**[7],Kokkos::LayoutRight,DeviceType> tdual_ffloat_2d_n7;
  typedef typename tdual_ffloat_2d_n7::t_dev_const_randomread t_ffloat_2d_n7_randomread;
  typedef typename tdual_ffloat_2d_n7::t_host t_host_ffloat_2d_n7;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  //NeighListKokkos<DeviceType> k_list;

  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  Kokkos::View<int**,DeviceType> d_neighbors_short;
  Kokkos::View<int*,DeviceType> d_numneigh_short;

  friend void pair_virial_fdotr_compute<PairTersoffMODKokkos>(PairTersoffMODKokkos*);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot (yet) use full neighbor list style with tersoff/mod/kk

Self-explanatory.

E: Cannot use chosen neighbor list style with tersoff/mod/kk

Self-explanatory.

*/
