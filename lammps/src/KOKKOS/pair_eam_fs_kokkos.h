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
PairStyle(eam/fs/kk,PairEAMFSKokkos<LMPDeviceType>);
PairStyle(eam/fs/kk/device,PairEAMFSKokkos<LMPDeviceType>);
PairStyle(eam/fs/kk/host,PairEAMFSKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_EAM_FS_KOKKOS_H
#define LMP_PAIR_EAM_FS_KOKKOS_H

#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "pair_eam.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

struct TagPairEAMFSPackForwardComm{};
struct TagPairEAMFSUnpackForwardComm{};
struct TagPairEAMFSInitialize{};

template<int NEIGHFLAG, int NEWTON_PAIR>
struct TagPairEAMFSKernelA{};

template<int EFLAG>
struct TagPairEAMFSKernelB{};

template<int EFLAG>
struct TagPairEAMFSKernelAB{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairEAMFSKernelC{};

// Cannot use virtual inheritance on the GPU

template<class DeviceType>
class PairEAMFSKokkos : public PairEAM, public KokkosBase {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairEAMFSKokkos(class LAMMPS *);
  virtual ~PairEAMFSKokkos();
  void compute(int, int);
  void init_style();
  void *extract(const char *, int &) { return nullptr; }
  void coeff(int, char **);

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairEAMFSPackForwardComm, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairEAMFSUnpackForwardComm, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairEAMFSInitialize, const int&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairEAMFSKernelA<NEIGHFLAG,NEWTON_PAIR>, const int&) const;

  template<int EFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairEAMFSKernelB<EFLAG>, const int&, EV_FLOAT&) const;

  template<int EFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairEAMFSKernelB<EFLAG>, const int&) const;

  template<int EFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairEAMFSKernelAB<EFLAG>, const int&, EV_FLOAT&) const;

  template<int EFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairEAMFSKernelAB<EFLAG>, const int&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairEAMFSKernelC<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairEAMFSKernelC<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

  int pack_forward_comm_kokkos(int, DAT::tdual_int_2d, int, DAT::tdual_xfloat_1d&,
                       int, int *);
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_xfloat_1d&);
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);

 protected:
  typename AT::t_x_array x;
  typename AT::t_f_array f;
  typename AT::t_int_1d type;
  typename AT::t_tagint_1d tag;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename ArrayTypes<DeviceType>::t_efloat_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_virial_array d_vatom;

  int need_dup;
  Kokkos::Experimental::ScatterView<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_rho;
  Kokkos::Experimental::ScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_f;
  Kokkos::Experimental::ScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_eatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_vatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_rho;
  Kokkos::Experimental::ScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_f;
  Kokkos::Experimental::ScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_eatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_vatom;

  DAT::tdual_ffloat_1d k_rho;
  DAT::tdual_ffloat_1d k_fp;
  typename AT::t_ffloat_1d d_rho;
  typename AT::t_ffloat_1d d_fp;
  HAT::t_ffloat_1d h_rho;
  HAT::t_ffloat_1d h_fp;

  typename AT::t_int_1d d_type2frho;
  typename AT::t_int_2d_dl d_type2rhor;
  typename AT::t_int_2d_dl d_type2z2r;

  typedef Kokkos::DualView<F_FLOAT**[7],DeviceType> tdual_ffloat_2d_n7;
  typedef typename tdual_ffloat_2d_n7::t_dev_const t_ffloat_2d_n7;
  typedef typename tdual_ffloat_2d_n7::t_host t_host_ffloat_2d_n7;

  t_ffloat_2d_n7 d_frho_spline;
  t_ffloat_2d_n7 d_rhor_spline;
  t_ffloat_2d_n7 d_z2r_spline;

  void file2array();
  void file2array_fs();
  void array2spline();
  void interpolate(int, double, double *, t_host_ffloat_2d_n7, int);
  void read_file(char *);

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d d_ilist;
  typename AT::t_int_1d d_numneigh;

  int iswap;
  int first;
  typename AT::t_int_2d d_sendlist;
  typename AT::t_xfloat_1d_um v_buf;

  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  friend void pair_virial_fdotr_compute<PairEAMFSKokkos>(PairEAMFSKokkos*);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot use chosen neighbor list style with pair eam/kk/fs

Self-explanatory.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: No matching element in EAM potential file

The EAM potential file does not contain elements that match the
requested elements.

E: Cannot open EAM potential file %s

The specified EAM potential file cannot be opened.  Check that the
path and name are correct.

E: Incorrect element names in EAM potential file

The element names in the EAM file do not match those requested.

*/
