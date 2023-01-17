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
PairStyle(gran/hooke/history/kk,PairGranHookeHistoryKokkos<LMPDeviceType>);
PairStyle(gran/hooke/history/kk/device,PairGranHookeHistoryKokkos<LMPDeviceType>);
PairStyle(gran/hooke/history/kk/host,PairGranHookeHistoryKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_GRAN_HOOKE_HISTORY_KOKKOS_H
#define LMP_PAIR_GRAN_HOOKE_HISTORY_KOKKOS_H

#include "pair_gran_hooke_history.h"
#include "pair_kokkos.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template <class DeviceType>
class FixNeighHistoryKokkos;

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, int SHEARUPDATE>
struct TagPairGranHookeHistoryCompute {};

struct TagPairGranHookeHistoryReduce {};

template <class DeviceType>
class PairGranHookeHistoryKokkos : public PairGranHookeHistory {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairGranHookeHistoryKokkos(class LAMMPS *);
  virtual ~PairGranHookeHistoryKokkos();
  virtual void compute(int, int);
  void init_style();

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairGranHookeHistoryReduce, const int ii) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, int SHEARUPDATE>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairGranHookeHistoryCompute<NEIGHFLAG,NEWTON_PAIR,EVFLAG,SHEARUPDATE>, const int, EV_FLOAT &ev) const;
  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, int SHEARUPDATE>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairGranHookeHistoryCompute<NEIGHFLAG,NEWTON_PAIR,EVFLAG,SHEARUPDATE>, const int) const;

  template<int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION
  void ev_tally_xyz(EV_FLOAT &ev, int i, int j,
                    F_FLOAT fx, F_FLOAT fy, F_FLOAT fz,
                    X_FLOAT delx, X_FLOAT dely, X_FLOAT delz) const;
  template<int NEIGHFLAG, int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION
  void ev_tally_xyz_atom(EV_FLOAT &ev, int i, int j,
                         F_FLOAT fx, F_FLOAT fy, F_FLOAT fz,
                         X_FLOAT delx, X_FLOAT dely, X_FLOAT delz) const;

 protected:
  typename AT::t_x_array_randomread x;
  typename AT::t_x_array c_x;
  typename AT::t_v_array_randomread v;
  typename AT::t_v_array_randomread omega;
  typename AT::t_f_array f;
  typename AT::t_f_array torque;
  typename AT::t_int_1d_randomread type;
  typename AT::t_int_1d_randomread mask;
  typename AT::t_float_1d_randomread rmass;
  typename AT::t_float_1d_randomread radius;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;
  typename AT::t_tagint_1d tag;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  typename AT::t_int_2d d_firsttouch;
  typename AT::t_float_2d d_firstshear;

  typename AT::t_neighbors_2d d_neighbors_touch;
  typename AT::t_int_1d d_numneigh_touch;

  int newton_pair;
  double special_lj[4];

  int neighflag;
  int nlocal,nall,eflag,vflag;

  FixNeighHistoryKokkos<DeviceType> *fix_historyKK;

  friend void pair_virial_fdotr_compute<PairGranHookeHistoryKokkos>(PairGranHookeHistoryKokkos*);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair granular requires atom attributes radius, rmass

The atom style defined does not have these attributes.

E: Pair granular requires ghost atoms store velocity

Use the comm_modify vel yes command to enable this.

E: Could not find pair fix neigh history ID

UNDOCUMENTED

U: Pair granular with shear history requires newton pair off

This is a current restriction of the implementation of pair
granular styles with history.

U: Could not find pair fix ID

A fix is created internally by the pair style to store shear
history information.  You cannot delete it.

*/
