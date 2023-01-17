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

#ifdef FIX_CLASS
// clang-format off
FixStyle(nve/sphere/kk,FixNVESphereKokkos<LMPDeviceType>);
FixStyle(nve/sphere/kk/device,FixNVESphereKokkos<LMPDeviceType>);
FixStyle(nve/sphere/kk/host,FixNVESphereKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_NVE_SPHERE_KOKKOS_H
#define LMP_FIX_NVE_SPHERE_KOKKOS_H

#include "fix_nve_sphere.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<class DeviceType>
class FixNVESphereKokkos : public FixNVESphere {
  public:
    FixNVESphereKokkos(class LAMMPS *, int, char **);
    virtual ~FixNVESphereKokkos() {}
    void cleanup_copy();
    void init();
    void initial_integrate(int);
    void final_integrate();

    KOKKOS_INLINE_FUNCTION
    void initial_integrate_item(const int i) const;
    KOKKOS_INLINE_FUNCTION
    void final_integrate_item(const int i) const;

  private:
    typename ArrayTypes<DeviceType>::t_x_array x;
    typename ArrayTypes<DeviceType>::t_v_array v;
    typename ArrayTypes<DeviceType>::t_v_array omega;
    typename ArrayTypes<DeviceType>::t_f_array f;
    typename ArrayTypes<DeviceType>::t_f_array torque;
    typename ArrayTypes<DeviceType>::t_float_1d rmass;
    typename ArrayTypes<DeviceType>::t_float_1d radius;
    typename ArrayTypes<DeviceType>::t_int_1d mask;
};

template <class DeviceType>
struct FixNVESphereKokkosInitialIntegrateFunctor {
  typedef DeviceType device_type;
  FixNVESphereKokkos<DeviceType> c;
  FixNVESphereKokkosInitialIntegrateFunctor(FixNVESphereKokkos<DeviceType> *c_ptr): c(*c_ptr) { c.cleanup_copy(); }
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    c.initial_integrate_item(i);
  }
};

template <class DeviceType>
struct FixNVESphereKokkosFinalIntegrateFunctor {
  typedef DeviceType device_type;
  FixNVESphereKokkos<DeviceType> c;
  FixNVESphereKokkosFinalIntegrateFunctor(FixNVESphereKokkos<DeviceType> *c_ptr): c(*c_ptr) { c.cleanup_copy(); }
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    c.final_integrate_item(i);
  }
};

} // namespace LAMMPS_NS

#endif // LMP_FIX_NVE_SPHERE_KOKKOS_H
#endif // FIX_CLASS
