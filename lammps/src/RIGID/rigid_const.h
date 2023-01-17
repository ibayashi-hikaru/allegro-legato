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
// clang-format off

#ifndef LMP_RIGID_CONST_H
#define LMP_RIGID_CONST_H

namespace LAMMPS_NS {
  namespace RigidConst {

    enum{SINGLE, MOLECULE, GROUP};
    enum{NONE, XYZ, XY, YZ, XZ};
    enum{ISO, ANISO, TRICLINIC};
    enum{FULL_BODY, INITIAL, FINAL, FORCE_TORQUE, VCM_ANGMOM, XCM_MASS, ITENSOR, DOF};

    enum {POINT     = 1<<0,
          SPHERE    = 1<<1,
          ELLIPSOID = 1<<2,
          LINE      = 1<<3,
          TRIANGLE  = 1<<4,
          DIPOLE    = 1<<5,
          OMEGA     = 1<<6,
          ANGMOM    = 1<<7,
          TORQUE    = 1<<8
    };

    static constexpr double TOLERANCE = 1.0e-6;
    static constexpr double EPSILON   = 1.0e-7;
    static constexpr double BIG       = 1.0e20;

    // moment of inertia prefactor for sphere
    static constexpr double SINERTIA = 0.4;
    // moment of inertia prefactor for ellipsoid
    static constexpr double EINERTIA = 0.2;
    // moment of inertia prefactor for line segment
    static constexpr double LINERTIA = 1.0/12.0;

    static constexpr int MAXLINE    = 1024;
    static constexpr int CHUNK      = 1024;
    static constexpr int DELTA_BODY = 10000;
    static constexpr int ATTRIBUTE_PERBODY = 20;
  }
}

#endif
