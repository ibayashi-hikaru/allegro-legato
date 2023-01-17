// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2020) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// There is no way to annotate an OpenMP construct that
// (a) accesses const variables, (b) has default(none),
// and (c) is valid in both OpenMP 3.0 and 4.0.
//
// (in OpenMP 3.0, the const variables have a predetermined
//  sharing attribute and are *forbidden* from being declared
//  in the omp construct.  In OpenMP 4.0, this predetermined
//  sharing attribute is removed, and thus they are *required*
//  to be declared in the omp construct)
//
// To date, most compilers still accept the OpenMP 3.0 form,
// so this is what LAMMPS primarily uses.  For those compilers
// that strictly implement OpenMP 4.0 (such as GCC 9.0 and later
// or Clang 10.0 and later), we give up default(none).

// autodetect OpenMP compatibility if not explicitly set

#ifndef LAMMPS_OMP_COMPAT
#  if defined(__INTEL_LLVM_COMPILER)
#      define LAMMPS_OMP_COMPAT 4
#  elif defined(__INTEL_COMPILER)
#    if __INTEL_COMPILER > 18
#      define LAMMPS_OMP_COMPAT 4
#    endif
#  elif defined(__clang__)
#    if __clang_major__ >= 10
#      define LAMMPS_OMP_COMPAT 4
#    endif
#  elif defined(__GNUC__)
#    if __GNUC__ >= 9
#      define LAMMPS_OMP_COMPAT 4
#    endif
#  endif
#endif

#if LAMMPS_OMP_COMPAT == 4
#  define LMP_SHARED(...)
#  define LMP_DEFAULT_NONE default(shared)
#else
#  define LMP_SHARED(...) shared(__VA_ARGS__)
#  define LMP_DEFAULT_NONE default(none)
#endif

