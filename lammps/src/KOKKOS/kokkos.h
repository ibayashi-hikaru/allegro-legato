// clang-format off
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

#ifndef KOKKOS_LMP_H
#define KOKKOS_LMP_H

#include "pointers.h"
#include "kokkos_type.h"
#include "pair_kokkos.h"

namespace LAMMPS_NS {

class KokkosLMP : protected Pointers {
 public:
  int kokkos_exists;
  int neighflag;
  int neighflag_qeq;
  int neighflag_qeq_set;
  int exchange_comm_classic;
  int forward_comm_classic;
  int forward_pair_comm_classic;
  int forward_fix_comm_classic;
  int reverse_comm_classic;
  int exchange_comm_on_host;
  int forward_comm_on_host;
  int reverse_comm_on_host;
  int exchange_comm_changed;
  int forward_comm_changed;
  int forward_pair_comm_changed;
  int forward_fix_comm_changed;
  int reverse_comm_changed;
  int nthreads,ngpus;
  int numa;
  int auto_sync;
  int gpu_aware_flag;
  int neigh_thread;
  int neigh_thread_set;
  int newtonflag;
  double binsize;

  static int is_finalized;
  static Kokkos::InitArguments args;
  static int init_ngpus;

  KokkosLMP(class LAMMPS *, int, char **);
  ~KokkosLMP();
  static void initialize(Kokkos::InitArguments, Error *);
  static void finalize();
  void accelerator(int, char **);
  int neigh_count(int);

  template<class DeviceType>
  int need_dup()
  {
    int value = 0;

    if (neighflag == HALFTHREAD)
      value = std::is_same<typename NeedDup<HALFTHREAD,DeviceType>::value,Kokkos::Experimental::ScatterDuplicated>::value;

    return value;
  }

 private:
  static void my_signal_handler(int);
};

}

#endif

/* ERROR/WARNING messages:

E: Invalid Kokkos command-line args

Self-explanatory.  See Section 2.7 of the manual for details.

E: Could not determine local MPI rank for multiple GPUs with Kokkos CUDA
because MPI library not recognized

The local MPI rank was not found in one of four supported environment variables.

E: Invalid number of threads requested for Kokkos: must be 1 or greater

Self-explanatory.

E: GPUs are requested but Kokkos has not been compiled for CUDA

Recompile Kokkos with CUDA support to use GPUs.

E: Kokkos has been compiled for CUDA, HIP, or SYCL but no GPUs are requested

One or more GPUs must be used when Kokkos is compiled for CUDA/HIP/SYCL.

W: Kokkos package already initalized, cannot reinitialize with different parameters

Self-explanatory.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

U: Must use Kokkos half/thread or full neighbor list with threads or GPUs

Using Kokkos half-neighbor lists with threading is not allowed.

E: Must use KOKKOS package option 'neigh full' with 'neigh/thread on'

The 'neigh/thread on' option requires a full neighbor list

*/
