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
   Contributing author: W. Michael Brown (Intel)
------------------------------------------------------------------------- */

#include "npair_half_bin_newton_tri_intel.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "neigh_list.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

NPairHalfBinNewtonTriIntel::NPairHalfBinNewtonTriIntel(LAMMPS *lmp) :
  NPairIntel(lmp) {}

/* ----------------------------------------------------------------------
   binned neighbor list construction with Newton's 3rd law for triclinic
   each owned atom i checks its own bin and other bins in triclinic stencil
   every pair stored exactly once by some processor
------------------------------------------------------------------------- */

void NPairHalfBinNewtonTriIntel::build(NeighList *list)
{
  if (nstencil > INTEL_MAX_STENCIL)
    error->all(FLERR, "Too many neighbor bins for INTEL package.");

  #ifdef _LMP_INTEL_OFFLOAD
  if (exclude)
    error->all(FLERR, "Exclusion lists not yet supported for Intel offload");
  #endif

  if (_fix->precision() == FixIntel::PREC_MODE_MIXED)
    hbnti(list, _fix->get_mixed_buffers());
  else if (_fix->precision() == FixIntel::PREC_MODE_DOUBLE)
    hbnti(list, _fix->get_double_buffers());
  else
    hbnti(list, _fix->get_single_buffers());

  _fix->stop_watch(TIME_HOST_NEIGHBOR);
}

template <class flt_t, class acc_t>
void NPairHalfBinNewtonTriIntel::
hbnti(NeighList *list, IntelBuffers<flt_t,acc_t> *buffers) {
  const int nlocal = (includegroup) ? atom->nfirst : atom->nlocal;
  list->inum = nlocal;

  int host_start = _fix->host_start_neighbor();
  const int off_end = _fix->offload_end_neighbor();

  #ifdef _LMP_INTEL_OFFLOAD
  if (off_end) grow_stencil();
  if (_fix->full_host_list()) host_start = 0;
  int offload_noghost = _fix->offload_noghost();
  #endif

  buffers->grow_list(list, atom->nlocal, comm->nthreads, 0, off_end);

  int need_ic = 0;
  if (atom->molecular != Atom::ATOMIC)
    dminimum_image_check(need_ic, neighbor->cutneighmax, neighbor->cutneighmax,
                         neighbor->cutneighmax);

  #ifdef _LMP_INTEL_OFFLOAD
  if (need_ic) {
    if (offload_noghost) {
      bin_newton<flt_t,acc_t,1,1,0,1,0>(1, list, buffers, 0, off_end);
      bin_newton<flt_t,acc_t,1,1,0,1,0>(0, list, buffers, host_start, nlocal,
                                        off_end);
    } else {
      bin_newton<flt_t,acc_t,0,1,0,1,0>(1, list, buffers, 0, off_end);
      bin_newton<flt_t,acc_t,0,1,0,1,0>(0, list, buffers, host_start, nlocal);
    }
  } else {
    if (offload_noghost) {
      bin_newton<flt_t,acc_t,1,0,0,1,0>(1, list, buffers, 0, off_end);
      bin_newton<flt_t,acc_t,1,0,0,1,0>(0, list, buffers, host_start, nlocal,
                                        off_end);
    } else {
      bin_newton<flt_t,acc_t,0,0,0,1,0>(1, list, buffers, 0, off_end);
      bin_newton<flt_t,acc_t,0,0,0,1,0>(0, list, buffers, host_start, nlocal);
    }
  }
  #else
  if (need_ic)
    bin_newton<flt_t,acc_t,0,1,0,1,0>(0, list, buffers, host_start, nlocal);
  else
    bin_newton<flt_t,acc_t,0,0,0,1,0>(0, list, buffers, host_start, nlocal);
  #endif
}
