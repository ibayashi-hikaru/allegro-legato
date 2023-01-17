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

/* ----------------------------------------------------------------------
   Contributing author: W. Michael Brown (Intel)
------------------------------------------------------------------------- */

#ifdef DIHEDRAL_CLASS
// clang-format off
DihedralStyle(opls/intel,DihedralOPLSIntel);
// clang-format on
#else

#ifndef LMP_DIHEDRAL_OPLS_INTEL_H
#define LMP_DIHEDRAL_OPLS_INTEL_H

#include "dihedral_opls.h"
#include "fix_intel.h"

namespace LAMMPS_NS {

class DihedralOPLSIntel : public DihedralOPLS {

 public:
  DihedralOPLSIntel(class LAMMPS *lmp);
  virtual void compute(int, int);
  void init_style();

 private:
  FixIntel *fix;

  template <class flt_t> class ForceConst;
  template <class flt_t, class acc_t>
  void compute(int eflag, int vflag, IntelBuffers<flt_t, acc_t> *buffers,
               const ForceConst<flt_t> &fc);
  template <int EVFLAG, int EFLAG, int NEWTON_BOND, class flt_t, class acc_t>
  void eval(const int vflag, IntelBuffers<flt_t, acc_t> *buffers, const ForceConst<flt_t> &fc);
  template <class flt_t, class acc_t>
  void pack_force_const(ForceConst<flt_t> &fc, IntelBuffers<flt_t, acc_t> *buffers);

#ifdef _LMP_INTEL_OFFLOAD
  int _use_base;
#endif

  template <class flt_t> class ForceConst {
   public:
    typedef struct {
      flt_t k1, k2, k3, k4;
    } fc_packed1;

    fc_packed1 *bp;

    ForceConst() : _nbondtypes(0) {}
    ~ForceConst() { set_ntypes(0, nullptr); }

    void set_ntypes(const int nbondtypes, Memory *memory);

   private:
    int _nbondtypes;
    Memory *_memory;
  };
  ForceConst<float> force_const_single;
  ForceConst<double> force_const_double;
};

}    // namespace LAMMPS_NS

#endif
#endif
