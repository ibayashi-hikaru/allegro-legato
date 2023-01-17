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

#ifdef IMPROPER_CLASS
// clang-format off
ImproperStyle(inversion/harmonic,ImproperInversionHarmonic);
// clang-format on
#else

#ifndef LMP_IMPROPER_INVERSION_HARMONIC_H
#define LMP_IMPROPER_INVERSION_HARMONIC_H

#include "improper.h"

namespace LAMMPS_NS {

class ImproperInversionHarmonic : public Improper {
 public:
  ImproperInversionHarmonic(class LAMMPS *);
  virtual ~ImproperInversionHarmonic();
  virtual void compute(int, int);
  void coeff(int, char **);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_data(FILE *);

 protected:
  double *kw, *w0;
  void invang(const int &i1, const int &i2, const int &i3, const int &i4, const int &type,
              const int &evflag, const int &eflag, const double &vb1x, const double &vb1y,
              const double &vb1z, const double &rrvb1, const double &rr2vb1, const double &vb2x,
              const double &vb2y, const double &vb2z, const double &rrvb2, const double &rr2vb2,
              const double &vb3x, const double &vb3y, const double &vb3z, const double &rrvb3,
              const double &rr2vb3);
  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
