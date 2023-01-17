/* -*- c++ -*- ---------------------------------------------------------
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
PairStyle(nb3b/harmonic,PairNb3bHarmonic);
// clang-format on
#else

#ifndef LMP_PAIR_NB3B_HARMONIC_H
#define LMP_PAIR_NB3B_HARMONIC_H

#include "pair.h"

namespace LAMMPS_NS {

class PairNb3bHarmonic : public Pair {
 public:
  PairNb3bHarmonic(class LAMMPS *);
  virtual ~PairNb3bHarmonic();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void init_style();

  static constexpr int NPARAMS_PER_LINE = 6;

 protected:
  struct Param {
    double k_theta, theta0, cutoff;
    double cut, cutsq;
    int ielement, jelement, kelement;
  };

  double cutmax;    // max cutoff for all elements
  Param *params;    // parameter set for an I-J-K interaction

  void allocate();
  void read_file(char *);
  void setup_params();
  void twobody(Param *, double, double &, int, double &);
  void threebody(Param *, Param *, Param *, double, double, double *, double *, double *, double *,
                 int, double &);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style nb3b/harmonic requires atom IDs

This is a requirement to use this potential.

E: Pair style nb3b/harmonic requires newton pair on

See the newton command.  This is a restriction to use this potential.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Cannot open nb3b/harmonic potential file %s

The specified potential file cannot be opened.  Check that the path
and name are correct.

E: Incorrect format in nb3b/harmonic potential file

Incorrect number of words per line in the potential file.

E: Illegal nb3b/harmonic parameter

One or more of the coefficients defined in the potential file is
invalid.

E: Potential file has duplicate entry

The potential file has more than one entry for the same element.

E: Potential file is missing an entry

The potential file does not have a needed entry.

*/
