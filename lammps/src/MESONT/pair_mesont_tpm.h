/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Contributing author: Maxim Shugaev (UVA), mvs9t@virginia.edu
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mesont/tpm,PairMESONTTPM);
// clang-format on
#else

#ifndef LMP_PAIR_MESONT_TPM_H
#define LMP_PAIR_MESONT_TPM_H

#include "pair.h"

namespace LAMMPS_NS {

class PairMESONTTPM : public Pair {
 public:
  PairMESONTTPM(class LAMMPS *);
  virtual ~PairMESONTTPM();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  virtual void init_style();

  double energy_s;                        // accumulated energies for stretching
  double energy_b;                        // accumulated energies for bending
  double energy_t;                        // accumulated energies for tube-tube interaction
  double *eatom_s, *eatom_b, *eatom_t;    // accumulated per-atom values

 protected:
  int BendingMode, TPMType;
  char *tab_path;
  int tab_path_length;
  double cut_global;
  double **cut;
  static int instance_count;
  int nmax;

  virtual void allocate();
  virtual void *extract(const char *, int &);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Pair style mesont/tpm requires newton pair on

newton_pair must be set to on

E: The selected cutoff is too small for the current system

cutoff must be increased.

E: Illegal pair_style command

Incorrect argument list in the style init.

E: Incorrect table path

Incorrect path to the table files.

E: Incorrect BendingMode

Self-explanatory.

E: Incorrect TPMType

Self-explanatory.

E: Inconsistent input and potential table

The tube diameter is inconsistent with the chirality specified
during generation of the potential table.

*/
