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

#ifdef BOND_CLASS
// clang-format off
BondStyle(table,BondTable);
// clang-format on
#else

#ifndef LMP_BOND_TABLE_H
#define LMP_BOND_TABLE_H

#include "bond.h"

namespace LAMMPS_NS {

class BondTable : public Bond {
 public:
  BondTable(class LAMMPS *);
  virtual ~BondTable();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  double equilibrium_distance(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  double single(int, double, int, int, double &);

 protected:
  int tabstyle, tablength;
  double *r0;

  struct Table {
    int ninput, fpflag;
    double fplo, fphi, r0;
    double lo, hi;
    double *rfile, *efile, *ffile;
    double *e2file, *f2file;
    double delta, invdelta, deltasq6;
    double *r, *e, *de, *f, *df, *e2, *f2;
  };

  int ntables;
  Table *tables;
  int *tabindex;

  void allocate();
  void null_table(Table *);
  void free_table(Table *);
  void read_table(Table *, char *, char *);
  void bcast_table(Table *);
  void spline_table(Table *);
  void compute_table(Table *);

  void param_extract(Table *, char *);
  void spline(double *, double *, int, double, double, double *);
  double splint(double *, double *, double *, int, double);

  void uf_lookup(int, double, double &, double &);
  void u_lookup(int, double, double &);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Unknown table style in bond style table

Self-explanatory.

E: Illegal number of bond table entries

There must be at least 2 table entries.

E: Invalid bond table length

Length must be 2 or greater.

E: Bond table values are not increasing

The values in the tabulated file must be monotonically increasing.

E: Cannot open file %s

The specified file cannot be opened.  Check that the path and name are
correct. If the file is a compressed file, also check that the gzip
executable can be found and run.

E: Did not find keyword in table file

Keyword used in pair_coeff command was not found in table file.

E: Premature end of file in bond table

UNDOCUMENTED

W: %d of %d force values in table are inconsistent with -dE/dr.\n  Should only be flagged at inflection points

UNDOCUMENTED

W: %d of %d lines in table were incomplete or could not be parsed completely

UNDOCUMENTED

E: Invalid keyword in bond table parameters

Self-explanatory.

E: Bond table parameters did not set N

List of bond table parameters must include N setting.

E: Illegal bond in bond style table

UNDOCUMENTED

E: Bond length < table inner cutoff: type %d length %g

UNDOCUMENTED

E: Bond length > table outer cutoff: type %d length %g

UNDOCUMENTED

*/
