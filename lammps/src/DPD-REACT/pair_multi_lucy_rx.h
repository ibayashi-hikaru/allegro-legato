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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(multi/lucy/rx,PairMultiLucyRX);
// clang-format on
#else

#ifndef LMP_PAIR_MULTI_LUCY_RX_H
#define LMP_PAIR_MULTI_LUCY_RX_H

#include "pair.h"

namespace LAMMPS_NS {

class PairMultiLucyRX : public Pair {
 public:
  PairMultiLucyRX(class LAMMPS *);
  virtual ~PairMultiLucyRX();

  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  virtual int pack_forward_comm(int, int *, double *, int, int *);
  virtual void unpack_forward_comm(int, int, double *);
  virtual int pack_reverse_comm(int, int, double *);
  virtual void unpack_reverse_comm(int, int *, double *);
  void computeLocalDensity();
  double rho_0;

 protected:
  enum { LOOKUP, LINEAR };

  int nmax;

  int tabstyle, tablength;
  struct Table {
    int ninput, rflag, fpflag, match;
    double rlo, rhi, fplo, fphi, cut;
    double *rfile, *efile, *ffile;
    double *e2file, *f2file;
    double innersq, delta, invdelta, deltasq6;
    double *rsq, *drsq, *e, *de, *f, *df, *e2, *f2;
  };
  int ntables;
  Table *tables;

  int **tabindex;

  virtual void allocate();
  void read_table(Table *, char *, char *);
  void param_extract(Table *, char *);
  void bcast_table(Table *);
  void spline_table(Table *);
  void compute_table(Table *);
  void null_table(Table *);
  void free_table(Table *);
  void spline(double *, double *, int, double, double, double *);
  double splint(double *, double *, double *, int, double);

  int nspecies;
  char *site1, *site2;
  int isite1, isite2;
  void getMixingWeights(int, double &, double &, double &, double &);
  bool fractionalWeighting;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Pair multi/lucy/rx command requires atom_style with density (e.g. dpd, meso)

Self-explanatory

E: Density < table inner cutoff

The local density inner is smaller than the inner cutoff

E: Density > table inner cutoff

The local density inner is greater than the inner cutoff

E: Only LOOKUP and LINEAR table styles have been implemented for pair multi/lucy/rx

Self-explanatory

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E:  Unknown table style in pair_style command

Self-explanatory

E: Illegal number of pair table entries

There must be at least 2 table entries.

E: Illegal pair_coeff command

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: PairMultiLucyRX requires a fix rx command

The fix rx command must come before the pair style command in the input file

E:  There are no rx species specified

There must be at least one species specified through the fix rx command

E: Invalid pair table length

Length of read-in pair table is invalid

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Cannot open file %s

The specified file cannot be opened.  Check that the path and name are
correct.

E: Did not find keyword in table file

Keyword used in pair_coeff command was not found in table file.

E: Invalid keyword in pair table parameters

Keyword used in list of table parameters is not recognized.

E: Pair table parameters did not set N

List of pair table parameters must include N setting.

*/
