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

#ifdef DUMP_CLASS
// clang-format off
DumpStyle(xtc,DumpXTC);
// clang-format on
#else

#ifndef LMP_DUMP_XTC_H
#define LMP_DUMP_XTC_H

#include "dump.h"
#include "xdr_compat.h"

namespace LAMMPS_NS {

class DumpXTC : public Dump {
 public:
  DumpXTC(class LAMMPS *, int, char **);
  virtual ~DumpXTC();

 private:
  int natoms, ntotal;
  int nevery_save;
  int unwrap_flag;    // 1 if atom coords are unwrapped, 0 if no
  float precision;    // user-adjustable precision setting
  float *coords;
  double sfactor, tfactor;    // scaling factors for positions and time unit
  XDR xd;

  void init_style();
  int modify_param(int, char **);
  void openfile();
  void write_header(bigint);
  void pack(tagint *);
  void write_data(int, double *);
  double memory_usage();

  void write_frame();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid dump xtc filename

Filenames used with the dump xtc style cannot be binary or compressed
or cause multiple files to be written.

E: Too many atoms for dump xtc

The system size must fit in a 32-bit integer to use this dump
style.

W: No automatic unit conversion to XTC file format conventions possible for units lj

This means no scaling will be performed.

E: Dump xtc requires sorting by atom ID

Use the dump_modify sort command to enable this.

E: Cannot set dump_modify flush for dump xtc

Self-explanatory.

E: Cannot use variable every setting for dump xtc

The format of this file requires snapshots at regular intervals.

E: Cannot change dump_modify every for dump xtc

The frequency of writing dump xtc snapshots cannot be changed.

E: Cannot open dump file

Self-explanatory.

E: Too big a timestep for dump xtc

The timestep must fit in a 32-bit integer to use this dump style.

E: Illegal dump_modify sfactor value (must be > 0.0)

Self-explanatory.

E: Illegal dump_modify tfactor value (must be > 0.0)

Self-explanatory.

*/
