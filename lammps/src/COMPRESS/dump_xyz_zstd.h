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
   Contributing author: Richard Berger (Temple U)
------------------------------------------------------------------------- */

#ifdef LAMMPS_ZSTD

#ifdef DUMP_CLASS
// clang-format off
DumpStyle(xyz/zstd,DumpXYZZstd);
// clang-format on
#else

#ifndef LMP_DUMP_XYZ_ZSTD_H
#define LMP_DUMP_XYZ_ZSTD_H

#include "dump_xyz.h"
#include "zstd_file_writer.h"

namespace LAMMPS_NS {

class DumpXYZZstd : public DumpXYZ {
 public:
  DumpXYZZstd(class LAMMPS *, int, char **);
  virtual ~DumpXYZZstd();

 protected:
  ZstdFileWriter writer;

  virtual void openfile();
  virtual void write_header(bigint);
  virtual void write_data(int, double *);
  virtual void write();

  virtual int modify_param(int, char **);
};

}    // namespace LAMMPS_NS

#endif
#endif
#endif

/* ERROR/WARNING messages:

E: Dump xyz/zstd only writes compressed files

The dump xyz/zstd output file name must have a .zst suffix.

E: Cannot open dump file

Self-explanatory.

*/
