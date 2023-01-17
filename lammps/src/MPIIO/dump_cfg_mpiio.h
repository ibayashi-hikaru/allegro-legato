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
DumpStyle(cfg/mpiio,DumpCFGMPIIO);
// clang-format on
#else

#ifndef LMP_DUMP_CFG_MPIIO_H
#define LMP_DUMP_CFG_MPIIO_H

#include "dump_cfg.h"

namespace LAMMPS_NS {

class DumpCFGMPIIO : public DumpCFG {
 public:
  DumpCFGMPIIO(class LAMMPS *, int, char **);
  virtual ~DumpCFGMPIIO();

 protected:
  bigint
      sumFileSize;    // size in bytes of the file up through this rank offset from the end of the header data
  char *headerBuffer;    // buffer for holding header data

  MPI_File mpifh;
  MPI_Offset mpifo, offsetFromHeader, headerSize, currentFileSize;
  int performEstimate;    // switch for write_data and write_header methods to use for gathering data and detemining filesize for preallocation vs actually writing the data
  char *filecurrent;      // name of file for this round (with % and * replaced)

#if defined(_OPENMP)
  int convert_string_omp(int, double *);    // multithreaded version of convert_string
#endif

  virtual void openfile();
  virtual void init_style();
  virtual void write_header(bigint);
  virtual void write();
  virtual void write_data(int, double *);

  typedef void (DumpCFGMPIIO::*FnPtrData)(int, double *);
  FnPtrData write_choice;    // ptr to write data functions
  void write_string(int, double *);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot open dump file %s

The output file for the dump command cannot be opened.  Check that the
path and name are correct.

E: Too much per-proc info for dump

Number of local atoms times number of columns must fit in a 32-bit
integer for dump.

E: Dump cfg requires one snapshot per file

Use the wildcard "*" character in the filename.

*/
