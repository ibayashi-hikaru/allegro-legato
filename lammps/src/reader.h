/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Contributed by Timothy Sirk
------------------------------------------------------------------------- */

#ifndef LMP_READER_H
#define LMP_READER_H

#include "pointers.h"

namespace LAMMPS_NS {

class Reader : protected Pointers {
 public:
  Reader(class LAMMPS *);
  virtual ~Reader() {}

  virtual void settings(int, char **);

  virtual int read_time(bigint &) = 0;
  virtual void skip() = 0;
  virtual bigint read_header(double[3][3], int &, int &, int, int, int *, char **, int, int, int &,
                             int &, int &, int &) = 0;
  virtual void read_atoms(int, int, double **) = 0;

  virtual void open_file(const char *);
  virtual void close_file();

 protected:
  FILE *fp;          // pointer to opened file or pipe
  int compressed;    // flag for dump file compression
};

}    // namespace LAMMPS_NS

#endif

/* ERROR/WARNING messages:

E: Cannot open gzipped file

LAMMPS was compiled without support for reading and writing gzipped
files through a pipeline to the gzip program with -DLAMMPS_GZIP.

E: Cannot open file %s

The specified file cannot be opened.  Check that the path and name are
correct. If the file is a compressed file, also check that the gzip
executable can be found and run.

*/
