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

#ifndef LMP_DUMP_H
#define LMP_DUMP_H

#include "pointers.h"    // IWYU pragma: export

namespace LAMMPS_NS {

class Dump : protected Pointers {
 public:
  char *id;                // user-defined name of Dump
  char *style;             // style of Dump
  char *filename;          // user-specified file
  int igroup, groupbit;    // group that Dump is performed on

  int first_flag;    // 0 if no initial dump, 1 if yes initial dump
  int clearstep;     // 1 if dump invokes computes, 0 if not

  int comm_forward;    // size of forward communication (0 if none)
  int comm_reverse;    // size of reverse communication (0 if none)

#if defined(LMP_QSORT)
  // static variable across all Dump objects
  static Dump *dumpptr;    // holds a ptr to Dump currently being used
#endif

  Dump(class LAMMPS *, int, char **);
  virtual ~Dump();
  void init();
  virtual void write();

  virtual int pack_forward_comm(int, int *, double *, int, int *) { return 0; }
  virtual void unpack_forward_comm(int, int, double *) {}
  virtual int pack_reverse_comm(int, int, double *) { return 0; }
  virtual void unpack_reverse_comm(int, int *, double *) {}

  void modify_params(int, char **);
  virtual double memory_usage();

 protected:
  int me, nprocs;    // proc info

  int compressed;          // 1 if dump file is written compressed, 0 no
  int binary;              // 1 if dump file is written binary, 0 no
  int multifile;           // 0 = one big file, 1 = one file per timestep
  int multiproc;           // 0 = proc 0 writes for all,
                           // else # of procs writing files
  int nclusterprocs;       // # of procs in my cluster that write to one file
  int filewriter;          // 1 if this proc writes a file, else 0
  int fileproc;            // ID of proc in my cluster who writes to file
  char *multiname;         // filename with % converted to cluster ID
  MPI_Comm clustercomm;    // MPI communicator within my cluster of procs

  int header_flag;          // 0 = item, 2 = xyz
  int flush_flag;           // 0 if no flush, 1 if flush every dump
  int sort_flag;            // 1 if sorted output
  int append_flag;          // 1 if open file in append mode, 0 if not
  int buffer_allow;         // 1 if style allows for buffer_flag, 0 if not
  int buffer_flag;          // 1 if buffer output as one big string, 0 if not
  int padflag;              // timestep padding in filename
  int pbcflag;              // 1 if remap dumped atoms via PBC, 0 if not
  int singlefile_opened;    // 1 = one big file, already opened, else 0
  int sortcol;              // 0 to sort on ID, 1-N on columns
  int sortcolm1;            // sortcol - 1
  int sortorder;            // ASCEND or DESCEND
  int time_flag;            // 1 if output accumulated time
  int unit_flag;            // 1 if dump should contain unit information
  int unit_count;           // # of times the unit information was written
  int delay_flag;           // 1 if delay output until delaystep
  int write_header_flag;    // 1 if write header, 0 if not

  bigint delaystep;

  int refreshflag;    // 1 if dump_modify refresh specified
  char *refresh;      // compute ID to invoke refresh() on
  int irefresh;       // index of compute

  char boundstr[9];    // encoding of boundary flags

  char *format;            // format string for the file write
  char *format_default;    // default format string

  char *format_line_user;    // user-specified format strings
  char *format_float_user;
  char *format_int_user;
  char *format_bigint_user;
  char **format_column_user;
  enum { INT, DOUBLE, STRING, BIGINT };

  FILE *fp;        // file to write dump to
  int size_one;    // # of quantities for one atom
  int nme;         // # of atoms in this dump from me
  int nsme;        // # of chars in string output from me

  double boxxlo, boxxhi;    // local copies of domain values
  double boxylo, boxyhi;    // lo/hi are bounding box for triclinic
  double boxzlo, boxzhi;
  double boxxy, boxxz, boxyz;

  int maxfiles;        // max number of files created, -1 == infinite
  int numfiles;        // number of files in names list
  int fileidx;         // index of file in names list
  char **nameslist;    // list of history file names

  bigint ntotal;         // total # of per-atom lines in snapshot
  int reorderflag;       // 1 if OK to reorder instead of sort
  int ntotal_reorder;    // # of atoms that must be in snapshot
  int nme_reorder;       // # of atoms I must own in snapshot
  tagint idlo;           // lowest ID I own when reordering

  int maxbuf;     // size of buf
  double *buf;    // memory for atom quantities
  int maxsbuf;    // size of sbuf
  char *sbuf;     // memory for atom quantities in string format

  int maxids;     // size of ids
  int maxsort;    // size of bufsort, idsort, index
  int maxproc;    // size of proclist
  tagint *ids;    // list of atom IDs, if sorting on IDs
  double *bufsort;
  tagint *idsort;
  int *index, *proclist;

  double **xpbc, **vpbc;
  imageint *imagepbc;
  int maxpbc;

  class Irregular *irregular;

  virtual void init_style() = 0;
  virtual void openfile();
  virtual int modify_param(int, char **) { return 0; }
  virtual void write_header(bigint) = 0;
  virtual int count();
  virtual void pack(tagint *) = 0;
  virtual int convert_string(int, double *) { return 0; }
  virtual void write_data(int, double *) = 0;
  void pbc_allocate();
  double compute_time();

  void sort();
#if defined(LMP_QSORT)
  static int idcompare(const void *, const void *);
  static int bufcompare(const void *, const void *);
  static int bufcompare_reverse(const void *, const void *);
#else
  static int idcompare(const int, const int, void *);
  static int bufcompare(const int, const int, void *);
  static int bufcompare_reverse(const int, const int, void *);
#endif
};

}    // namespace LAMMPS_NS

#endif

/* ERROR/WARNING messages:

E: Dump file MPI-IO output not allowed with % in filename

This is because a % signifies one file per processor and MPI-IO
creates one large file for all processors.

E: Cannot dump sort when multiple dump files are written

In this mode, each processor dumps its atoms to a file, so
no sorting is allowed.

E: Cannot dump sort on atom IDs with no atom IDs defined

Self-explanatory.

E: Dump sort column is invalid

Self-explanatory.

E: Too many atoms to dump sort

Cannot sort when running with more than 2^31 atoms.

E: Dump could not find refresh compute ID

UNDOCUMENTED

E: Too much per-proc info for dump

Number of local atoms times number of columns must fit in a 32-bit
integer for dump.

E: Too much buffered per-proc info for dump

The size of the buffered string must fit in a 32-bit integer for a
dump.

E: Cannot open gzipped file

LAMMPS was compiled without support for reading and writing gzipped
files through a pipeline to the gzip program with -DLAMMPS_GZIP.

E: Cannot open dump file

Self-explanatory.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Dump_modify buffer yes not allowed for this style

Self-explanatory.

E: Cannot use dump_modify fileper without % in dump file name

Self-explanatory.

E: Cannot use dump_modify nfile without % in dump file name

Self-explanatory.

*/
