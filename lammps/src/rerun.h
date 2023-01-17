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

#ifdef COMMAND_CLASS
// clang-format off
CommandStyle(rerun,Rerun);
// clang-format on
#else

#ifndef LMP_RERUN_H
#define LMP_RERUN_H

#include "command.h"

namespace LAMMPS_NS {

class Rerun : public Command {
 public:
  Rerun(class LAMMPS *);
  void command(int, char **);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Rerun command before simulation box is defined

The rerun command cannot be used before a read_data, read_restart, or
create_box command.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Rerun dump file does not contain requested snapshot

Self-explanatory.

E: Read rerun dump file timestep > specified stop

Self-explanatory.

*/
