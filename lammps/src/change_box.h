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
CommandStyle(change_box,ChangeBox);
// clang-format on
#else

#ifndef LMP_CHANGE_BOX_H
#define LMP_CHANGE_BOX_H

#include "command.h"

namespace LAMMPS_NS {

class ChangeBox : public Command {
 public:
  ChangeBox(class LAMMPS *);
  void command(int, char **);

 private:
  int scaleflag;
  double scale[3];

  struct Operation {
    int style, flavor;
    int dim, boundindex;
    int vdim1, vdim2;
    double flo, fhi, ftilt;
    double dlo, dhi, dtilt;
    double scale;
  };

  Operation *ops;
  int nops;

  double boxlo[3], h_inv[6];

  void options(int, char **);
  void save_box_state();
  void volume_preserve(int, int, double);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Change_box command before simulation box is defined

Self-explanatory.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Cannot change_box after reading restart file with per-atom info

This is because the restart file info cannot be migrated with the
atoms.  You can get around this by performing a 0-timestep run which
will assign the restart file info to actual atoms.

E: Could not find change_box group ID

Group ID used in the change_box command does not exist.

E: Cannot change_box in z dimension for 2d simulation

Self-explanatory.

E: Change_box volume used incorrectly

The "dim volume" option must be used immediately following one or two
settings for "dim1 ..." (and optionally "dim2 ...") and must be for a
different dimension, i.e. dim != dim1 and dim != dim2.

E: Cannot change_box in xz or yz for 2d simulation

Self-explanatory.

E: Cannot change box tilt factors for orthogonal box

Cannot use tilt factors unless the simulation box is non-orthogonal.

E: Cannot change box z boundary to non-periodic for a 2d simulation

Self-explanatory.

E: Cannot change box to orthogonal when tilt is non-zero

Self-explanatory.

E: Cannot change box ortho/triclinic with dumps defined

This is because some dumps store the shape of the box.  You need to
use undump to discard the dump, change the box, then redefine a new
dump.

E: Cannot change box ortho/triclinic with certain fixes defined

This is because those fixes store the shape of the box.  You need to
use unfix to discard the fix, change the box, then redefine a new
fix.

W: Attempting to remap atoms in rigid bodies

UNDOCUMENTED

W: Lost atoms via change_box: original %ld current %ld

The command options you have used caused atoms to be lost.

*/
