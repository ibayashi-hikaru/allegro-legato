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
CommandStyle(read_data,ReadData);
// clang-format on
#else

#ifndef LMP_READ_DATA_H
#define LMP_READ_DATA_H

#include "command.h"

namespace LAMMPS_NS {

class ReadData : public Command {
 public:
  ReadData(class LAMMPS *);
  ~ReadData();
  void command(int, char **);

 private:
  int me, compressed;
  char *line, *keyword, *buffer, *style;
  FILE *fp;
  char **coeffarg;
  int ncoeffarg, maxcoeffarg;
  char argoffset1[8], argoffset2[8];

  bigint id_offset, mol_offset;

  int nlocal_previous;
  bigint natoms;
  bigint nbonds, nangles, ndihedrals, nimpropers;
  int ntypes;
  int nbondtypes, nangletypes, ndihedraltypes, nimpropertypes;

  bigint nellipsoids;
  class AtomVecEllipsoid *avec_ellipsoid;
  bigint nlines;
  class AtomVecLine *avec_line;
  bigint ntris;
  class AtomVecTri *avec_tri;
  bigint nbodies;
  class AtomVecBody *avec_body;

  // box info

  double boxlo[3], boxhi[3];
  double xy, xz, yz;
  int triclinic;

  // optional args

  int addflag, offsetflag, shiftflag, coeffflag;
  tagint addvalue;
  int toffset, boffset, aoffset, doffset, ioffset;
  double shift[3];
  int extra_atom_types, extra_bond_types, extra_angle_types;
  int extra_dihedral_types, extra_improper_types;
  int groupbit;

  int nfix;
  int *fix_index;
  char **fix_header;
  char **fix_section;

  // methods

  void open(char *);
  void scan(int &, int &, int &, int &);
  int reallocate(int **, int, int);
  void header(int);
  void parse_keyword(int);
  void skip_lines(bigint);
  void parse_coeffs(char *, const char *, int, int, int);
  int style_match(const char *, const char *);

  void atoms();
  void velocities();

  void bonds(int);
  void bond_scan(int, char *, int *);
  void angles(int);
  void dihedrals(int);
  void impropers(int);

  void bonus(bigint, class AtomVec *, const char *);
  void bodies(int, class AtomVec *);

  void mass();
  void paircoeffs();
  void pairIJcoeffs();
  void bondcoeffs();
  void anglecoeffs(int);
  void dihedralcoeffs(int);
  void impropercoeffs(int);

  void fix(int, char *);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Read data add atomID offset is too big

UNDOCUMENTED

E: Read data add molID offset is too big

UNDOCUMENTED

E: Non-zero read_data shift z value for 2d simulation

Self-explanatory.

E: No bonds allowed with this atom style

Self-explanatory.

E: No angles allowed with this atom style

Self-explanatory.

E: No dihedrals allowed with this atom style

Self-explanatory.

E: No impropers allowed with this atom style

Self-explanatory.

E: No bonded interactions allowed with this atom style

UNDOCUMENTED

E: Fix ID for read_data does not exist

Self-explanatory.

E: Cannot run 2d simulation with non-periodic Z dimension

Use the boundary command to make the z dimension periodic in order to
run a 2d simulation.

E: Cannot read_data without add keyword after simulation box is defined

Self-explanatory.

E: Cannot use read_data add before simulation box is defined

Self-explanatory.

E: Cannot use read_data offset without add flag

Self-explanatory.

E: Cannot use read_data shift without add flag

Self-explanatory.

E: Cannot use read_data extra with add flag

Self-explanatory.

W: Atom style in data file differs from currently defined atom style

Self-explanatory.

E: Must read Atoms before Velocities

The Atoms section of a data file must come before a Velocities
section.

E: Invalid data file section: Bonds

Atom style does not allow bonds.

E: Must read Atoms before Bonds

The Atoms section of a data file must come before a Bonds section.

E: Invalid data file section: Angles

Atom style does not allow angles.

E: Must read Atoms before Angles

The Atoms section of a data file must come before an Angles section.

E: Invalid data file section: Dihedrals

Atom style does not allow dihedrals.

E: Must read Atoms before Dihedrals

The Atoms section of a data file must come before a Dihedrals section.

E: Invalid data file section: Impropers

Atom style does not allow impropers.

E: Must read Atoms before Impropers

The Atoms section of a data file must come before an Impropers
section.

E: Invalid data file section: Ellipsoids

Atom style does not allow ellipsoids.

E: Must read Atoms before Ellipsoids

The Atoms section of a data file must come before a Ellipsoids
section.

E: Invalid data file section: Lines

Atom style does not allow lines.

E: Must read Atoms before Lines

The Atoms section of a data file must come before a Lines section.

E: Invalid data file section: Triangles

Atom style does not allow triangles.

E: Must read Atoms before Triangles

The Atoms section of a data file must come before a Triangles section.

E: Invalid data file section: Bodies

Atom style does not allow bodies.

E: Must read Atoms before Bodies

The Atoms section of a data file must come before a Bodies section.

E: Must define pair_style before Pair Coeffs

Must use a pair_style command before reading a data file that defines
Pair Coeffs.

W: Pair style in data file differs from currently defined pair style

Self-explanatory.

E: Must define pair_style before PairIJ Coeffs

Must use a pair_style command before reading a data file that defines
PairIJ Coeffs.

E: Invalid data file section: Bond Coeffs

Atom style does not allow bonds.

E: Must define bond_style before Bond Coeffs

Must use a bond_style command before reading a data file that
defines Bond Coeffs.

W: Bond style in data file differs from currently defined bond style

Self-explanatory.

E: Invalid data file section: Angle Coeffs

Atom style does not allow angles.

E: Must define angle_style before Angle Coeffs

Must use an angle_style command before reading a data file that
defines Angle Coeffs.

W: Angle style in data file differs from currently defined angle style

Self-explanatory.

E: Invalid data file section: Dihedral Coeffs

Atom style does not allow dihedrals.

E: Must define dihedral_style before Dihedral Coeffs

Must use a dihedral_style command before reading a data file that
defines Dihedral Coeffs.

W: Dihedral style in data file differs from currently defined dihedral style

Self-explanatory.

E: Invalid data file section: Improper Coeffs

Atom style does not allow impropers.

E: Must define improper_style before Improper Coeffs

Must use an improper_style command before reading a data file that
defines Improper Coeffs.

W: Improper style in data file differs from currently defined improper style

Self-explanatory.

E: Invalid data file section: BondBond Coeffs

Atom style does not allow angles.

E: Must define angle_style before BondBond Coeffs

Must use an angle_style command before reading a data file that
defines Angle Coeffs.

E: Invalid data file section: BondAngle Coeffs

Atom style does not allow angles.

E: Must define angle_style before BondAngle Coeffs

Must use an angle_style command before reading a data file that
defines Angle Coeffs.

E: Invalid data file section: MiddleBondTorsion Coeffs

Atom style does not allow dihedrals.

E: Must define dihedral_style before MiddleBondTorsion Coeffs

Must use a dihedral_style command before reading a data file that
defines MiddleBondTorsion Coeffs.

E: Invalid data file section: EndBondTorsion Coeffs

Atom style does not allow dihedrals.

E: Must define dihedral_style before EndBondTorsion Coeffs

Must use a dihedral_style command before reading a data file that
defines EndBondTorsion Coeffs.

E: Invalid data file section: AngleTorsion Coeffs

Atom style does not allow dihedrals.

E: Must define dihedral_style before AngleTorsion Coeffs

Must use a dihedral_style command before reading a data file that
defines AngleTorsion Coeffs.

E: Invalid data file section: AngleAngleTorsion Coeffs

Atom style does not allow dihedrals.

E: Must define dihedral_style before AngleAngleTorsion Coeffs

Must use a dihedral_style command before reading a data file that
defines AngleAngleTorsion Coeffs.

E: Invalid data file section: BondBond13 Coeffs

Atom style does not allow dihedrals.

E: Must define dihedral_style before BondBond13 Coeffs

Must use a dihedral_style command before reading a data file that
defines BondBond13 Coeffs.

E: Invalid data file section: AngleAngle Coeffs

Atom style does not allow impropers.

E: Must define improper_style before AngleAngle Coeffs

Must use an improper_style command before reading a data file that
defines AngleAngle Coeffs.

E: Unknown identifier in data file: %s

A section of the data file cannot be read by LAMMPS.

E: No atoms in data file

The header of the data file indicated that atoms would be included,
but they are not present.

E: Needed molecular topology not in data file

The header of the data file indicated bonds, angles, etc would be
included, but they are not present.

E: Needed bonus data not in data file

Some atom styles require bonus data.  See the read_data doc page for
details.

E: Read_data shrink wrap did not assign all atoms correctly

This is typically because the box-size specified in the data file is
large compared to the actual extent of atoms in a shrink-wrapped
dimension.  When LAMMPS shrink-wraps the box atoms will be lost if the
processor they are re-assigned to is too far away.  Choose a box
size closer to the actual extent of the atoms.

E: Unexpected end of data file

LAMMPS hit the end of the data file while attempting to read a
section.  Something is wrong with the format of the data file.

E: No ellipsoids allowed with this atom style

Self-explanatory.  Check data file.

E: No lines allowed with this atom style

Self-explanatory.  Check data file.

E: No triangles allowed with this atom style

Self-explanatory.  Check data file.

E: No bodies allowed with this atom style

Self-explanatory.  Check data file.

E: System in data file is too big

See the setting for bigint in the src/lmptype.h file.

E: Bonds defined but no bond types

The data file header lists bonds but no bond types.

E: Angles defined but no angle types

The data file header lists angles but no angle types.

E: Dihedrals defined but no dihedral types

The data file header lists dihedrals but no dihedral types.

E: Impropers defined but no improper types

The data file header lists improper but no improper types.

E: No molecule topology allowed with atom style template

The data file cannot specify the number of bonds, angles, etc,
because this info if inferred from the molecule templates.

E: Did not assign all atoms correctly

Atoms read in from a data file were not assigned correctly to
processors.  This is likely due to some atom coordinates being
outside a non-periodic simulation box.

E: Subsequent read data induced too many bonds per atom

See the extra/bond/per/atom keyword for the create_box
or the read_data command to set this limit larger.

E: Bonds assigned incorrectly

Bonds read in from the data file were not assigned correctly to atoms.
This means there is something invalid about the topology definitions.

E: Subsequent read data induced too many angles per atom

See the extra/angle/per/atom keyword for the create_box
or the read_data command to set this limit larger.

E: Angles assigned incorrectly

Angles read in from the data file were not assigned correctly to
atoms.  This means there is something invalid about the topology
definitions.

E: Subsequent read data induced too many dihedrals per atom

See the extra/dihedral/per/atom keyword for the create_box
or the read_data command to set this limit larger.

E: Dihedrals assigned incorrectly

Dihedrals read in from the data file were not assigned correctly to
atoms.  This means there is something invalid about the topology
definitions.

E: Subsequent read data induced too many impropers per atom

See the extra/improper/per/atom keyword for the create_box
or the read_data command to set this limit larger.

E: Impropers assigned incorrectly

Impropers read in from the data file were not assigned correctly to
atoms.  This means there is something invalid about the topology
definitions.

E: Too few values in body lines in data file

Self-explanatory.

E: Too many values in body lines in data file

Self-explanatory.

E: Too many lines in one body in data file - boost MAXBODY

MAXBODY is a setting at the top of the src/read_data.cpp file.
Set it larger and re-compile the code.

E: Unexpected empty line in PairCoeffs section

Read a blank line where there should be coefficient data.

E: Unexpected empty line in BondCoeffs section

Read a blank line where there should be coefficient data.

E: Unexpected empty line in AngleCoeffs section

Read a blank line where there should be coefficient data.

E: Unexpected empty line in DihedralCoeffs section

Read a blank line where there should be coefficient data.

E: Unexpected empty line in ImproperCoeffs section

Read a blank line where there should be coefficient data.

E: Cannot open gzipped file

LAMMPS was compiled without support for reading and writing gzipped
files through a pipeline to the gzip program with -DLAMMPS_GZIP.

E: Cannot open file %s

The specified file cannot be opened.  Check that the path and name are
correct. If the file is a compressed file, also check that the gzip
executable can be found and run.

U: Read data add offset is too big

It cannot be larger than the size of atom IDs, e.g. the maximum 32-bit
integer.

*/
