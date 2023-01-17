// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Contributing author: Maxim Shugaev (UVA), mvs9t@virginia.edu
------------------------------------------------------------------------- */

#include "atom_vec_mesont.h"
#include "atom.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecMesoNT::AtomVecMesoNT(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::ATOMIC;
  mass_type = PER_TYPE;

  atom->mesont_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = (char *) "rmass radius length buckling bond_nt molecule";
  fields_copy = (char *) "rmass radius length buckling bond_nt molecule";
  fields_comm = (char *) "";
  fields_comm_vel = (char *) "";
  fields_reverse = (char *) "";
  fields_border = (char *) "rmass radius length buckling bond_nt molecule";
  fields_border_vel = (char *) "rmass radius length buckling bond_nt molecule";
  fields_exchange = (char *) "rmass radius length buckling bond_nt molecule";
  fields_restart = (char *) "rmass radius length buckling bond_nt molecule";
  fields_create = (char *) "rmass radius length buckling bond_nt molecule";
  fields_data_atom = (char *) "id molecule type bond_nt rmass radius length buckling x";
  fields_data_vel = (char *) "id v";

  setup_fields();
}
