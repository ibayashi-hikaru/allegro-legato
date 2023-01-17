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
------------------------------------------------------------------------- */

#include "pair_airebo_morse.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairAIREBOMorse::PairAIREBOMorse(LAMMPS *lmp) : PairAIREBO(lmp) {
  variant = AIREBO_M;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairAIREBOMorse::settings(int narg, char **arg)
{
  PairAIREBO::settings(narg,arg);

  morseflag = 1;
}
