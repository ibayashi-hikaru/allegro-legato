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

/* ----------------------------------------------------------------------
   Contributing author: Rezwanur Rahman, John Foster (UTSA)
------------------------------------------------------------------------- */

#include "compute_plasticity_atom.h"
#include <cstring>
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "fix_peri_neigh.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputePlasticityAtom::
ComputePlasticityAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute plasticity/atom command");

  if (!force->pair_match("peri/eps",1))
    error->all(FLERR,"Compute plasticity/atom cannot be used "
               "with this pair style");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  plasticity = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputePlasticityAtom::~ComputePlasticityAtom()
{
  memory->destroy(plasticity);
}

/* ---------------------------------------------------------------------- */

void ComputePlasticityAtom::init()
{
  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"plasticity/peri") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute plasticity/atom");

  // find associated PERI_NEIGH fix that must exist

  ifix_peri = modify->find_fix_by_style("^PERI_NEIGH");
  if (ifix_peri == -1)
    error->all(FLERR,"Compute plasticity/atom requires a Peridynamics pair style");
}

/* ---------------------------------------------------------------------- */

void ComputePlasticityAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow damage array if necessary

  if (atom->nmax > nmax) {
    memory->destroy(plasticity);
    nmax = atom->nmax;
    memory->create(plasticity,nmax,"plasticity/atom:plasticity");
    vector_atom = plasticity;
  }

  // extract plasticity for each atom in group

  double *lambdaValue = ((FixPeriNeigh *) modify->fix[ifix_peri])->lambdaValue;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) plasticity[i] = lambdaValue[i];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputePlasticityAtom::memory_usage()
{
  double bytes = (double)nmax * sizeof(double);
  return bytes;
}
