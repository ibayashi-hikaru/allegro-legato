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

#include "compute_sph_rho_atom.h"
#include <cstring>
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSPHRhoAtom::ComputeSPHRhoAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute sph/rho/atom command");
  if (atom->rho_flag != 1)
    error->all(FLERR,"Compute sph/rho/atom command requires atom_style sph");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  rhoVector = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputeSPHRhoAtom::~ComputeSPHRhoAtom()
{
  memory->sfree(rhoVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSPHRhoAtom::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"rhoVector/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute rhoVector/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeSPHRhoAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow rhoVector array if necessary

  if (atom->nmax > nmax) {
    memory->sfree(rhoVector);
    nmax = atom->nmax;
    rhoVector = (double *) memory->smalloc(nmax*sizeof(double),"atom:rhoVector");
    vector_atom = rhoVector;
  }

  // compute kinetic energy for each atom in group

  double *rho = atom->rho;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
              rhoVector[i] = rho[i];
      }
      else {
              rhoVector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSPHRhoAtom::memory_usage()
{
  double bytes = (double)nmax * sizeof(double);
  return bytes;
}
