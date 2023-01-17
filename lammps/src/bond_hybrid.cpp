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

#include "bond_hybrid.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"

#include <cctype>
#include <cstring>

using namespace LAMMPS_NS;

#define EXTRA 1000

/* ---------------------------------------------------------------------- */

BondHybrid::BondHybrid(LAMMPS *lmp) : Bond(lmp)
{
  writedata = 0;
  nstyles = 0;
  has_quartic = -1;
  nbondlist = nullptr;
  maxbond = nullptr;
  bondlist = nullptr;
}

/* ---------------------------------------------------------------------- */

BondHybrid::~BondHybrid()
{
  if (nstyles) {
    for (int i = 0; i < nstyles; i++) delete styles[i];
    delete[] styles;
    for (int i = 0; i < nstyles; i++) delete[] keywords[i];
    delete[] keywords;
  }

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(map);
    delete[] nbondlist;
    delete[] maxbond;
    for (int i = 0; i < nstyles; i++) memory->destroy(bondlist[i]);
    delete[] bondlist;
  }
}

/* ---------------------------------------------------------------------- */

void BondHybrid::compute(int eflag, int vflag)
{
  int i, j, m, n;

  // save ptrs to original bondlist

  int nbondlist_orig = neighbor->nbondlist;
  int **bondlist_orig = neighbor->bondlist;

  // if this is re-neighbor step, create sub-style bondlists
  // nbondlist[] = length of each sub-style list
  // realloc sub-style bondlist if necessary
  // load sub-style bondlist with 3 values from original bondlist

  if (neighbor->ago == 0) {
    for (m = 0; m < nstyles; m++) nbondlist[m] = 0;
    for (i = 0; i < nbondlist_orig; i++) {
      m = map[bondlist_orig[i][2]];
      if (m >= 0) nbondlist[m]++;
    }
    for (m = 0; m < nstyles; m++) {
      if (nbondlist[m] > maxbond[m]) {
        memory->destroy(bondlist[m]);
        maxbond[m] = nbondlist[m] + EXTRA;
        memory->create(bondlist[m], maxbond[m], 3, "bond_hybrid:bondlist");
      }
      nbondlist[m] = 0;
    }
    for (i = 0; i < nbondlist_orig; i++) {
      m = map[bondlist_orig[i][2]];
      if (m < 0) continue;
      n = nbondlist[m];
      bondlist[m][n][0] = bondlist_orig[i][0];
      bondlist[m][n][1] = bondlist_orig[i][1];
      bondlist[m][n][2] = bondlist_orig[i][2];
      nbondlist[m]++;
    }
  }

  // call each sub-style's compute function
  // set neighbor->bondlist to sub-style bondlist before call
  // accumulate sub-style global/peratom energy/virial in hybrid

  ev_init(eflag, vflag);

  // need to clear per-thread storage once here, when using multiple threads
  // with thread-enabled substyles to avoid uninitlialized data access.

  const int nthreads = comm->nthreads;
  if (nthreads > 1) {
    const bigint nall = atom->nlocal + atom->nghost;
    if (eflag_atom) memset(&eatom[0], 0, nall * nthreads * sizeof(double));
    if (vflag_atom) memset(&vatom[0][0], 0, 6 * nall * nthreads * sizeof(double));
  }

  for (m = 0; m < nstyles; m++) {
    neighbor->nbondlist = nbondlist[m];
    neighbor->bondlist = bondlist[m];

    styles[m]->compute(eflag, vflag);

    if (eflag_global) energy += styles[m]->energy;
    if (vflag_global)
      for (n = 0; n < 6; n++) virial[n] += styles[m]->virial[n];
    if (eflag_atom) {
      n = atom->nlocal;
      if (force->newton_bond) n += atom->nghost;
      double *eatom_substyle = styles[m]->eatom;
      for (i = 0; i < n; i++) eatom[i] += eatom_substyle[i];
    }
    if (vflag_atom) {
      n = atom->nlocal;
      if (force->newton_bond) n += atom->nghost;
      double **vatom_substyle = styles[m]->vatom;
      for (i = 0; i < n; i++)
        for (j = 0; j < 6; j++) vatom[i][j] += vatom_substyle[i][j];
    }
  }

  // restore ptrs to original bondlist

  neighbor->nbondlist = nbondlist_orig;
  neighbor->bondlist = bondlist_orig;
}

/* ---------------------------------------------------------------------- */

void BondHybrid::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(map, n + 1, "bond:map");
  memory->create(setflag, n + 1, "bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;

  nbondlist = new int[nstyles];
  maxbond = new int[nstyles];
  bondlist = new int **[nstyles];
  for (int m = 0; m < nstyles; m++) maxbond[m] = 0;
  for (int m = 0; m < nstyles; m++) bondlist[m] = nullptr;
}

/* ----------------------------------------------------------------------
   create one bond style for each arg in list
------------------------------------------------------------------------- */

void BondHybrid::settings(int narg, char **arg)
{
  int i, m, istyle;

  if (narg < 1) error->all(FLERR, "Illegal bond_style command");

  // delete old lists, since cannot just change settings

  if (nstyles) {
    for (i = 0; i < nstyles; i++) delete styles[i];
    delete[] styles;
    for (i = 0; i < nstyles; i++) delete[] keywords[i];
    delete[] keywords;
    has_quartic = -1;
  }

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(map);
    delete[] nbondlist;
    delete[] maxbond;
    for (i = 0; i < nstyles; i++) memory->destroy(bondlist[i]);
    delete[] bondlist;
  }
  allocated = 0;

  // count sub-styles by skipping numeric args
  // one exception is 1st arg of style "table", which is non-numeric word
  // need a better way to skip these exceptions

  nstyles = 0;
  i = 0;
  while (i < narg) {
    if (strcmp(arg[i], "table") == 0) i++;
    i++;
    while (i < narg && !isalpha(arg[i][0])) i++;
    nstyles++;
  }

  // allocate list of sub-styles

  styles = new Bond *[nstyles];
  keywords = new char *[nstyles];

  // allocate each sub-style and call its settings() with subset of args
  // allocate uses suffix, but don't store suffix version in keywords,
  //   else syntax in coeff() will not match
  // define subset of args for a sub-style by skipping numeric args
  // one exception is 1st arg of style "table", which is non-numeric
  // need a better way to skip these exceptions

  int dummy;
  nstyles = 0;
  i = 0;

  while (i < narg) {

    for (m = 0; m < nstyles; m++)
      if (strcmp(arg[i], keywords[m]) == 0)
        error->all(FLERR, "Bond style hybrid cannot use same bond style twice");

    if (strcmp(arg[i], "hybrid") == 0)
      error->all(FLERR, "Bond style hybrid cannot have hybrid as an argument");

    if (strcmp(arg[i], "none") == 0)
      error->all(FLERR, "Bond style hybrid cannot have none as an argument");

    // register index of quartic bond type,
    // so that bond type 0 can be mapped to it

    if (strncmp(arg[i], "quartic", 7) == 0) has_quartic = m;

    styles[nstyles] = force->new_bond(arg[i], 1, dummy);
    force->store_style(keywords[nstyles], arg[i], 0);

    istyle = i;
    if (strcmp(arg[i], "table") == 0) i++;
    i++;
    while (i < narg && !isalpha(arg[i][0])) i++;
    styles[nstyles]->settings(i - istyle - 1, &arg[istyle + 1]);
    nstyles++;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one type
---------------------------------------------------------------------- */

void BondHybrid::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);

  // 2nd arg = bond sub-style name
  // allow for "none" as valid sub-style name

  int m;
  for (m = 0; m < nstyles; m++)
    if (strcmp(arg[1], keywords[m]) == 0) break;

  int none = 0;
  if (m == nstyles) {
    if (strcmp(arg[1], "none") == 0)
      none = 1;
    else
      error->all(FLERR, "Bond coeff for hybrid has invalid style");
  }

  // move 1st arg to 2nd arg
  // just copy ptrs, since arg[] points into original input line

  arg[1] = arg[0];

  // invoke sub-style coeff() starting with 1st arg

  if (!none) styles[m]->coeff(narg - 1, &arg[1]);

  // set setflag and which type maps to which sub-style
  // if sub-style is none: set hybrid setflag, wipe out map

  for (int i = ilo; i <= ihi; i++) {
    setflag[i] = 1;
    if (none)
      map[i] = -1;
    else
      map[i] = m;
  }
}

/* ---------------------------------------------------------------------- */

void BondHybrid::init_style()
{
  for (int m = 0; m < nstyles; m++)
    if (styles[m]) styles[m]->init_style();

  // bond style quartic will set broken bonds to bond type 0, so we need
  // to create an entry for it in the bond type to sub-style map

  if (has_quartic >= 0) map[0] = has_quartic;
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondHybrid::equilibrium_distance(int i)
{
  if (map[i] < 0) error->one(FLERR, "Invoked bond equil distance on bond style none");
  return styles[map[i]]->equilibrium_distance(i);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void BondHybrid::write_restart(FILE *fp)
{
  fwrite(&nstyles, sizeof(int), 1, fp);

  int n;
  for (int m = 0; m < nstyles; m++) {
    n = strlen(keywords[m]) + 1;
    fwrite(&n, sizeof(int), 1, fp);
    fwrite(keywords[m], sizeof(char), n, fp);
    styles[m]->write_restart_settings(fp);
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void BondHybrid::read_restart(FILE *fp)
{
  int me = comm->me;
  if (me == 0) utils::sfread(FLERR, &nstyles, sizeof(int), 1, fp, nullptr, error);
  MPI_Bcast(&nstyles, 1, MPI_INT, 0, world);
  styles = new Bond *[nstyles];
  keywords = new char *[nstyles];

  allocate();

  int n, dummy;
  for (int m = 0; m < nstyles; m++) {
    if (me == 0) utils::sfread(FLERR, &n, sizeof(int), 1, fp, nullptr, error);
    MPI_Bcast(&n, 1, MPI_INT, 0, world);
    keywords[m] = new char[n];
    if (me == 0) utils::sfread(FLERR, keywords[m], sizeof(char), n, fp, nullptr, error);
    MPI_Bcast(keywords[m], n, MPI_CHAR, 0, world);
    styles[m] = force->new_bond(keywords[m], 0, dummy);
    styles[m]->read_restart_settings(fp);
  }
}

/* ---------------------------------------------------------------------- */

double BondHybrid::single(int type, double rsq, int i, int j, double &fforce)

{
  if (map[type] < 0) error->one(FLERR, "Invoked bond single on bond style none");
  return styles[map[type]]->single(type, rsq, i, j, fforce);
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double BondHybrid::memory_usage()
{
  double bytes = (double) maxeatom * sizeof(double);
  bytes += (double) maxvatom * 6 * sizeof(double);
  for (int m = 0; m < nstyles; m++) bytes += (double) maxbond[m] * 3 * sizeof(int);
  for (int m = 0; m < nstyles; m++)
    if (styles[m]) bytes += styles[m]->memory_usage();
  return bytes;
}
