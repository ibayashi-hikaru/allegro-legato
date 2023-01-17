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

#include "compute_sna_atom.h"
#include <cstring>

#include "sna.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

ComputeSNAAtom::ComputeSNAAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), cutsq(nullptr), list(nullptr), sna(nullptr),
  radelem(nullptr), wjelem(nullptr)
{
  double rmin0, rfac0;
  int twojmax, switchflag, bzeroflag, bnormflag, wselfallflag;
  radelem = nullptr;
  wjelem = nullptr;

  int ntypes = atom->ntypes;
  int nargmin = 6+2*ntypes;

  if (narg < nargmin) error->all(FLERR,"Illegal compute sna/atom command");

  // default values

  rmin0 = 0.0;
  switchflag = 1;
  bzeroflag = 1;
  bnormflag = 0;
  quadraticflag = 0;
  chemflag = 0;
  bnormflag = 0;
  wselfallflag = 0;
  nelements = 1;

  // offset by 1 to match up with types

  memory->create(radelem,ntypes+1,"sna/atom:radelem");
  memory->create(wjelem,ntypes+1,"sna/atom:wjelem");

  rcutfac = atof(arg[3]);
  rfac0 = atof(arg[4]);
  twojmax = atoi(arg[5]);

  for (int i = 0; i < ntypes; i++)
    radelem[i+1] = atof(arg[6+i]);
  for (int i = 0; i < ntypes; i++)
    wjelem[i+1] = atof(arg[6+ntypes+i]);

  // construct cutsq

  double cut;
  cutmax = 0.0;
  memory->create(cutsq,ntypes+1,ntypes+1,"sna/atom:cutsq");
  for (int i = 1; i <= ntypes; i++) {
    cut = 2.0*radelem[i]*rcutfac;
    if (cut > cutmax) cutmax = cut;
    cutsq[i][i] = cut*cut;
    for (int j = i+1; j <= ntypes; j++) {
      cut = (radelem[i]+radelem[j])*rcutfac;
      cutsq[i][j] = cutsq[j][i] = cut*cut;
    }
  }

  // process optional args

  int iarg = nargmin;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"rmin0") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute sna/atom command");
      rmin0 = atof(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"switchflag") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute sna/atom command");
      switchflag = atoi(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"bzeroflag") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute sna/atom command");
      bzeroflag = atoi(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"quadraticflag") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute sna/atom command");
      quadraticflag = atoi(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"chem") == 0) {
      if (iarg+2+ntypes > narg)
        error->all(FLERR,"Illegal compute sna/atom command");
      chemflag = 1;
      memory->create(map,ntypes+1,"compute_sna_atom:map");
      nelements = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      for (int i = 0; i < ntypes; i++) {
        int jelem = utils::inumeric(FLERR,arg[iarg+2+i],false,lmp);
        if (jelem < 0 || jelem >= nelements)
          error->all(FLERR,"Illegal compute sna/atom command");
        map[i+1] = jelem;
      }
      iarg += 2+ntypes;
    } else if (strcmp(arg[iarg],"bnormflag") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute sna/atom command");
      bnormflag = atoi(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"wselfallflag") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute sna/atom command");
      wselfallflag = atoi(arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal compute sna/atom command");
  }

  snaptr = new SNA(lmp, rfac0, twojmax,
                   rmin0, switchflag, bzeroflag,
                   chemflag, bnormflag, wselfallflag, nelements);

  ncoeff = snaptr->ncoeff;
  size_peratom_cols = ncoeff;
  if (quadraticflag) size_peratom_cols += (ncoeff*(ncoeff+1))/2;
  peratom_flag = 1;

  nmax = 0;
  sna = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputeSNAAtom::~ComputeSNAAtom()
{
  memory->destroy(sna);
  memory->destroy(radelem);
  memory->destroy(wjelem);
  memory->destroy(cutsq);
  delete snaptr;
}

/* ---------------------------------------------------------------------- */

void ComputeSNAAtom::init()
{
  if (force->pair == nullptr)
    error->all(FLERR,"Compute sna/atom requires a pair style be defined");

  if (cutmax > force->pair->cutforce)
    error->all(FLERR,"Compute sna/atom cutoff is longer than pairwise cutoff");

  // need an occasional full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"sna/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute sna/atom");
  snaptr->init();
}

/* ---------------------------------------------------------------------- */

void ComputeSNAAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeSNAAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow sna array if necessary

  if (atom->nmax > nmax) {
    memory->destroy(sna);
    nmax = atom->nmax;
    memory->create(sna,nmax,size_peratom_cols,"sna/atom:sna");
    array_atom = sna;
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  const int inum = list->inum;
  const int* const ilist = list->ilist;
  const int* const numneigh = list->numneigh;
  int** const firstneigh = list->firstneigh;
  int * const type = atom->type;

  // compute sna for each atom in group
  // use full neighbor list to count atoms less than cutoff

  double** const x = atom->x;
  const int* const mask = atom->mask;

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (mask[i] & groupbit) {

      const double xtmp = x[i][0];
      const double ytmp = x[i][1];
      const double ztmp = x[i][2];
      const int itype = type[i];
      int ielem = 0;
      if (chemflag)
        ielem = map[itype];
      const double radi = radelem[itype];
      const int* const jlist = firstneigh[i];
      const int jnum = numneigh[i];

      // insure rij, inside, and typej  are of size jnum

      snaptr->grow_rij(jnum);

      // rij[][3] = displacements between atom I and those neighbors
      // inside = indices of neighbors of I within cutoff
      // typej = types of neighbors of I within cutoff

      int ninside = 0;
      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj];
        j &= NEIGHMASK;

        const double delx = xtmp - x[j][0];
        const double dely = ytmp - x[j][1];
        const double delz = ztmp - x[j][2];
        const double rsq = delx*delx + dely*dely + delz*delz;
        int jtype = type[j];
        int jelem = 0;
        if (chemflag)
          jelem = map[jtype];
        if (rsq < cutsq[itype][jtype] && rsq>1e-20) {
          snaptr->rij[ninside][0] = delx;
          snaptr->rij[ninside][1] = dely;
          snaptr->rij[ninside][2] = delz;
          snaptr->inside[ninside] = j;
          snaptr->wj[ninside] = wjelem[jtype];
          snaptr->rcutij[ninside] = (radi+radelem[jtype])*rcutfac;
          snaptr->element[ninside] = jelem; // element index for multi-element snap
          ninside++;
        }
      }

      snaptr->compute_ui(ninside, ielem);
      snaptr->compute_zi();
      snaptr->compute_bi(ielem);
      for (int icoeff = 0; icoeff < ncoeff; icoeff++)
        sna[i][icoeff] = snaptr->blist[icoeff];
      if (quadraticflag) {
        int ncount = ncoeff;
        for (int icoeff = 0; icoeff < ncoeff; icoeff++) {
          double bi = snaptr->blist[icoeff];

          // diagonal element of quadratic matrix

          sna[i][ncount++] = 0.5*bi*bi;

          // upper-triangular elements of quadratic matrix

          for (int jcoeff = icoeff+1; jcoeff < ncoeff; jcoeff++)
            sna[i][ncount++] = bi*snaptr->blist[jcoeff];
        }
      }
    } else {
      for (int icoeff = 0; icoeff < size_peratom_cols; icoeff++)
        sna[i][icoeff] = 0.0;
    }
  }

}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double ComputeSNAAtom::memory_usage()
{
  double bytes = (double)nmax*size_peratom_cols * sizeof(double); // sna
  bytes += snaptr->memory_usage();                        // SNA object

  return bytes;
}

