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

#include "npair_full_bin_ghost.h"
#include "neigh_list.h"
#include "atom.h"
#include "atom_vec.h"
#include "molecule.h"
#include "domain.h"
#include "my_page.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

NPairFullBinGhost::NPairFullBinGhost(LAMMPS *lmp) : NPair(lmp) {}

/* ----------------------------------------------------------------------
   binned neighbor list construction for all neighbors
   include neighbors of ghost atoms, but no "special neighbors" for ghosts
   every neighbor pair appears in list of both atoms i and j
------------------------------------------------------------------------- */

void NPairFullBinGhost::build(NeighList *list)
{
  int i,j,k,n,itype,jtype,ibin,which,imol,iatom,moltemplate;
  tagint tagprev;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int xbin,ybin,zbin,xbin2,ybin2,zbin2;
  int *neighptr;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;
  tagint **special = atom->special;
  int **nspecial = atom->nspecial;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;

  int *molindex = atom->molindex;
  int *molatom = atom->molatom;
  Molecule **onemols = atom->avec->onemols;
  if (molecular == Atom::TEMPLATE) moltemplate = 1;
  else moltemplate = 0;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  MyPage<int> *ipage = list->ipage;

  int inum = 0;
  ipage->reset();

  // loop over owned & ghost atoms, storing neighbors

  for (i = 0; i < nall; i++) {
    n = 0;
    neighptr = ipage->vget();

    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    if (moltemplate) {
      imol = molindex[i];
      iatom = molatom[i];
      tagprev = tag[i] - iatom - 1;
    }

    // loop over all atoms in surrounding bins in stencil including self
    // when i is a ghost atom, must check if stencil bin is out of bounds
    // skip i = j
    // no molecular test when i = ghost atom

    if (i < nlocal) {
      ibin = atom2bin[i];
      for (k = 0; k < nstencil; k++) {
        for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
          if (i == j) continue;

          jtype = type[j];
          if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx*delx + dely*dely + delz*delz;

          if (rsq <= cutneighsq[itype][jtype]) {
            if (molecular != Atom::ATOMIC) {
              if (!moltemplate)
                which = find_special(special[i],nspecial[i],tag[j]);
              else if (imol >= 0)
                which = find_special(onemols[imol]->special[iatom],
                                     onemols[imol]->nspecial[iatom],
                                     tag[j]-tagprev);
              else which = 0;
              if (which == 0) neighptr[n++] = j;
              else if (domain->minimum_image_check(delx,dely,delz))
                neighptr[n++] = j;
              else if (which > 0) neighptr[n++] = j ^ (which << SBBITS);
            } else neighptr[n++] = j;
          }
        }
      }

    } else {
      ibin = coord2bin(x[i],xbin,ybin,zbin);
      for (k = 0; k < nstencil; k++) {
        xbin2 = xbin + stencilxyz[k][0];
        ybin2 = ybin + stencilxyz[k][1];
        zbin2 = zbin + stencilxyz[k][2];
        if (xbin2 < 0 || xbin2 >= mbinx ||
            ybin2 < 0 || ybin2 >= mbiny ||
            zbin2 < 0 || zbin2 >= mbinz) continue;
        for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
          if (i == j) continue;

          jtype = type[j];
          if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx*delx + dely*dely + delz*delz;

          if (rsq <= cutneighghostsq[itype][jtype]) neighptr[n++] = j;
        }
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage->vgot(n);
    if (ipage->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }

  list->inum = atom->nlocal;
  list->gnum = inum - atom->nlocal;
}
