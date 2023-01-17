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

#include "npair_half_respa_nsq_newton.h"
#include "neigh_list.h"
#include "atom.h"
#include "atom_vec.h"
#include "group.h"
#include "molecule.h"
#include "domain.h"
#include "my_page.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

NPairHalfRespaNsqNewton::NPairHalfRespaNsqNewton(LAMMPS *lmp) : NPair(lmp) {}

/* ----------------------------------------------------------------------
   multiple respa lists
   N^2 / 2 search for neighbor pairs with full Newton's 3rd law
   pair added to list if atoms i and j are both owned and i < j
   if j is ghost only me or other proc adds pair
   decision based on itag,jtag tests
------------------------------------------------------------------------- */

void NPairHalfRespaNsqNewton::build(NeighList *list)
{
  int i,j,n,itype,jtype,itag,jtag,n_inner,n_middle,bitmask;
  int imol,iatom,moltemplate;
  tagint tagprev;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *neighptr,*neighptr_inner,*neighptr_middle;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;
  tagint **special = atom->special;
  int **nspecial = atom->nspecial;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  if (includegroup) {
    nlocal = atom->nfirst;
    bitmask = group->bitmask[includegroup];
  }

  int *molindex = atom->molindex;
  int *molatom = atom->molatom;
  Molecule **onemols = atom->avec->onemols;
  if (molecular == Atom::TEMPLATE) moltemplate = 1;
  else moltemplate = 0;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  MyPage<int> *ipage = list->ipage;

  int *ilist_inner = list->ilist_inner;
  int *numneigh_inner = list->numneigh_inner;
  int **firstneigh_inner = list->firstneigh_inner;
  MyPage<int> *ipage_inner = list->ipage_inner;

  int *ilist_middle,*numneigh_middle,**firstneigh_middle;
  MyPage<int> *ipage_middle;
  int respamiddle = list->respamiddle;
  if (respamiddle) {
    ilist_middle = list->ilist_middle;
    numneigh_middle = list->numneigh_middle;
    firstneigh_middle = list->firstneigh_middle;
    ipage_middle = list->ipage_middle;
  }

  int inum = 0;
  int which = 0;
  int minchange = 0;
  ipage->reset();
  ipage_inner->reset();
  if (respamiddle) ipage_middle->reset();

  for (i = 0; i < nlocal; i++) {
    n = n_inner = 0;
    neighptr = ipage->vget();
    neighptr_inner = ipage_inner->vget();
    if (respamiddle) {
      n_middle = 0;
      neighptr_middle = ipage_middle->vget();
    }

    itag = tag[i];
    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    if (moltemplate) {
      imol = molindex[i];
      iatom = molatom[i];
      tagprev = tag[i] - iatom - 1;
    }

    // loop over remaining atoms, owned and ghost

    for (j = i+1; j < nall; j++) {
      if (includegroup && !(mask[j] & bitmask)) continue;

      if (j >= nlocal) {
        jtag = tag[j];
        if (itag > jtag) {
          if ((itag+jtag) % 2 == 0) continue;
        } else if (itag < jtag) {
          if ((itag+jtag) % 2 == 1) continue;
        } else {
          if (x[j][2] < ztmp) continue;
          if (x[j][2] == ztmp) {
            if (x[j][1] < ytmp) continue;
            if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
          }
        }
      }

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
          else if ((minchange = domain->minimum_image_check(delx,dely,delz)))
            neighptr[n++] = j;
          else if (which > 0) neighptr[n++] = j ^ (which << SBBITS);
        } else neighptr[n++] = j;

        if (rsq < cut_inner_sq) {
          if (which == 0) neighptr_inner[n_inner++] = j;
          else if (minchange) neighptr_inner[n_inner++] = j;
          else if (which > 0) neighptr_inner[n_inner++] = j ^ (which << SBBITS);
        }

        if (respamiddle &&
            rsq < cut_middle_sq && rsq > cut_middle_inside_sq) {
          if (which == 0) neighptr_middle[n_middle++] = j;
          else if (minchange) neighptr_middle[n_middle++] = j;
          else if (which > 0)
            neighptr_middle[n_middle++] = j ^ (which << SBBITS);
        }
      }
    }

    ilist[inum] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage->vgot(n);
    if (ipage->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");

    ilist_inner[inum] = i;
    firstneigh_inner[i] = neighptr_inner;
    numneigh_inner[i] = n_inner;
    ipage_inner->vgot(n_inner);
    if (ipage_inner->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");

    if (respamiddle) {
      ilist_middle[inum] = i;
      firstneigh_middle[i] = neighptr_middle;
      numneigh_middle[i] = n_middle;
      ipage_middle->vgot(n_middle);
      if (ipage_middle->status())
        error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
    }

    inum++;
  }

  list->inum = inum;
  list->inum_inner = inum;
  if (respamiddle) list->inum_middle = inum;
}
