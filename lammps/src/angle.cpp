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

#include "angle.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "math_const.h"
#include "suffix.h"
#include "atom_masks.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

Angle::Angle(LAMMPS *lmp) : Pointers(lmp)
{
  energy = 0.0;
  virial[0] = virial[1] = virial[2] = virial[3] = virial[4] = virial[5] = 0.0;
  writedata = 1;

  allocated = 0;
  suffix_flag = Suffix::NONE;

  maxeatom = maxvatom = maxcvatom = 0;
  eatom = nullptr;
  vatom = nullptr;
  cvatom = nullptr;
  setflag = nullptr;
  centroidstressflag = CENTROID_AVAIL;

  execution_space = Host;
  datamask_read = ALL_MASK;
  datamask_modify = ALL_MASK;

  copymode = 0;
}

/* ---------------------------------------------------------------------- */

Angle::~Angle()
{
  if (copymode) return;

  memory->destroy(eatom);
  memory->destroy(vatom);
  memory->destroy(cvatom);
}

/* ----------------------------------------------------------------------
   check if all coeffs are set
------------------------------------------------------------------------- */

void Angle::init()
{
  if (!allocated && atom->nangletypes)
    error->all(FLERR,"Angle coeffs are not set");
  for (int i = 1; i <= atom->nangletypes; i++)
    if (setflag[i] == 0) error->all(FLERR,"All angle coeffs are not set");

  init_style();
}

/* ----------------------------------------------------------------------
   setup for energy, virial computation
   see integrate::ev_set() for bitwise settings of eflag/vflag
   set the following flags, values are otherwise set to 0:
     evflag       != 0 if any bits of eflag or vflag are set
     eflag_global != 0 if ENERGY_GLOBAL bit of eflag set
     eflag_atom   != 0 if ENERGY_ATOM bit of eflag set
     eflag_either != 0 if eflag_global or eflag_atom is set
     vflag_global != 0 if VIRIAL_PAIR or VIRIAL_FDOTR bit of vflag set
     vflag_atom   != 0 if VIRIAL_ATOM bit of vflag set
     vflag_atom   != 0 if VIRIAL_CENTROID bit of vflag set
                       and centroidstressflag != CENTROID_AVAIL
     cvflag_atom  != 0 if VIRIAL_CENTROID bit of vflag set
                       and centroidstressflag = CENTROID_AVAIL
     vflag_either != 0 if any of vflag_global, vflag_atom, cvflag_atom is set
------------------------------------------------------------------------- */

void Angle::ev_setup(int eflag, int vflag, int alloc)
{
  int i,n;

  evflag = 1;

  eflag_either = eflag;
  eflag_global = eflag & ENERGY_GLOBAL;
  eflag_atom = eflag & ENERGY_ATOM;

  vflag_global = vflag & (VIRIAL_PAIR | VIRIAL_FDOTR);
  vflag_atom = vflag & VIRIAL_ATOM;
  if (vflag & VIRIAL_CENTROID && centroidstressflag != CENTROID_AVAIL)
    vflag_atom = 1;
  cvflag_atom = 0;
  if (vflag & VIRIAL_CENTROID && centroidstressflag == CENTROID_AVAIL)
    cvflag_atom = 1;
  vflag_either = vflag_global || vflag_atom || cvflag_atom;

  // reallocate per-atom arrays if necessary

  if (eflag_atom && atom->nmax > maxeatom) {
    maxeatom = atom->nmax;
    if (alloc) {
      memory->destroy(eatom);
      memory->create(eatom,comm->nthreads*maxeatom,"angle:eatom");
    }
  }
  if (vflag_atom && atom->nmax > maxvatom) {
    maxvatom = atom->nmax;
    if (alloc) {
      memory->destroy(vatom);
      memory->create(vatom,comm->nthreads*maxvatom,6,"angle:vatom");
    }
  }
  if (cvflag_atom && atom->nmax > maxcvatom) {
    maxcvatom = atom->nmax;
    if (alloc) {
      memory->destroy(cvatom);
      memory->create(cvatom,comm->nthreads*maxcvatom,9,"angle:cvatom");
    }
  }

  // zero accumulators

  if (eflag_global) energy = 0.0;
  if (vflag_global) for (i = 0; i < 6; i++) virial[i] = 0.0;
  if (eflag_atom && alloc) {
    n = atom->nlocal;
    if (force->newton_bond) n += atom->nghost;
    for (i = 0; i < n; i++) eatom[i] = 0.0;
  }
  if (vflag_atom && alloc) {
    n = atom->nlocal;
    if (force->newton_bond) n += atom->nghost;
    for (i = 0; i < n; i++) {
      vatom[i][0] = 0.0;
      vatom[i][1] = 0.0;
      vatom[i][2] = 0.0;
      vatom[i][3] = 0.0;
      vatom[i][4] = 0.0;
      vatom[i][5] = 0.0;
    }
  }
  if (cvflag_atom && alloc) {
    n = atom->nlocal;
    if (force->newton_bond) n += atom->nghost;
    for (i = 0; i < n; i++) {
      cvatom[i][0] = 0.0;
      cvatom[i][1] = 0.0;
      cvatom[i][2] = 0.0;
      cvatom[i][3] = 0.0;
      cvatom[i][4] = 0.0;
      cvatom[i][5] = 0.0;
      cvatom[i][6] = 0.0;
      cvatom[i][7] = 0.0;
      cvatom[i][8] = 0.0;
      cvatom[i][9] = 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
   tally energy and virial into global and per-atom accumulators
   virial = r1F1 + r2F2 + r3F3 = (r1-r2) F1 + (r3-r2) F3 = del1*f1 + del2*f3
------------------------------------------------------------------------- */

void Angle::ev_tally(int i, int j, int k, int nlocal, int newton_bond,
                     double eangle, double *f1, double *f3,
                     double delx1, double dely1, double delz1,
                     double delx2, double dely2, double delz2)
{
  double eanglethird,v[6];

  if (eflag_either) {
    if (eflag_global) {
      if (newton_bond) energy += eangle;
      else {
        eanglethird = THIRD*eangle;
        if (i < nlocal) energy += eanglethird;
        if (j < nlocal) energy += eanglethird;
        if (k < nlocal) energy += eanglethird;
      }
    }
    if (eflag_atom) {
      eanglethird = THIRD*eangle;
      if (newton_bond || i < nlocal) eatom[i] += eanglethird;
      if (newton_bond || j < nlocal) eatom[j] += eanglethird;
      if (newton_bond || k < nlocal) eatom[k] += eanglethird;
    }
  }

  if (vflag_either) {
    v[0] = delx1*f1[0] + delx2*f3[0];
    v[1] = dely1*f1[1] + dely2*f3[1];
    v[2] = delz1*f1[2] + delz2*f3[2];
    v[3] = delx1*f1[1] + delx2*f3[1];
    v[4] = delx1*f1[2] + delx2*f3[2];
    v[5] = dely1*f1[2] + dely2*f3[2];

    if (vflag_global) {
      if (newton_bond) {
        virial[0] += v[0];
        virial[1] += v[1];
        virial[2] += v[2];
        virial[3] += v[3];
        virial[4] += v[4];
        virial[5] += v[5];
      } else {
        if (i < nlocal) {
          virial[0] += THIRD*v[0];
          virial[1] += THIRD*v[1];
          virial[2] += THIRD*v[2];
          virial[3] += THIRD*v[3];
          virial[4] += THIRD*v[4];
          virial[5] += THIRD*v[5];
        }
        if (j < nlocal) {
          virial[0] += THIRD*v[0];
          virial[1] += THIRD*v[1];
          virial[2] += THIRD*v[2];
          virial[3] += THIRD*v[3];
          virial[4] += THIRD*v[4];
          virial[5] += THIRD*v[5];
        }
        if (k < nlocal) {
          virial[0] += THIRD*v[0];
          virial[1] += THIRD*v[1];
          virial[2] += THIRD*v[2];
          virial[3] += THIRD*v[3];
          virial[4] += THIRD*v[4];
          virial[5] += THIRD*v[5];
        }
      }
    }

    if (vflag_atom) {
      if (newton_bond || i < nlocal) {
        vatom[i][0] += THIRD*v[0];
        vatom[i][1] += THIRD*v[1];
        vatom[i][2] += THIRD*v[2];
        vatom[i][3] += THIRD*v[3];
        vatom[i][4] += THIRD*v[4];
        vatom[i][5] += THIRD*v[5];
      }
      if (newton_bond || j < nlocal) {
        vatom[j][0] += THIRD*v[0];
        vatom[j][1] += THIRD*v[1];
        vatom[j][2] += THIRD*v[2];
        vatom[j][3] += THIRD*v[3];
        vatom[j][4] += THIRD*v[4];
        vatom[j][5] += THIRD*v[5];
      }
      if (newton_bond || k < nlocal) {
        vatom[k][0] += THIRD*v[0];
        vatom[k][1] += THIRD*v[1];
        vatom[k][2] += THIRD*v[2];
        vatom[k][3] += THIRD*v[3];
        vatom[k][4] += THIRD*v[4];
        vatom[k][5] += THIRD*v[5];
      }
    }
  }

  // per-atom centroid virial
  if (cvflag_atom) {

    // r0 = (r1+r2+r3)/3
    // rij = ri-rj
    // total virial = r10*f1 + r20*f2 + r30*f3
    // del1: r12
    // del2: r32

    if (newton_bond || i < nlocal) {
      double a1[3];

      // a1 = r10 = (2*r12 -   r32)/3
      a1[0] = THIRD*(2*delx1-delx2);
      a1[1] = THIRD*(2*dely1-dely2);
      a1[2] = THIRD*(2*delz1-delz2);

      cvatom[i][0] += a1[0]*f1[0];
      cvatom[i][1] += a1[1]*f1[1];
      cvatom[i][2] += a1[2]*f1[2];
      cvatom[i][3] += a1[0]*f1[1];
      cvatom[i][4] += a1[0]*f1[2];
      cvatom[i][5] += a1[1]*f1[2];
      cvatom[i][6] += a1[1]*f1[0];
      cvatom[i][7] += a1[2]*f1[0];
      cvatom[i][8] += a1[2]*f1[1];
    }
    if (newton_bond || j < nlocal) {
      double a2[3];
      double f2[3];

      // a2 = r20 = ( -r12 -   r32)/3
      a2[0] = THIRD*(-delx1-delx2);
      a2[1] = THIRD*(-dely1-dely2);
      a2[2] = THIRD*(-delz1-delz2);

      f2[0] = - f1[0] - f3[0];
      f2[1] = - f1[1] - f3[1];
      f2[2] = - f1[2] - f3[2];

      cvatom[j][0] += a2[0]*f2[0];
      cvatom[j][1] += a2[1]*f2[1];
      cvatom[j][2] += a2[2]*f2[2];
      cvatom[j][3] += a2[0]*f2[1];
      cvatom[j][4] += a2[0]*f2[2];
      cvatom[j][5] += a2[1]*f2[2];
      cvatom[j][6] += a2[1]*f2[0];
      cvatom[j][7] += a2[2]*f2[0];
      cvatom[j][8] += a2[2]*f2[1];
    }
    if (newton_bond || k < nlocal) {
      double a3[3];

      // a3 = r30 = ( -r12 + 2*r32)/3
      a3[0] = THIRD*(-delx1+2*delx2);
      a3[1] = THIRD*(-dely1+2*dely2);
      a3[2] = THIRD*(-delz1+2*delz2);

      cvatom[k][0] += a3[0]*f3[0];
      cvatom[k][1] += a3[1]*f3[1];
      cvatom[k][2] += a3[2]*f3[2];
      cvatom[k][3] += a3[0]*f3[1];
      cvatom[k][4] += a3[0]*f3[2];
      cvatom[k][5] += a3[1]*f3[2];
      cvatom[k][6] += a3[1]*f3[0];
      cvatom[k][7] += a3[2]*f3[0];
      cvatom[k][8] += a3[2]*f3[1];
    }
  }
}

/* ---------------------------------------------------------------------- */

double Angle::memory_usage()
{
  double bytes = (double)comm->nthreads*maxeatom * sizeof(double);
  bytes += (double)comm->nthreads*maxvatom*6 * sizeof(double);
  bytes += (double)comm->nthreads*maxcvatom*9 * sizeof(double);
  return bytes;
}
