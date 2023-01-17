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
   Contributing author: Mingjian Wen (University of Minnesota)
   e-mail: wenxx151@umn.edu, wenxx151@gmail.com

   This implements the DRIP model as described in
   M. Wen, S. Carr, S. Fang, E. Kaxiras, and E. B. Tadmor,
   Phys. Rev. B, 98, 235404 (2018).
------------------------------------------------------------------------- */

#include "pair_drip.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

#define MAXLINE 1024
#define DELTA 4
#define HALF 0.5

// inline functions
static inline double dot(double const *x, double const *y)
{
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

static inline void mat_dot_vec(PairDRIP::V3 const *X, double const *y, double *const z)
{
  for (int k = 0; k < 3; k++) { z[k] = X[k][0] * y[0] + X[k][1] * y[1] + X[k][2] * y[2]; }
}

/* ---------------------------------------------------------------------- */

PairDRIP::PairDRIP(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);

  params = nullptr;
  nearest3neigh = nullptr;
  cutmax = 0.0;
}

/* ---------------------------------------------------------------------- */

PairDRIP::~PairDRIP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }

  memory->destroy(params);
  memory->destroy(elem2param);
  memory->destroy(nearest3neigh);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairDRIP::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "Pair style drip requires newton pair on");
  if (!atom->molecule_flag) error->all(FLERR, "Pair style drip requires atom attribute molecule");

  // need a full neighbor list, including neighbors of ghosts
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->ghost = 1;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairDRIP::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  memory->create(cutsq, n, n, "pair:cutsq");
  map = new int[n];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairDRIP::settings(int narg, char ** /* arg */)
{
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");
  if (!utils::strmatch(force->pair_style, "^hybrid/overlay"))
    error->all(FLERR, "Pair style drip must be used as sub-style with hybrid/overlay");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDRIP::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  map_element2type(narg - 3, arg + 3);
  read_file(arg[2]);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairDRIP::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  int itype = map[i];
  int jtype = map[j];
  int iparam_ij = elem2param[itype][jtype];
  Param &p = params[iparam_ij];

  // max cutoff is the main cutoff plus the normal cutoff such that
  return p.rcut + p.ncut;
}

/* ----------------------------------------------------------------------
   read DRIP file
------------------------------------------------------------------------- */

void PairDRIP::read_file(char *filename)
{
  memory->sfree(params);
  params = nullptr;
  nparams = maxparam = 0;

  // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, filename, "drip", unit_convert_flag);
    char *line;

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    double conversion_factor = utils::get_conversion_factor(utils::ENERGY, unit_convert);

    while ((line = reader.next_line(NPARAMS_PER_LINE))) {

      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();
        std::string jname = values.next_string();

        // ielement,jelement = 1st args
        // if both args are in element list, then parse this line
        // else skip to next entry in file
        int ielement, jelement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements) continue;
        for (jelement = 0; jelement < nelements; jelement++)
          if (jname == elements[jelement]) break;
        if (jelement == nelements) continue;

        // expand storage, if needed

        if (nparams == maxparam) {
          maxparam += DELTA;
          params = (Param *) memory->srealloc(params, maxparam * sizeof(Param), "pair:params");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(params + nparams, 0, DELTA * sizeof(Param));
        }

        params[nparams].ielement = ielement;
        params[nparams].jelement = jelement;
        params[nparams].C0 = values.next_double();
        params[nparams].C2 = values.next_double();
        params[nparams].C4 = values.next_double();
        params[nparams].C = values.next_double();
        params[nparams].delta = values.next_double();
        params[nparams].lambda = values.next_double();
        params[nparams].A = values.next_double();
        params[nparams].z0 = values.next_double();
        params[nparams].B = values.next_double();
        params[nparams].eta = values.next_double();
        params[nparams].rhocut = values.next_double();
        params[nparams].rcut = values.next_double();
        params[nparams].ncut = values.next_double();

      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      if (unit_convert) {
        params[nparams].C0 *= conversion_factor;
        params[nparams].C2 *= conversion_factor;
        params[nparams].C4 *= conversion_factor;
        params[nparams].C *= conversion_factor;
        params[nparams].A *= conversion_factor;
        params[nparams].B *= conversion_factor;
      }

      // convenient precomputations
      params[nparams].rhocutsq = params[nparams].rhocut * params[nparams].rhocut;
      params[nparams].rcutsq = params[nparams].rcut * params[nparams].rcut;
      params[nparams].ncutsq = params[nparams].ncut * params[nparams].ncut;

      nparams++;
    }
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    params = (Param *) memory->srealloc(params, maxparam * sizeof(Param), "pair:params");
  }

  MPI_Bcast(params, maxparam * sizeof(Param), MPI_BYTE, 0, world);

  memory->destroy(elem2param);
  memory->create(elem2param, nelements, nelements, "pair:elem2param");
  for (int i = 0; i < nelements; i++) {
    for (int j = 0; j < nelements; j++) {
      int n = -1;
      for (int m = 0; m < nparams; m++) {
        if (i == params[m].ielement && j == params[m].jelement) {
          if (n >= 0) error->all(FLERR, "DRIP potential file has duplicate entry");
          n = m;
        }
      }
      if (n < 0) error->all(FLERR, "Potential file is missing an entry");
      elem2param[i][j] = n;
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairDRIP::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double ni[3];
  double dni_dri[3][3], dni_drnb1[3][3];
  double dni_drnb2[3][3], dni_drnb3[3][3];

  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  find_nearest3neigh();

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (nearest3neigh[i][0] == -1) { continue; }
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = map[type[i]];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // normal and its derivatives w.r.t. atom i and its 3 nearest neighbors
    calc_normal(i, ni, dni_dri, dni_drnb1, dni_drnb2, dni_drnb3);

    double fi[3] = {0., 0., 0.};

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      if (nearest3neigh[j][0] == -1) { continue; }
      jtype = map[type[j]];

      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;
      int iparam_ij = elem2param[itype][jtype];
      Param &p = params[iparam_ij];
      double rcutsq = p.rcutsq;

      // only include the interaction between different layers
      if (rsq < rcutsq && atom->molecule[i] != atom->molecule[j]) {

        double fj[3] = {0., 0., 0.};
        double rvec[3] = {delx, dely, delz};

        double phi_attr = calc_attractive(p, rsq, rvec, fi, fj);

        double phi_repul = calc_repulsive(i, j, p, rsq, rvec, ni, dni_dri, dni_drnb1, dni_drnb2,
                                          dni_drnb3, fi, fj);

        if (eflag)
          evdwl = HALF * (phi_repul + phi_attr);
        else
          evdwl = 0.0;
        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, 0, 0, 0, 0);

        f[j][0] += fj[0];
        f[j][1] += fj[1];
        f[j][2] += fj[2];
        if (vflag_either) v_tally2_newton(j, fj, x[j]);
      }
    }    //loop over jj

    f[i][0] += fi[0];
    f[i][1] += fi[1];
    f[i][2] += fi[2];
    if (vflag_either) v_tally2_newton(i, fi, x[i]);

  }    // loop over ii

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   Attractive part, i.e. the r^(-6) part
------------------------------------------------------------------------- */

double PairDRIP::calc_attractive(Param &p, double const rsq, double const *rvec, double *const fi,
                                 double *const fj)
{
  double const z0 = p.z0;
  double const A = p.A;
  double const cutoff = p.rcut;
  double const r = sqrt(rsq);

  double roz0_sq = rsq / (z0 * z0);
  double dtp;
  double tp = tap(r, cutoff, dtp);
  double r6 = A / (roz0_sq * roz0_sq * roz0_sq);
  double dr6 = -6 * r6 / r;
  double phi = -r6 * tp;

  double fpair = -HALF * (r6 * dtp + dr6 * tp);

  fi[0] += rvec[0] * fpair / r;
  fi[1] += rvec[1] * fpair / r;
  fi[2] += rvec[2] * fpair / r;
  fj[0] -= rvec[0] * fpair / r;
  fj[1] -= rvec[1] * fpair / r;
  fj[2] -= rvec[2] * fpair / r;

  return phi;
}

/* ----------------------------------------------------------------------
   Repulsive part that depends on transverse distance and dihedral angle
------------------------------------------------------------------------- */

double PairDRIP::calc_repulsive(int const i, int const j, Param &p, double const rsq,
                                double const *rvec, double const *ni, V3 const *dni_dri,
                                V3 const *dni_drnb1, V3 const *dni_drnb2, V3 const *dni_drnb3,
                                double *const fi, double *const fj)
{
  double **f = atom->f;
  double **x = atom->x;

  double C0 = p.C0;
  double C2 = p.C2;
  double C4 = p.C4;
  double C = p.C;
  double delta = p.delta;
  double lambda = p.lambda;
  double z0 = p.z0;
  double cutoff = p.rcut;

  // nearest 3 neighbors of atoms i and j
  int nbi1 = nearest3neigh[i][0];
  int nbi2 = nearest3neigh[i][1];
  int nbi3 = nearest3neigh[i][2];
  int nbj1 = nearest3neigh[j][0];
  int nbj2 = nearest3neigh[j][1];
  int nbj3 = nearest3neigh[j][2];

  double fnbi1[3];
  double fnbi2[3];
  double fnbi3[3];
  double fnbj1[3];
  double fnbj2[3];
  double fnbj3[3];
  V3 dgij_dri;
  V3 dgij_drj;
  V3 dgij_drk1;
  V3 dgij_drk2;
  V3 dgij_drk3;
  V3 dgij_drl1;
  V3 dgij_drl2;
  V3 dgij_drl3;
  V3 drhosqij_dri;
  V3 drhosqij_drj;
  V3 drhosqij_drnb1;
  V3 drhosqij_drnb2;
  V3 drhosqij_drnb3;

  double r = sqrt(rsq);

  // derivative of rhosq w.r.t. atoms i j and the nearests 3 neighs of i
  get_drhosqij(rvec, ni, dni_dri, dni_drnb1, dni_drnb2, dni_drnb3, drhosqij_dri, drhosqij_drj,
               drhosqij_drnb1, drhosqij_drnb2, drhosqij_drnb3);

  // transverse decay function f(rho) and its derivative w.r.t. rhosq
  double rhosqij;
  double dtdij;
  double tdij = td(C0, C2, C4, delta, rvec, r, ni, rhosqij, dtdij);

  // dihedral angle function and its derivateives
  double dgij_drhosq;
  double gij = dihedral(i, j, p, rhosqij, dgij_drhosq, dgij_dri, dgij_drj, dgij_drk1, dgij_drk2,
                        dgij_drk3, dgij_drl1, dgij_drl2, dgij_drl3);

  double V2 = C + tdij + gij;

  // tap part
  double dtp;
  double tp = tap(r, cutoff, dtp);

  // exponential part
  double V1 = exp(-lambda * (r - z0));
  double dV1 = -V1 * lambda;

  // total energy
  double phi = tp * V1 * V2;

  for (int k = 0; k < 3; k++) {
    // forces due to derivatives of tap and V1
    double tmp = HALF * (dtp * V1 + tp * dV1) * V2 * rvec[k] / r;
    fi[k] += tmp;
    fj[k] -= tmp;

    // contributions from transverse decay part tdij and the dihedral part gij

    // derivative of V2 contribute to atoms i, j
    fi[k] -= HALF * tp * V1 * ((dtdij + dgij_drhosq) * drhosqij_dri[k] + dgij_dri[k]);
    fj[k] -= HALF * tp * V1 * ((dtdij + dgij_drhosq) * drhosqij_drj[k] + dgij_drj[k]);
    // derivative of V2 contribute to nearest 3 neighs of atom i
    fnbi1[k] = -HALF * tp * V1 * ((dtdij + dgij_drhosq) * drhosqij_drnb1[k] + dgij_drk1[k]);
    fnbi2[k] = -HALF * tp * V1 * ((dtdij + dgij_drhosq) * drhosqij_drnb2[k] + dgij_drk2[k]);
    fnbi3[k] = -HALF * tp * V1 * ((dtdij + dgij_drhosq) * drhosqij_drnb3[k] + dgij_drk3[k]);
    // derivative of V2 contribute to nearest 3 neighs of atom j
    fnbj1[k] = -HALF * tp * V1 * dgij_drl1[k];
    fnbj2[k] = -HALF * tp * V1 * dgij_drl2[k];
    fnbj3[k] = -HALF * tp * V1 * dgij_drl3[k];
  }

  for (int k = 0; k < 3; k++) {
    f[nbi1][k] += fnbi1[k];
    f[nbi2][k] += fnbi2[k];
    f[nbi3][k] += fnbi3[k];
    f[nbj1][k] += fnbj1[k];
    f[nbj2][k] += fnbj2[k];
    f[nbj3][k] += fnbj3[k];
  }

  if (vflag_either) {
    v_tally2_newton(nbi1, fnbi1, x[nbi1]);
    v_tally2_newton(nbi2, fnbi2, x[nbi2]);
    v_tally2_newton(nbi3, fnbi3, x[nbi3]);
    v_tally2_newton(nbj1, fnbj1, x[nbj1]);
    v_tally2_newton(nbj2, fnbj2, x[nbj2]);
    v_tally2_newton(nbj3, fnbj3, x[nbj3]);
  }

  return phi;
}

/* ---------------------------------------------------------------------- */

void PairDRIP::find_nearest3neigh()
{
  int i, j, ii, jj, allnum, inum, jnum, itype, jtype, size;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  int *type = atom->type;

  allnum = list->inum + list->gnum;
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  size = allnum;
  memory->destroy(nearest3neigh);
  memory->create(nearest3neigh, size, 3, "pair:nearest3neigh");

  for (ii = 0; ii < allnum; ii++) {
    i = ilist[ii];

    // If "NULL" used in pair_coeff, i could be larger than allnum
    if (i >= size) {
      size = i + 1;
      memory->grow(nearest3neigh, size, 3, "pair:nearest3neigh");
    }

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = map[type[i]];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // init nb1 to be the 1st nearest neigh, nb3 the 3rd nearest
    int nb1 = -1;
    int nb2 = -1;
    int nb3 = -1;
    double nb1_rsq = 1.0e10 + 1;
    double nb2_rsq = 2.0e10;
    double nb3_rsq = 3.0e10;

    for (jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = map[type[j]];
      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;

      int iparam_ij = elem2param[itype][jtype];
      double ncutsq = params[iparam_ij].ncutsq;

      if (rsq < ncutsq && atom->molecule[i] == atom->molecule[j]) {
        // find the 3 nearest neigh
        if (rsq < nb1_rsq) {
          nb3 = nb2;
          nb2 = nb1;
          nb1 = j;
          nb3_rsq = nb2_rsq;
          nb2_rsq = nb1_rsq;
          nb1_rsq = rsq;
        } else if (rsq < nb2_rsq) {
          nb3 = nb2;
          nb2 = j;
          nb3_rsq = nb2_rsq;
          nb2_rsq = rsq;
        } else if (rsq < nb3_rsq) {
          nb3 = j;
          nb3_rsq = rsq;
        }
      }
    }    // loop over jj

    // store neighbors to be used later to compute normal
    if (nb3_rsq >= 1.0e10) {
      if (i < inum) {
        error->one(FLERR,
                   "No enough neighbors to construct normal. Check the "
                   "configuration to see whether atoms fly away.");
      } else {
        // This only happens for ghost atoms that are near the boundary of the
        // domain (i.e. r > r_cut + n_cut). These ghost atoms will not be
        // the i j atoms in the compute function, but only neighbors of j atoms.
        // It is allowed not to have three neighbors for these atoms, since
        // their normals are not needed.
        nearest3neigh[i][0] = -1;
        nearest3neigh[i][1] = -1;
        nearest3neigh[i][2] = -1;
      }
    } else {
      nearest3neigh[i][0] = nb1;
      nearest3neigh[i][1] = nb2;
      nearest3neigh[i][2] = nb3;
    }
  }    // loop over ii
}

/* ---------------------------------------------------------------------- */

void PairDRIP::calc_normal(int const i, double *const normal, V3 *const dn_dri, V3 *const dn_drk1,
                           V3 *const dn_drk2, V3 *const dn_drk3)
{
  int k1 = nearest3neigh[i][0];
  int k2 = nearest3neigh[i][1];
  int k3 = nearest3neigh[i][2];

  // normal does not depend on i, setting to zero
  for (int j = 0; j < 3; j++) {
    for (int k = 0; k < 3; k++) { dn_dri[j][k] = 0.0; }
  }

  // get normal and derives of normal w.r.t to its 3 nearest neighbors
  double **x = atom->x;
  deriv_cross(x[k1], x[k2], x[k3], normal, dn_drk1, dn_drk2, dn_drk3);
}

/* ---------------------------------------------------------------------- */

void PairDRIP::get_drhosqij(double const *rij, double const *ni, V3 const *dni_dri,
                            V3 const *dni_drn1, V3 const *dni_drn2, V3 const *dni_drn3,
                            double *const drhosq_dri, double *const drhosq_drj,
                            double *const drhosq_drn1, double *const drhosq_drn2,
                            double *const drhosq_drn3)
{
  int k;
  double ni_dot_rij = 0;
  double dni_dri_dot_rij[3];
  double dni_drn1_dot_rij[3];
  double dni_drn2_dot_rij[3];
  double dni_drn3_dot_rij[3];

  ni_dot_rij = dot(ni, rij);
  mat_dot_vec(dni_dri, rij, dni_dri_dot_rij);
  mat_dot_vec(dni_drn1, rij, dni_drn1_dot_rij);
  mat_dot_vec(dni_drn2, rij, dni_drn2_dot_rij);
  mat_dot_vec(dni_drn3, rij, dni_drn3_dot_rij);

  for (k = 0; k < 3; k++) {
    drhosq_dri[k] = -2 * rij[k] - 2 * ni_dot_rij * (-ni[k] + dni_dri_dot_rij[k]);
    drhosq_drj[k] = 2 * rij[k] - 2 * ni_dot_rij * ni[k];
    drhosq_drn1[k] = -2 * ni_dot_rij * dni_drn1_dot_rij[k];
    drhosq_drn2[k] = -2 * ni_dot_rij * dni_drn2_dot_rij[k];
    drhosq_drn3[k] = -2 * ni_dot_rij * dni_drn3_dot_rij[k];
  }
}

/* ----------------------------------------------------------------------
   derivartive of transverse decay function f(rho) w.r.t. rho
------------------------------------------------------------------------- */

double PairDRIP::td(double C0, double C2, double C4, double delta, double const *const rvec,
                    double r, const double *const n, double &rho_sq, double &dtd)
{
  double n_dot_r = dot(n, rvec);

  rho_sq = r * r - n_dot_r * n_dot_r;

  // in case n is [0, 0, 1] and rho_sq is negative due to numerical error
  if (rho_sq < 0) { rho_sq = 0; }

  double del_sq = delta * delta;
  double rod_sq = rho_sq / del_sq;
  double td = exp(-rod_sq) * (C0 + rod_sq * (C2 + rod_sq * C4));
  dtd = -td / del_sq + exp(-rod_sq) * (C2 + 2 * C4 * rod_sq) / del_sq;

  return td;
}

/* ----------------------------------------------------------------------
   derivartive of dihedral angle func gij w.r.t rho, and atom positions
------------------------------------------------------------------------- */

double PairDRIP::dihedral(const int i, const int j, Param &p, double const rhosq, double &d_drhosq,
                          double *const d_dri, double *const d_drj, double *const d_drk1,
                          double *const d_drk2, double *const d_drk3, double *const d_drl1,
                          double *const d_drl2, double *const d_drl3)
{
  double **x = atom->x;

  // get parameter
  double B = p.B;
  double eta = p.eta;
  double cut_rhosq = p.rhocutsq;

  // local vars
  double cos_kl[3][3];           // cos_omega_k1ijl1, cos_omega_k1ijl2 ...
  double d_dcos_kl[3][3];        // deriv of dihedral w.r.t to cos_omega_kijl
  double dcos_kl[3][3][4][3];    // 4 indicates k, i, j, l. e.g. dcoskl[0][1][0]
                                 // means dcos_omega_k1ijl2 / drk

  // if larger than cutoff of rho, return 0
  if (rhosq >= cut_rhosq) {
    d_drhosq = 0;
    for (int dim = 0; dim < 3; dim++) {
      d_dri[dim] = 0;
      d_drj[dim] = 0;
      d_drk1[dim] = 0;
      d_drk2[dim] = 0;
      d_drk3[dim] = 0;
      d_drl1[dim] = 0;
      d_drl2[dim] = 0;
      d_drl3[dim] = 0;
    }
    double dihe = 0.0;
    return dihe;
  }
  // 3 neighs of atoms i and j
  int k[3];
  int l[3];
  for (int m = 0; m < 3; m++) {
    k[m] = nearest3neigh[i][m];
    l[m] = nearest3neigh[j][m];
  }

  // cos_omega_kijl and the derivatives w.r.t coordinates
  for (int m = 0; m < 3; m++) {
    for (int n = 0; n < 3; n++) {
      cos_kl[m][n] = deriv_cos_omega(x[k[m]], x[i], x[j], x[l[n]], dcos_kl[m][n][0],
                                     dcos_kl[m][n][1], dcos_kl[m][n][2], dcos_kl[m][n][3]);
    }
  }

  double epart1 = exp(-eta * cos_kl[0][0] * cos_kl[0][1] * cos_kl[0][2]);
  double epart2 = exp(-eta * cos_kl[1][0] * cos_kl[1][1] * cos_kl[1][2]);
  double epart3 = exp(-eta * cos_kl[2][0] * cos_kl[2][1] * cos_kl[2][2]);
  double D2 = epart1 + epart2 + epart3;

  // cutoff function
  double d_drhosq_tap;
  double D0 = B * tap_rho(rhosq, cut_rhosq, d_drhosq_tap);

  // dihedral energy
  double dihe = D0 * D2;

  // deriv of dihedral w.r.t rhosq
  d_drhosq = B * d_drhosq_tap * D2;

  // deriv of dihedral w.r.t cos_omega_kijl
  d_dcos_kl[0][0] = -D0 * epart1 * eta * cos_kl[0][1] * cos_kl[0][2];
  d_dcos_kl[0][1] = -D0 * epart1 * eta * cos_kl[0][0] * cos_kl[0][2];
  d_dcos_kl[0][2] = -D0 * epart1 * eta * cos_kl[0][0] * cos_kl[0][1];
  d_dcos_kl[1][0] = -D0 * epart2 * eta * cos_kl[1][1] * cos_kl[1][2];
  d_dcos_kl[1][1] = -D0 * epart2 * eta * cos_kl[1][0] * cos_kl[1][2];
  d_dcos_kl[1][2] = -D0 * epart2 * eta * cos_kl[1][0] * cos_kl[1][1];
  d_dcos_kl[2][0] = -D0 * epart3 * eta * cos_kl[2][1] * cos_kl[2][2];
  d_dcos_kl[2][1] = -D0 * epart3 * eta * cos_kl[2][0] * cos_kl[2][2];
  d_dcos_kl[2][2] = -D0 * epart3 * eta * cos_kl[2][0] * cos_kl[2][1];

  // initialization to be zero and later add values
  for (int dim = 0; dim < 3; dim++) {
    d_drk1[dim] = 0.;
    d_drk2[dim] = 0.;
    d_drk3[dim] = 0.;
    d_dri[dim] = 0.;
    d_drj[dim] = 0.;
    d_drl1[dim] = 0.;
    d_drl2[dim] = 0.;
    d_drl3[dim] = 0.;
  }

  for (int m = 0; m < 3; m++) {
    for (int dim = 0; dim < 3; dim++) {
      d_drk1[dim] += d_dcos_kl[0][m] * dcos_kl[0][m][0][dim];
      d_drk2[dim] += d_dcos_kl[1][m] * dcos_kl[1][m][0][dim];
      d_drk3[dim] += d_dcos_kl[2][m] * dcos_kl[2][m][0][dim];
      d_drl1[dim] += d_dcos_kl[m][0] * dcos_kl[m][0][3][dim];
      d_drl2[dim] += d_dcos_kl[m][1] * dcos_kl[m][1][3][dim];
      d_drl3[dim] += d_dcos_kl[m][2] * dcos_kl[m][2][3][dim];
    }
    for (int n = 0; n < 3; n++) {
      for (int dim = 0; dim < 3; dim++) {
        d_dri[dim] += d_dcos_kl[m][n] * dcos_kl[m][n][1][dim];
        d_drj[dim] += d_dcos_kl[m][n] * dcos_kl[m][n][2][dim];
      }
    }
  }

  return dihe;
}

/* ----------------------------------------------------------------------
   compute cos(omega_kijl) and the derivateives
------------------------------------------------------------------------- */

double PairDRIP::deriv_cos_omega(double const *rk, double const *ri, double const *rj,
                                 double const *rl, double *const dcos_drk, double *const dcos_dri,
                                 double *const dcos_drj, double *const dcos_drl)
{
  double ejik[3];
  double eijl[3];
  double tmp1[3];
  double tmp2[3];
  double dejik_dri[3][3];
  double dejik_drj[3][3];
  double dejik_drk[3][3];
  double deijl_dri[3][3];
  double deijl_drj[3][3];
  double deijl_drl[3][3];

  // ejik and derivatives
  // Note the returned dejik_dri ... are actually the transpose
  deriv_cross(ri, rj, rk, ejik, dejik_dri, dejik_drj, dejik_drk);

  // flip sign
  // deriv_cross computes rij cross rik, here we need rji cross rik
  for (int m = 0; m < 3; m++) {
    ejik[m] = -ejik[m];
    for (int n = 0; n < 3; n++) {
      dejik_dri[m][n] = -dejik_dri[m][n];
      dejik_drj[m][n] = -dejik_drj[m][n];
      dejik_drk[m][n] = -dejik_drk[m][n];
    }
  }

  // eijl and derivatives
  deriv_cross(rj, ri, rl, eijl, deijl_drj, deijl_dri, deijl_drl);
  // flip sign
  for (int m = 0; m < 3; m++) {
    eijl[m] = -eijl[m];
    for (int n = 0; n < 3; n++) {
      deijl_drj[m][n] = -deijl_drj[m][n];
      deijl_dri[m][n] = -deijl_dri[m][n];
      deijl_drl[m][n] = -deijl_drl[m][n];
    }
  }

  // dcos_drk
  mat_dot_vec(dejik_drk, eijl, dcos_drk);
  // dcos_dri
  mat_dot_vec(dejik_dri, eijl, tmp1);
  mat_dot_vec(deijl_dri, ejik, tmp2);
  for (int m = 0; m < 3; m++) { dcos_dri[m] = tmp1[m] + tmp2[m]; }
  // dcos_drj
  mat_dot_vec(dejik_drj, eijl, tmp1);
  mat_dot_vec(deijl_drj, ejik, tmp2);
  for (int m = 0; m < 3; m++) { dcos_drj[m] = tmp1[m] + tmp2[m]; }
  // dcos drl
  mat_dot_vec(deijl_drl, ejik, dcos_drl);

  // cos_oemga_kijl
  double cos_omega = dot(ejik, eijl);

  return cos_omega;
}

/* ---------------------------------------------------------------------- */

double PairDRIP::tap(double r, double cutoff, double &dtap)
{
  double t;
  double r_min = 0;

  if (r <= r_min) {
    t = 1;
    dtap = 0;
  } else {
    double roc = (r - r_min) / (cutoff - r_min);
    double roc_sq = roc * roc;
    t = roc_sq * roc_sq * (-35.0 + 84.0 * roc + roc_sq * (-70.0 + 20.0 * roc)) + 1;
    dtap =
        roc_sq * roc / (cutoff - r_min) * (-140.0 + 420.0 * roc + roc_sq * (-420.0 + 140.0 * roc));
  }

  return t;
}

/* ---------------------------------------------------------------------- */

double PairDRIP::tap_rho(double rhosq, double cut_rhosq, double &drhosq)
{
  double roc_sq;
  double roc;
  double t;

  roc_sq = rhosq / cut_rhosq;
  roc = sqrt(roc_sq);
  t = roc_sq * roc_sq * (-35.0 + 84.0 * roc + roc_sq * (-70.0 + 20.0 * roc)) + 1;

  // Note this dtap/drho_sq not dtap/drho
  drhosq = roc_sq / cut_rhosq * (-70.0 + 210.0 * roc + roc_sq * (-210.0 + 70.0 * roc));

  return t;
}

/* ----------------------------------------------------------------------
   Compute the normalized cross product of two vector rkl, rkm, and the
   derivates w.r.t rk, rl, rm.
   Note, the returned dcross_drk, dcross_drl, and dcross_drm are actually the
   transpose.
------------------------------------------------------------------------- */

void PairDRIP::deriv_cross(double const *rk, double const *rl, double const *rm,
                           double *const cross, V3 *const dcross_drk, V3 *const dcross_drl,
                           V3 *const dcross_drm)
{
  double x[3];
  double y[3];
  double p[3];
  double q;
  double q_cubic;
  double d_invq_d_x0;
  double d_invq_d_x1;
  double d_invq_d_x2;
  double d_invq_d_y0;
  double d_invq_d_y1;
  double d_invq_d_y2;

  int i, j;

  // get x = rkl and y = rkm
  for (i = 0; i < 3; i++) {
    x[i] = rl[i] - rk[i];
    y[i] = rm[i] - rk[i];
  }

  // cross product
  p[0] = x[1] * y[2] - x[2] * y[1];
  p[1] = x[2] * y[0] - x[0] * y[2];
  p[2] = x[0] * y[1] - x[1] * y[0];

  q = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);

  // normalized cross
  cross[0] = p[0] / q;
  cross[1] = p[1] / q;
  cross[2] = p[2] / q;

  // compute derivatives
  // derivative of inverse q (i.e. 1/q) w.r.t x and y
  q_cubic = q * q * q;
  d_invq_d_x0 = (+p[1] * y[2] - p[2] * y[1]) / q_cubic;
  d_invq_d_x1 = (-p[0] * y[2] + p[2] * y[0]) / q_cubic;
  d_invq_d_x2 = (p[0] * y[1] - p[1] * y[0]) / q_cubic;
  d_invq_d_y0 = (-p[1] * x[2] + p[2] * x[1]) / q_cubic;
  d_invq_d_y1 = (p[0] * x[2] - p[2] * x[0]) / q_cubic;
  d_invq_d_y2 = (-p[0] * x[1] + p[1] * x[0]) / q_cubic;

  // dcross/drl transposed
  dcross_drl[0][0] = p[0] * d_invq_d_x0;
  dcross_drl[0][1] = -y[2] / q + p[1] * d_invq_d_x0;
  dcross_drl[0][2] = y[1] / q + p[2] * d_invq_d_x0;

  dcross_drl[1][0] = y[2] / q + p[0] * d_invq_d_x1;
  dcross_drl[1][1] = p[1] * d_invq_d_x1;
  dcross_drl[1][2] = -y[0] / q + p[2] * d_invq_d_x1;

  dcross_drl[2][0] = -y[1] / q + p[0] * d_invq_d_x2;
  dcross_drl[2][1] = y[0] / q + p[1] * d_invq_d_x2;
  dcross_drl[2][2] = p[2] * d_invq_d_x2;

  // dcross/drm transposed
  dcross_drm[0][0] = p[0] * d_invq_d_y0;
  dcross_drm[0][1] = x[2] / q + p[1] * d_invq_d_y0;
  dcross_drm[0][2] = -x[1] / q + p[2] * d_invq_d_y0;

  dcross_drm[1][0] = -x[2] / q + p[0] * d_invq_d_y1;
  dcross_drm[1][1] = p[1] * d_invq_d_y1;
  dcross_drm[1][2] = x[0] / q + p[2] * d_invq_d_y1;

  dcross_drm[2][0] = x[1] / q + p[0] * d_invq_d_y2;
  dcross_drm[2][1] = -x[0] / q + p[1] * d_invq_d_y2;
  dcross_drm[2][2] = p[2] * d_invq_d_y2;

  // dcross/drk transposed
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) { dcross_drk[i][j] = -(dcross_drl[i][j] + dcross_drm[i][j]); }
  }
}
