/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Wengen Ouyang (Tel Aviv University)
   e-mail: w.g.ouyang at gmail dot com

   This is a full version of the potential described in
   [Maaravi et al, J. Phys. Chem. C 121, 22826-22835 (2017)]
   The definition of normals are the same as that in
   [Kolmogorov & Crespi, Phys. Rev. B 71, 235415 (2005)]
------------------------------------------------------------------------- */

#include "pair_ilp_graphene_hbn.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "interlayer_taper.h"
#include "memory.h"
#include "my_page.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace InterLayer;

#define MAXLINE 1024
#define DELTA 4
#define PGDELTA 1

static const char cite_ilp[] =
    "ilp/graphene/hbn potential doi:10.1021/acs.nanolett.8b02848\n"
    "@Article{Ouyang2018\n"
    " author = {W. Ouyang, D. Mandelli, M. Urbakh, and O. Hod},\n"
    " title = {Nanoserpents: Graphene Nanoribbon Motion on Two-Dimensional Hexagonal Materials},\n"
    " journal = {Nano Letters},\n"
    " volume =  18,\n"
    " pages =   {6009}\n"
    " year =    2018,\n"
    "}\n\n";

/* ---------------------------------------------------------------------- */

PairILPGrapheneHBN::PairILPGrapheneHBN(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);

  if (lmp->citeme) lmp->citeme->add(cite_ilp);

  nextra = 2;
  pvector = new double[nextra];

  // initialize element to parameter maps
  params = nullptr;
  cutILPsq = nullptr;

  nmax = 0;
  maxlocal = 0;
  ILP_numneigh = nullptr;
  ILP_firstneigh = nullptr;
  ipage = nullptr;
  pgsize = oneatom = 0;

  normal = nullptr;
  dnormal = nullptr;
  dnormdri = nullptr;

  // always compute energy offset
  offset_flag = 1;

  // turn on the taper function by default
  tap_flag = 1;
}

/* ---------------------------------------------------------------------- */

PairILPGrapheneHBN::~PairILPGrapheneHBN()
{
  memory->destroy(ILP_numneigh);
  memory->sfree(ILP_firstneigh);
  delete[] ipage;
  delete[] pvector;
  memory->destroy(normal);
  memory->destroy(dnormal);
  memory->destroy(dnormdri);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(offset);
  }

  memory->destroy(elem2param);
  memory->destroy(cutILPsq);
  memory->sfree(params);
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairILPGrapheneHBN::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(offset, n, n, "pair:offset");
  map = new int[n];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairILPGrapheneHBN::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 2) error->all(FLERR, "Illegal pair_style command");
  if (!utils::strmatch(force->pair_style, "^hybrid/overlay"))
    error->all(FLERR, "Pair style ilp/graphene/hbn must be used as sub-style with hybrid/overlay");

  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  if (narg == 2) tap_flag = utils::numeric(FLERR, arg[1], false, lmp);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairILPGrapheneHBN::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  map_element2type(narg - 3, arg + 3);
  read_file(arg[2]);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairILPGrapheneHBN::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  if (!offset_flag) error->all(FLERR, "Must use 'pair_modify shift yes' with this pair style");

  if (offset_flag && (cut_global > 0.0)) {
    int iparam_ij = elem2param[map[i]][map[j]];
    Param &p = params[iparam_ij];
    offset[i][j] =
        -p.C6 * pow(1.0 / cut_global, 6) / (1.0 + exp(-p.d * (cut_global / p.seff - 1.0)));
  } else
    offset[i][j] = 0.0;
  offset[j][i] = offset[i][j];

  return cut_global;
}

/* ----------------------------------------------------------------------
   read Interlayer potential file
------------------------------------------------------------------------- */

void PairILPGrapheneHBN::read_file(char *filename)
{
  memory->sfree(params);
  params = nullptr;
  nparams = maxparam = 0;

  // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, filename, "ilp/graphene/hbn", unit_convert_flag);
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

        // load up parameter settings and error check their values

        params[nparams].ielement = ielement;
        params[nparams].jelement = jelement;
        params[nparams].z0 = values.next_double();
        params[nparams].alpha = values.next_double();
        params[nparams].delta = values.next_double();
        params[nparams].epsilon = values.next_double();
        params[nparams].C = values.next_double();
        params[nparams].d = values.next_double();
        params[nparams].sR = values.next_double();
        params[nparams].reff = values.next_double();
        params[nparams].C6 = values.next_double();
        // S provides a convenient scaling of all energies
        params[nparams].S = values.next_double();
        params[nparams].rcut = values.next_double();

      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      // energies in meV further scaled by S
      // S = 43.3634 meV = 1 kcal/mol

      double meV = 1e-3 * params[nparams].S;
      if (unit_convert) meV *= conversion_factor;

      params[nparams].C *= meV;
      params[nparams].C6 *= meV;
      params[nparams].epsilon *= meV;

      // precompute some quantities
      params[nparams].delta2inv = pow(params[nparams].delta, -2.0);
      params[nparams].lambda = params[nparams].alpha / params[nparams].z0;
      params[nparams].seff = params[nparams].sR * params[nparams].reff;

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
  memory->destroy(cutILPsq);
  memory->create(elem2param, nelements, nelements, "pair:elem2param");
  memory->create(cutILPsq, nelements, nelements, "pair:cutILPsq");
  for (int i = 0; i < nelements; i++) {
    for (int j = 0; j < nelements; j++) {
      int n = -1;
      for (int m = 0; m < nparams; m++) {
        if (i == params[m].ielement && j == params[m].jelement) {
          if (n >= 0) error->all(FLERR, "ILP potential file has duplicate entry");
          n = m;
        }
      }
      if (n < 0) error->all(FLERR, "Potential file is missing an entry");
      elem2param[i][j] = n;
      cutILPsq[i][j] = params[n].rcut * params[n].rcut;
    }
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairILPGrapheneHBN::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style ilp/graphene/hbn requires newton pair on");
  if (!atom->molecule_flag)
    error->all(FLERR, "Pair style ilp/graphene/hbn requires atom attribute molecule");

  // need a full neighbor list, including neighbors of ghosts

  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->ghost = 1;

  // local ILP neighbor list
  // create pages if first time or if neighbor pgsize/oneatom has changed

  int create = 0;
  if (ipage == nullptr) create = 1;
  if (pgsize != neighbor->pgsize) create = 1;
  if (oneatom != neighbor->oneatom) create = 1;

  if (create) {
    delete[] ipage;
    pgsize = neighbor->pgsize;
    oneatom = neighbor->oneatom;

    int nmypage = comm->nthreads;
    ipage = new MyPage<int>[nmypage];
    for (int i = 0; i < nmypage; i++) ipage[i].init(oneatom, pgsize, PGDELTA);
  }
}

/* ---------------------------------------------------------------------- */
void PairILPGrapheneHBN::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  pvector[0] = pvector[1] = 0.0;

  // Build full neighbor list
  ILP_neigh();
  // Calculate the normals and its derivatives
  calc_normal();
  // Calculate the van der Waals force and energy
  calc_FvdW(eflag, vflag);
  // Calculate the repulsive force and energy
  calc_FRep(eflag, vflag);

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   van der Waals forces and energy
------------------------------------------------------------------------- */

void PairILPGrapheneHBN::calc_FvdW(int eflag, int /* vflag */)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  tagint itag, jtag;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r, Rcut, r2inv, r6inv, r8inv, Tap, dTap, Vilp, TSvdw, TSvdw2inv, fsum;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itag = tag[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
      jtag = tag[j];

      // two-body interactions from full neighbor list, skip half of them
      if (itag > jtag) {
        if ((itag + jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag + jtag) % 2 == 1) continue;
      } else {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp && x[j][1] < ytmp) continue;
        if (x[j][2] == ztmp && x[j][1] == ytmp && x[j][0] < xtmp) continue;
      }

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      // only include the interaction between different layers
      if (rsq < cutsq[itype][jtype] && atom->molecule[i] != atom->molecule[j]) {

        int iparam_ij = elem2param[map[itype]][map[jtype]];
        Param &p = params[iparam_ij];

        r = sqrt(rsq);
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        r8inv = r6inv * r2inv;
        // turn on/off taper function
        if (tap_flag) {
          Rcut = sqrt(cutsq[itype][jtype]);
          Tap = calc_Tap(r, Rcut);
          dTap = calc_dTap(r, Rcut);
        } else {
          Tap = 1.0;
          dTap = 0.0;
        }

        TSvdw = 1.0 + exp(-p.d * (r / p.seff - 1.0));
        TSvdw2inv = pow(TSvdw, -2.0);
        Vilp = -p.C6 * r6inv / TSvdw;

        // derivatives
        fpair = -6.0 * p.C6 * r8inv / TSvdw +
            p.C6 * p.d / p.seff * (TSvdw - 1.0) * TSvdw2inv * r8inv * r;
        fsum = fpair * Tap - Vilp * dTap / r;

        f[i][0] += fsum * delx;
        f[i][1] += fsum * dely;
        f[i][2] += fsum * delz;
        f[j][0] -= fsum * delx;
        f[j][1] -= fsum * dely;
        f[j][2] -= fsum * delz;

        if (eflag) pvector[0] += evdwl = Vilp * Tap;
        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fsum, delx, dely, delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Repulsive forces and energy
------------------------------------------------------------------------- */

void PairILPGrapheneHBN::calc_FRep(int eflag, int /* vflag */)
{
  int i, j, ii, jj, inum, jnum, itype, jtype, k, kk;
  double prodnorm1, fkcx, fkcy, fkcz;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair, fpair1;
  double rsq, r, Rcut, rhosq1, exp0, exp1, Tap, dTap, Vilp;
  double frho1, Erep, fsum, rdsq1;
  int *ilist, *jlist, *numneigh, **firstneigh;
  int *ILP_neighs_i;

  evdwl = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double dprodnorm1[3] = {0.0, 0.0, 0.0};
  double fp1[3] = {0.0, 0.0, 0.0};
  double fprod1[3] = {0.0, 0.0, 0.0};
  double delki[3] = {0.0, 0.0, 0.0};
  double fk[3] = {0.0, 0.0, 0.0};

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  //calculate exp(-lambda*(r-z0))*[epsilon/2 + f(rho_ij)]
  // loop over neighbors of owned atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      // only include the interaction between different layers
      if (rsq < cutsq[itype][jtype] && atom->molecule[i] != atom->molecule[j]) {

        int iparam_ij = elem2param[map[itype]][map[jtype]];
        Param &p = params[iparam_ij];

        r = sqrt(rsq);
        // turn on/off taper function
        if (tap_flag) {
          Rcut = sqrt(cutsq[itype][jtype]);
          Tap = calc_Tap(r, Rcut);
          dTap = calc_dTap(r, Rcut);
        } else {
          Tap = 1.0;
          dTap = 0.0;
        }

        // Calculate the transverse distance
        prodnorm1 = normal[i][0] * delx + normal[i][1] * dely + normal[i][2] * delz;
        rhosq1 = rsq - prodnorm1 * prodnorm1;    // rho_ij
        rdsq1 = rhosq1 * p.delta2inv;            // (rho_ij/delta)^2

        // store exponents
        exp0 = exp(-p.lambda * (r - p.z0));
        exp1 = exp(-rdsq1);

        frho1 = exp1 * p.C;
        Erep = 0.5 * p.epsilon + frho1;
        Vilp = exp0 * Erep;

        // derivatives
        fpair = p.lambda * exp0 / r * Erep;
        fpair1 = 2.0 * exp0 * frho1 * p.delta2inv;
        fsum = fpair + fpair1;
        // derivatives of the product of rij and ni, the result is a vector
        dprodnorm1[0] =
            dnormdri[0][0][i] * delx + dnormdri[1][0][i] * dely + dnormdri[2][0][i] * delz;
        dprodnorm1[1] =
            dnormdri[0][1][i] * delx + dnormdri[1][1][i] * dely + dnormdri[2][1][i] * delz;
        dprodnorm1[2] =
            dnormdri[0][2][i] * delx + dnormdri[1][2][i] * dely + dnormdri[2][2][i] * delz;
        fp1[0] = prodnorm1 * normal[i][0] * fpair1;
        fp1[1] = prodnorm1 * normal[i][1] * fpair1;
        fp1[2] = prodnorm1 * normal[i][2] * fpair1;
        fprod1[0] = prodnorm1 * dprodnorm1[0] * fpair1;
        fprod1[1] = prodnorm1 * dprodnorm1[1] * fpair1;
        fprod1[2] = prodnorm1 * dprodnorm1[2] * fpair1;

        fkcx = (delx * fsum - fp1[0]) * Tap - Vilp * dTap * delx / r;
        fkcy = (dely * fsum - fp1[1]) * Tap - Vilp * dTap * dely / r;
        fkcz = (delz * fsum - fp1[2]) * Tap - Vilp * dTap * delz / r;

        f[i][0] += fkcx - fprod1[0] * Tap;
        f[i][1] += fkcy - fprod1[1] * Tap;
        f[i][2] += fkcz - fprod1[2] * Tap;
        f[j][0] -= fkcx;
        f[j][1] -= fkcy;
        f[j][2] -= fkcz;

        // calculate the forces acted on the neighbors of atom i from atom j
        ILP_neighs_i = ILP_firstneigh[i];
        for (kk = 0; kk < ILP_numneigh[i]; kk++) {
          k = ILP_neighs_i[kk];
          if (k == i) continue;
          // derivatives of the product of rij and ni respect to rk, k=0,1,2, where atom k is the neighbors of atom i
          dprodnorm1[0] = dnormal[0][0][kk][i] * delx + dnormal[1][0][kk][i] * dely +
              dnormal[2][0][kk][i] * delz;
          dprodnorm1[1] = dnormal[0][1][kk][i] * delx + dnormal[1][1][kk][i] * dely +
              dnormal[2][1][kk][i] * delz;
          dprodnorm1[2] = dnormal[0][2][kk][i] * delx + dnormal[1][2][kk][i] * dely +
              dnormal[2][2][kk][i] * delz;
          fk[0] = (-prodnorm1 * dprodnorm1[0] * fpair1) * Tap;
          fk[1] = (-prodnorm1 * dprodnorm1[1] * fpair1) * Tap;
          fk[2] = (-prodnorm1 * dprodnorm1[2] * fpair1) * Tap;
          f[k][0] += fk[0];
          f[k][1] += fk[1];
          f[k][2] += fk[2];
          delki[0] = x[k][0] - x[i][0];
          delki[1] = x[k][1] - x[i][1];
          delki[2] = x[k][2] - x[i][2];
          if (evflag)
            ev_tally_xyz(k, i, nlocal, newton_pair, 0.0, 0.0, fk[0], fk[1], fk[2], delki[0],
                         delki[1], delki[2]);
        }

        if (eflag) pvector[1] += evdwl = Tap * Vilp;
        if (evflag)
          ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0, fkcx, fkcy, fkcz, delx, dely, delz);
      }
    }    // loop over jj
  }      // loop over ii
}

/* ----------------------------------------------------------------------
   create ILP neighbor list from main neighbor list to calculate normals
------------------------------------------------------------------------- */

void PairILPGrapheneHBN::ILP_neigh()
{
  int i, j, ii, jj, n, allnum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;
  int *neighptr;

  double **x = atom->x;
  int *type = atom->type;

  if (atom->nmax > maxlocal) {
    maxlocal = atom->nmax;
    memory->destroy(ILP_numneigh);
    memory->sfree(ILP_firstneigh);
    memory->create(ILP_numneigh, maxlocal, "ILPGrapheneHBN:numneigh");
    ILP_firstneigh =
        (int **) memory->smalloc(maxlocal * sizeof(int *), "ILPGrapheneHBN:firstneigh");
  }

  allnum = list->inum + list->gnum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // store all ILP neighs of owned and ghost atoms
  // scan full neighbor list of I

  ipage->reset();

  for (ii = 0; ii < allnum; ii++) {
    i = ilist[ii];

    n = 0;
    neighptr = ipage->vget();

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = map[type[i]];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = map[type[j]];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq != 0 && rsq < cutILPsq[itype][jtype] && atom->molecule[i] == atom->molecule[j]) {
        neighptr[n++] = j;
      }
    }    // loop over jj

    ILP_firstneigh[i] = neighptr;
    ILP_numneigh[i] = n;
    if (n > 3)
      error->one(FLERR,
                 "There are too many neighbors for some atoms, please check your configuration");

    ipage->vgot(n);
    if (ipage->status()) error->one(FLERR, "Neighbor list overflow, boost neigh_modify one");
  }
}

/* ----------------------------------------------------------------------
   Calculate the normals for each atom
------------------------------------------------------------------------- */
void PairILPGrapheneHBN::calc_normal()
{
  int i, j, ii, jj, inum, jnum;
  int cont, id, ip, m;
  double nn, xtp, ytp, ztp, delx, dely, delz, nn2;
  int *ilist, *jlist;
  double pv12[3], pv31[3], pv23[3], n1[3], dni[3], dnn[3][3], vet[3][3], dpvdri[3][3];
  double dn1[3][3][3], dpv12[3][3][3], dpv23[3][3][3], dpv31[3][3][3];

  double **x = atom->x;

  // grow normal array if necessary

  if (atom->nmax > nmax) {
    memory->destroy(normal);
    memory->destroy(dnormal);
    memory->destroy(dnormdri);
    nmax = atom->nmax;
    memory->create(normal, nmax, 3, "ILPGrapheneHBN:normal");
    memory->create(dnormdri, 3, 3, nmax, "ILPGrapheneHBN:dnormdri");
    memory->create(dnormal, 3, 3, 3, nmax, "ILPGrapheneHBN:dnormal");
  }

  inum = list->inum;
  ilist = list->ilist;
  //Calculate normals
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    //   Initialize the arrays
    for (id = 0; id < 3; id++) {
      pv12[id] = 0.0;
      pv31[id] = 0.0;
      pv23[id] = 0.0;
      n1[id] = 0.0;
      dni[id] = 0.0;
      normal[i][id] = 0.0;
      for (ip = 0; ip < 3; ip++) {
        vet[ip][id] = 0.0;
        dnn[ip][id] = 0.0;
        dpvdri[ip][id] = 0.0;
        dnormdri[ip][id][i] = 0.0;
        for (m = 0; m < 3; m++) {
          dpv12[ip][id][m] = 0.0;
          dpv31[ip][id][m] = 0.0;
          dpv23[ip][id][m] = 0.0;
          dn1[ip][id][m] = 0.0;
          dnormal[ip][id][m][i] = 0.0;
        }
      }
    }

    xtp = x[i][0];
    ytp = x[i][1];
    ztp = x[i][2];

    cont = 0;
    jlist = ILP_firstneigh[i];
    jnum = ILP_numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = x[j][0] - xtp;
      dely = x[j][1] - ytp;
      delz = x[j][2] - ztp;
      vet[cont][0] = delx;
      vet[cont][1] = dely;
      vet[cont][2] = delz;
      cont++;
    }

    if (cont <= 1) {
      normal[i][0] = 0.0;
      normal[i][1] = 0.0;
      normal[i][2] = 1.0;
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dnormdri[id][ip][i] = 0.0;
          for (m = 0; m < 3; m++) { dnormal[id][ip][m][i] = 0.0; }
        }
      }
    } else if (cont == 2) {
      pv12[0] = vet[0][1] * vet[1][2] - vet[1][1] * vet[0][2];
      pv12[1] = vet[0][2] * vet[1][0] - vet[1][2] * vet[0][0];
      pv12[2] = vet[0][0] * vet[1][1] - vet[1][0] * vet[0][1];
      // derivatives of pv12[0] to ri
      dpvdri[0][0] = 0.0;
      dpvdri[0][1] = vet[0][2] - vet[1][2];
      dpvdri[0][2] = vet[1][1] - vet[0][1];
      // derivatives of pv12[1] to ri
      dpvdri[1][0] = vet[1][2] - vet[0][2];
      dpvdri[1][1] = 0.0;
      dpvdri[1][2] = vet[0][0] - vet[1][0];
      // derivatives of pv12[2] to ri
      dpvdri[2][0] = vet[0][1] - vet[1][1];
      dpvdri[2][1] = vet[1][0] - vet[0][0];
      dpvdri[2][2] = 0.0;

      dpv12[0][0][0] = 0.0;
      dpv12[0][1][0] = vet[1][2];
      dpv12[0][2][0] = -vet[1][1];
      dpv12[1][0][0] = -vet[1][2];
      dpv12[1][1][0] = 0.0;
      dpv12[1][2][0] = vet[1][0];
      dpv12[2][0][0] = vet[1][1];
      dpv12[2][1][0] = -vet[1][0];
      dpv12[2][2][0] = 0.0;

      // derivatives respect to the second neighbor, atom l
      dpv12[0][0][1] = 0.0;
      dpv12[0][1][1] = -vet[0][2];
      dpv12[0][2][1] = vet[0][1];
      dpv12[1][0][1] = vet[0][2];
      dpv12[1][1][1] = 0.0;
      dpv12[1][2][1] = -vet[0][0];
      dpv12[2][0][1] = -vet[0][1];
      dpv12[2][1][1] = vet[0][0];
      dpv12[2][2][1] = 0.0;

      // derivatives respect to the third neighbor, atom n
      // derivatives of pv12 to rn is zero
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) { dpv12[id][ip][2] = 0.0; }
      }

      n1[0] = pv12[0];
      n1[1] = pv12[1];
      n1[2] = pv12[2];
      // the magnitude of the normal vector
      nn2 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
      nn = sqrt(nn2);
      if (nn == 0) error->one(FLERR, "The magnitude of the normal vector is zero");
      // the unit normal vector
      normal[i][0] = n1[0] / nn;
      normal[i][1] = n1[1] / nn;
      normal[i][2] = n1[2] / nn;
      // derivatives of nn, dnn:3x1 vector
      dni[0] = (n1[0] * dpvdri[0][0] + n1[1] * dpvdri[1][0] + n1[2] * dpvdri[2][0]) / nn;
      dni[1] = (n1[0] * dpvdri[0][1] + n1[1] * dpvdri[1][1] + n1[2] * dpvdri[2][1]) / nn;
      dni[2] = (n1[0] * dpvdri[0][2] + n1[1] * dpvdri[1][2] + n1[2] * dpvdri[2][2]) / nn;
      // derivatives of unit vector ni respect to ri, the result is 3x3 matrix
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dnormdri[id][ip][i] = dpvdri[id][ip] / nn - n1[id] * dni[ip] / nn2;
        }
      }
      // derivatives of non-normalized normal vector, dn1:3x3x3 array
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          for (m = 0; m < 3; m++) { dn1[id][ip][m] = dpv12[id][ip][m]; }
        }
      }
      // derivatives of nn, dnn:3x3 vector
      // dnn[id][m]: the derivative of nn respect to r[id][m], id,m=0,1,2
      // r[id][m]: the id's component of atom m
      for (m = 0; m < 3; m++) {
        for (id = 0; id < 3; id++) {
          dnn[id][m] = (n1[0] * dn1[0][id][m] + n1[1] * dn1[1][id][m] + n1[2] * dn1[2][id][m]) / nn;
        }
      }
      // dnormal[id][ip][m][i]: the derivative of normal[id] respect to r[ip][m], id,ip=0,1,2
      // for atom m, which is a neighbor atom of atom i, m=0,jnum-1
      for (m = 0; m < 3; m++) {
        for (id = 0; id < 3; id++) {
          for (ip = 0; ip < 3; ip++) {
            dnormal[id][ip][m][i] = dn1[id][ip][m] / nn - n1[id] * dnn[ip][m] / nn2;
          }
        }
      }
    }
    //##############################################################################################

    else if (cont == 3) {
      pv12[0] = vet[0][1] * vet[1][2] - vet[1][1] * vet[0][2];
      pv12[1] = vet[0][2] * vet[1][0] - vet[1][2] * vet[0][0];
      pv12[2] = vet[0][0] * vet[1][1] - vet[1][0] * vet[0][1];
      // derivatives respect to the first neighbor, atom k
      dpv12[0][0][0] = 0.0;
      dpv12[0][1][0] = vet[1][2];
      dpv12[0][2][0] = -vet[1][1];
      dpv12[1][0][0] = -vet[1][2];
      dpv12[1][1][0] = 0.0;
      dpv12[1][2][0] = vet[1][0];
      dpv12[2][0][0] = vet[1][1];
      dpv12[2][1][0] = -vet[1][0];
      dpv12[2][2][0] = 0.0;
      // derivatives respect to the second neighbor, atom l
      dpv12[0][0][1] = 0.0;
      dpv12[0][1][1] = -vet[0][2];
      dpv12[0][2][1] = vet[0][1];
      dpv12[1][0][1] = vet[0][2];
      dpv12[1][1][1] = 0.0;
      dpv12[1][2][1] = -vet[0][0];
      dpv12[2][0][1] = -vet[0][1];
      dpv12[2][1][1] = vet[0][0];
      dpv12[2][2][1] = 0.0;

      // derivatives respect to the third neighbor, atom n
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) { dpv12[id][ip][2] = 0.0; }
      }

      pv31[0] = vet[2][1] * vet[0][2] - vet[0][1] * vet[2][2];
      pv31[1] = vet[2][2] * vet[0][0] - vet[0][2] * vet[2][0];
      pv31[2] = vet[2][0] * vet[0][1] - vet[0][0] * vet[2][1];
      // derivatives respect to the first neighbor, atom k
      dpv31[0][0][0] = 0.0;
      dpv31[0][1][0] = -vet[2][2];
      dpv31[0][2][0] = vet[2][1];
      dpv31[1][0][0] = vet[2][2];
      dpv31[1][1][0] = 0.0;
      dpv31[1][2][0] = -vet[2][0];
      dpv31[2][0][0] = -vet[2][1];
      dpv31[2][1][0] = vet[2][0];
      dpv31[2][2][0] = 0.0;
      // derivatives respect to the third neighbor, atom n
      dpv31[0][0][2] = 0.0;
      dpv31[0][1][2] = vet[0][2];
      dpv31[0][2][2] = -vet[0][1];
      dpv31[1][0][2] = -vet[0][2];
      dpv31[1][1][2] = 0.0;
      dpv31[1][2][2] = vet[0][0];
      dpv31[2][0][2] = vet[0][1];
      dpv31[2][1][2] = -vet[0][0];
      dpv31[2][2][2] = 0.0;
      // derivatives respect to the second neighbor, atom l
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) { dpv31[id][ip][1] = 0.0; }
      }

      pv23[0] = vet[1][1] * vet[2][2] - vet[2][1] * vet[1][2];
      pv23[1] = vet[1][2] * vet[2][0] - vet[2][2] * vet[1][0];
      pv23[2] = vet[1][0] * vet[2][1] - vet[2][0] * vet[1][1];
      // derivatives respect to the second neighbor, atom k
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) { dpv23[id][ip][0] = 0.0; }
      }
      // derivatives respect to the second neighbor, atom l
      dpv23[0][0][1] = 0.0;
      dpv23[0][1][1] = vet[2][2];
      dpv23[0][2][1] = -vet[2][1];
      dpv23[1][0][1] = -vet[2][2];
      dpv23[1][1][1] = 0.0;
      dpv23[1][2][1] = vet[2][0];
      dpv23[2][0][1] = vet[2][1];
      dpv23[2][1][1] = -vet[2][0];
      dpv23[2][2][1] = 0.0;
      // derivatives respect to the third neighbor, atom n
      dpv23[0][0][2] = 0.0;
      dpv23[0][1][2] = -vet[1][2];
      dpv23[0][2][2] = vet[1][1];
      dpv23[1][0][2] = vet[1][2];
      dpv23[1][1][2] = 0.0;
      dpv23[1][2][2] = -vet[1][0];
      dpv23[2][0][2] = -vet[1][1];
      dpv23[2][1][2] = vet[1][0];
      dpv23[2][2][2] = 0.0;

      //############################################################################################
      // average the normal vectors by using the 3 neighboring planes
      n1[0] = (pv12[0] + pv31[0] + pv23[0]) / cont;
      n1[1] = (pv12[1] + pv31[1] + pv23[1]) / cont;
      n1[2] = (pv12[2] + pv31[2] + pv23[2]) / cont;
      // the magnitude of the normal vector
      nn2 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
      nn = sqrt(nn2);
      if (nn == 0) error->one(FLERR, "The magnitude of the normal vector is zero");
      // the unit normal vector
      normal[i][0] = n1[0] / nn;
      normal[i][1] = n1[1] / nn;
      normal[i][2] = n1[2] / nn;

      // for the central atoms, dnormdri is always zero
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) { dnormdri[id][ip][i] = 0.0; }
      }

      // derivatives of non-normalized normal vector, dn1:3x3x3 array
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          for (m = 0; m < 3; m++) {
            dn1[id][ip][m] = (dpv12[id][ip][m] + dpv23[id][ip][m] + dpv31[id][ip][m]) / cont;
          }
        }
      }
      // derivatives of nn, dnn:3x3 vector
      // dnn[id][m]: the derivative of nn respect to r[id][m], id,m=0,1,2
      // r[id][m]: the id's component of atom m
      for (m = 0; m < 3; m++) {
        for (id = 0; id < 3; id++) {
          dnn[id][m] = (n1[0] * dn1[0][id][m] + n1[1] * dn1[1][id][m] + n1[2] * dn1[2][id][m]) / nn;
        }
      }
      // dnormal[id][ip][m][i]: the derivative of normal[id] respect to r[ip][m], id,ip=0,1,2
      // for atom m, which is a neighbor atom of atom i, m=0,jnum-1
      for (m = 0; m < 3; m++) {
        for (id = 0; id < 3; id++) {
          for (ip = 0; ip < 3; ip++) {
            dnormal[id][ip][m][i] = dn1[id][ip][m] / nn - n1[id] * dnn[ip][m] / nn2;
          }
        }
      }
    } else {
      error->one(FLERR, "There are too many neighbors for calculating normals");
    }

    //##############################################################################################
  }
}

/* ---------------------------------------------------------------------- */

double PairILPGrapheneHBN::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                                  double /*factor_coul*/, double factor_lj, double &fforce)
{
  double r, r2inv, r6inv, r8inv, forcelj, philj, fpair;
  double Tap, dTap, Vilp, TSvdw, TSvdw2inv;

  int iparam_ij = elem2param[map[itype]][map[jtype]];
  Param &p = params[iparam_ij];

  r = sqrt(rsq);
  // turn on/off taper function
  if (tap_flag) {
    Tap = calc_Tap(r, sqrt(cutsq[itype][jtype]));
    dTap = calc_dTap(r, sqrt(cutsq[itype][jtype]));
  } else {
    Tap = 1.0;
    dTap = 0.0;
  }

  r2inv = 1.0 / rsq;
  r6inv = r2inv * r2inv * r2inv;
  r8inv = r2inv * r6inv;

  TSvdw = 1.0 + exp(-p.d * (r / p.seff - 1.0));
  TSvdw2inv = pow(TSvdw, -2.0);
  Vilp = -p.C6 * r6inv / TSvdw;
  // derivatives
  fpair = -6.0 * p.C6 * r8inv / TSvdw + p.d / p.seff * p.C6 * (TSvdw - 1.0) * r6inv * TSvdw2inv / r;
  forcelj = fpair;
  fforce = factor_lj * (forcelj * Tap - Vilp * dTap / r);

  philj = Vilp * Tap;
  return factor_lj * philj;
}
