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
   Contributing author: Ray Shan (Sandia)
------------------------------------------------------------------------- */

#include "pair_coul_streitz.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

#define DELTA 4
#define PGDELTA 1
#define MAXNEIGH 24

/* ---------------------------------------------------------------------- */

PairCoulStreitz::PairCoulStreitz(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  nmax = 0;

  params = nullptr;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairCoulStreitz::~PairCoulStreitz()
{
  memory->sfree(params);
  memory->destroy(elem1param);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
    memory->destroy(qeq_x);
    memory->destroy(qeq_j);
    memory->destroy(qeq_g);
    memory->destroy(qeq_z);
    memory->destroy(qeq_c);
  }
}

/* ---------------------------------------------------------------------- */

void PairCoulStreitz::allocate()
{
 allocated = 1;
 int n = atom->ntypes;

 memory->create(setflag,n+1,n+1,"pair:setflag");
 memory->create(cutsq,n+1,n+1,"pair:cutsq");
 memory->create(scale,n+1,n+1,"pair:scale");
 memory->create(qeq_x,n+1,"pair:qeq_x");
 memory->create(qeq_j,n+1,"pair:qeq_j");
 memory->create(qeq_g,n+1,"pair:qeq_g");
 memory->create(qeq_z,n+1,"pair:qeq_z");
 memory->create(qeq_c,n+1,"pair:qeq_c");

 map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairCoulStreitz::settings(int narg, char **arg)
{
  if (narg < 2) error->all(FLERR,"Illegal pair_style command");

  cut_coul = utils::numeric(FLERR,arg[0],false,lmp);

  if (strcmp(arg[1],"wolf") == 0) {
    kspacetype = 1;
    g_wolf = utils::numeric(FLERR,arg[2],false,lmp);
  } else if (strcmp(arg[1],"ewald") == 0) {
    ewaldflag = pppmflag = 1;
    kspacetype = 2;
  } else {
    error->all(FLERR,"Illegal pair_style command");
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairCoulStreitz::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  map_element2type(narg-3, arg+3);

  // read potential file and initialize potential parameters

  read_file(arg[2]);
  setup_params();

  const int n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
        scale[i][j] = 1.0;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairCoulStreitz::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair style coul/streitz requires atom attribute q");

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  cut_coulsq = cut_coul * cut_coul;

  // insure use of KSpace long-range solver when ewald specified, set g_ewald

  if (ewaldflag) {
    if (force->kspace == nullptr)
      error->all(FLERR,"Pair style requires a KSpace style");
    g_ewald = force->kspace->g_ewald;
  }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairCoulStreitz::init_one(int i, int j)
{
  scale[j][i] = scale[i][j];

  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  return cut_coul;
}

/* ---------------------------------------------------------------------- */

void PairCoulStreitz::read_file(char *file)
{
  memory->sfree(params);
  params = nullptr;
  nparams = 0;
  maxparam = 0;

  // open file on proc 0
  if (comm->me == 0) {
    PotentialFileReader reader(lmp, file, "coul/streitz");
    char * line;

    while ((line = reader.next_line(NPARAMS_PER_LINE))) {
      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();

        // ielement = 1st args
        int ielement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;

        // load up parameter settings and error check their values

        if (nparams == maxparam) {
          maxparam += DELTA;
          params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
                                              "pair:params");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(params + nparams, 0, DELTA*sizeof(Param));
        }


        params[nparams].ielement = ielement;
        params[nparams].chi = values.next_double();
        params[nparams].eta = values.next_double();
        params[nparams].gamma = values.next_double();
        params[nparams].zeta = values.next_double();
        params[nparams].zcore = values.next_double();

      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      // parameter sanity check

      if (params[nparams].eta < 0.0 || params[nparams].zeta < 0.0 ||
          params[nparams].zcore < 0.0 || params[nparams].gamma != 0.0 )
        error->one(FLERR,"Illegal coul/streitz parameter");

      nparams++;
    }
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    params = (Param *) memory->srealloc(params,maxparam*sizeof(Param), "pair:params");
  }

  MPI_Bcast(params, maxparam*sizeof(Param), MPI_BYTE, 0, world);
}

/* ---------------------------------------------------------------------- */

void PairCoulStreitz::setup_params()
{
  int i,m,n;

  // set elem1param

  memory->destroy(elem1param);
  memory->create(elem1param,nelements,"pair:elem1param");

  for (i = 0; i < nelements; i++) {
    n = -1;
    for (m = 0; m < nparams; m++) {
      if (i == params[m].ielement) {
        if (n >= 0) error->all(FLERR,"Potential file has duplicate entry");
        n = m;
      }
    }
    if (n < 0) error->all(FLERR,"Potential file is missing an entry");
    elem1param[i] = n;
  }

  // Wolf sum self energy

  if (kspacetype == 1) {
    double a = g_wolf;
    double r = cut_coul;
    double ar = a*r;

    woself = 0.50*erfc(ar)/r + a/MY_PIS;  // kc constant not yet multiplied
    dwoself = -(erfc(ar)/r/r + 2.0*a/MY_PIS*exp(-ar*ar)/r);
  }
}

/* ---------------------------------------------------------------------- */

void PairCoulStreitz::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum;
  int itype, jtype, iparam_i,iparam_j;
  int *ilist, *jlist, *numneigh, **firstneigh;

  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  double xtmp, ytmp, ztmp, ecoul, fpair;
  double qi, qj, selfion, r, rsq, delr[3];
  double zei, zej, zj, ci_jfi, dci_jfi, ci_fifj, dci_fifj;
  double forcecoul, factor_coul;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  double *special_coul = force->special_coul;

  ecoul = 0.0;
  selfion = fpair = 0.0;
  ci_jfi = ci_fifj = dci_jfi = dci_fifj = 0.0;
  forcecoul = 0.0;

  ev_init(eflag,vflag);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Wolf sum

  if (kspacetype == 1) {

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = map[type[i]];
    iparam_i = elem1param[itype];
    qi = q[i];
    zei = params[iparam_i].zeta;

    // self energy: ionization + wolf sum

    selfion = self(&params[iparam_i],qi);

    if (evflag) ev_tally(i,i,nlocal,0,0.0,selfion,0.0,0.0,0.0,0.0);

    // two-body interaction

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      jtype = map[type[j]];
      iparam_j = elem1param[jtype];
      qj = q[j];
      zej = params[iparam_j].zeta;
      zj = params[iparam_j].zcore;
      factor_coul = special_coul[sbmask(j)];

      delr[0] = xtmp - x[j][0];
      delr[1] = ytmp - x[j][1];
      delr[2] = ztmp - x[j][2];
      rsq = delr[0]*delr[0] + delr[1]*delr[1] + delr[2]*delr[2];

      if (rsq > cut_coulsq) continue;

      r = sqrt(rsq);

      // Streitz-Mintmire Coulomb integrals

      coulomb_integral_wolf(zei, zej, r, ci_jfi, dci_jfi, ci_fifj, dci_fifj);

      // Wolf Sum

      wolf_sum(qi, qj, zj, r, ci_jfi, dci_jfi, ci_fifj, dci_fifj,
               ecoul, forcecoul);

      // Forces

      fpair = -forcecoul / r;

      f[i][0] += delr[0]*fpair;
      f[i][1] += delr[1]*fpair;
      f[i][2] += delr[2]*fpair;
      if (newton_pair || j < nlocal) {
        f[j][0] -= delr[0]*fpair;
        f[j][1] -= delr[1]*fpair;
        f[j][2] -= delr[2]*fpair;
      }

      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           0.0,ecoul,fpair,delr[0],delr[1],delr[2]);
    }

  }

  // Ewald Sum

  } else if (kspacetype == 2) {

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = map[type[i]];
    iparam_i = elem1param[itype];
    qi = q[i];
    zei = params[iparam_i].zeta;

    // self ionizition energy, only on i atom

    selfion = self(&params[iparam_i],qi);

    if (evflag) ev_tally(i,i,nlocal,0,0.0,selfion,0.0,0.0,0.0,0.0);

    // two-body interaction

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = map[type[j]];
      iparam_j = elem1param[jtype];
      qj = q[j];
      zej = params[iparam_j].zeta;
      zj = params[iparam_j].zcore;
      factor_coul = special_coul[sbmask(j)];

      delr[0] = xtmp - x[j][0];
      delr[1] = ytmp - x[j][1];
      delr[2] = ztmp - x[j][2];
      rsq = delr[0]*delr[0] + delr[1]*delr[1] + delr[2]*delr[2];

      if (rsq > cut_coulsq) continue;

      r = sqrt(rsq);

      // Streitz-Mintmire Coulomb integrals

      coulomb_integral_ewald(zei, zej, r, ci_jfi, dci_jfi, ci_fifj, dci_fifj);

      // Ewald: real-space

      ewald_sum(qi, qj, zj, r, ci_jfi, dci_jfi, ci_fifj, dci_fifj,
                      ecoul, forcecoul, factor_coul);

      // Forces

      fpair = -forcecoul / r;

      f[i][0] += delr[0]*fpair;
      f[i][1] += delr[1]*fpair;
      f[i][2] += delr[2]*fpair;
      if (newton_pair || j < nlocal) {
        f[j][0] -= delr[0]*fpair;
        f[j][1] -= delr[1]*fpair;
        f[j][2] -= delr[2]*fpair;
      }

      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           0.0,ecoul,fpair,delr[0],delr[1],delr[2]);
    }
  }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

double PairCoulStreitz::self(Param *param, double qi)
{
 double s1=param->chi, s2=param->eta;
 double qqrd2e = force->qqrd2e;

 if (kspacetype == 1) return 1.0*qi*(s1+qi*(0.50*s2 - qqrd2e*woself));

 if (kspacetype == 2) return 1.0*qi*(s1+qi*(0.50*s2));

 return 0.0;
}

/* ---------------------------------------------------------------------- */

void PairCoulStreitz::coulomb_integral_wolf(double zei, double zej, double r,
                  double &ci_jfi, double &dci_jfi, double &ci_fifj,
                  double &dci_fifj)
{
  double rinv = 1.0/r;
  double rinv2 = rinv*rinv;

  double exp2zir = exp(-2.0*zei*r);
  double zei2 = zei*zei;
  double zei4 = zei2*zei2;
  double zei6 = zei2*zei4;

  double exp2zjr = exp(-2.0*zej*r);
  double zej2 = zej*zej;
  double zej4 = zej2*zej2;
  double zej6 = zej2*zej4;

  double sm1 = 11.0/8.0;
  double sm2 = 3.00/4.0;
  double sm3 = 1.00/6.0;
  double e1, e2, e3, e4;

  double rc = cut_coul;
  double rcinv = 1.0/rc;
  double rcinv2 = rcinv*rcinv;
  double exp2zirsh = exp(-2.0*zei*rc);
  double exp2zjrsh = exp(-2.0*zej*rc);
  double eshift, fshift;

  e1 = e2 = e3 = e4 = 0.0;

  eshift = -zei*exp2zirsh - rcinv*exp2zirsh;
  fshift = 2.0*zei2*exp2zirsh + rcinv2*exp2zirsh + 2.0*zei*rcinv*exp2zirsh;

  ci_jfi = -zei*exp2zir - rinv*exp2zir - eshift - (r-rc)*fshift;
  dci_jfi = 2.0*zei2*exp2zir + rinv2*exp2zir + 2.0*zei*rinv*exp2zir - fshift;

  if (zei == zej) {

    eshift = -exp2zirsh*(rcinv + zei*(sm1 + sm2*zei*rc + sm3*zei2*rc*rc));
    fshift =  exp2zirsh*(rcinv2 + 2.0*zei*rcinv + zei2*
                (2.0 + 7.0/6.0*zei*rc + 1.0/3.0*zei2*rc*rc));

    ci_fifj = -exp2zir*(rinv + zei*(sm1 + sm2*zei*r + sm3*zei2*r*r))
              - eshift - (r-rc)*fshift;
    dci_fifj = exp2zir*(rinv2 + 2.0*zei*rinv + zei2*
                (2.0 + 7.0/6.0*zei*r + 1.0/3.0*zei2*r*r)) - fshift;

  } else {

    e1 = zei*zej4/((zei+zej)*(zei+zej)*(zei-zej)*(zei-zej));
    e2 = zej*zei4/((zei+zej)*(zei+zej)*(zej-zei)*(zej-zei));
    e3 = (3.0*zei2*zej4-zej6) /
         ((zei+zej)*(zei+zej)*(zei+zej)*(zei-zej)*(zei-zej)*(zei-zej));
    e4 = (3.0*zej2*zei4-zei6) /
         ((zei+zej)*(zei+zej)*(zei+zej)*(zej-zei)*(zej-zei)*(zej-zei));

    eshift = -exp2zirsh*(e1+e3/rc) - exp2zjrsh*(e2+e4/rc);
    fshift = (exp2zirsh*(2.0*zei*(e1+e3/rc) + e3*rcinv2)
            + exp2zjrsh*(2.0*zej*(e2+e4/rc) + e4*rcinv2));

    ci_fifj = -exp2zir*(e1+e3/r) - exp2zjr*(e2+e4/r)
              - eshift - (r-rc)*fshift;
    dci_fifj = (exp2zir*(2.0*zei*(e1+e3/r) + e3*rinv2) +
                exp2zjr*(2.0*zej*(e2+e4/r) + e4*rinv2)) - fshift;
  }
}

/* ---------------------------------------------------------------------- */

void PairCoulStreitz::wolf_sum(double qi, double qj, double zj, double r,
                double ci_jfi, double dci_jfi, double ci_fifj,
                double dci_fifj, double &etmp, double &ftmp)
{
  double a = g_wolf;
  double rc = cut_coul;
  double qqrd2e = force->qqrd2e;

  double erfcr = erfc(a*r);
  double derfcr = exp(-a*a*r*r);
  double erfcrc = erfc(a*rc);

  double etmp1, etmp2, etmp3;
  double ftmp1, ftmp2, ftmp3;

  etmp = etmp1 = etmp2 = etmp3 = 0.0;
  ftmp = ftmp1 = ftmp2 = ftmp3 = 0.0;

  etmp1 = erfcr/r - erfcrc/rc;
  etmp2 = qi * zj * (ci_jfi - ci_fifj);
  etmp3 = qi * qj * 0.50 * (etmp1 + ci_fifj);

  ftmp1 = -erfcr/r/r - 2.0*a/MY_PIS*derfcr/r - dwoself;
  ftmp2 = qi * zj * (dci_jfi - dci_fifj);
  ftmp3 = qi * qj * 0.50 * (ftmp1 + dci_fifj);

  etmp = qqrd2e * (etmp2 + etmp3);
  ftmp = qqrd2e * (ftmp2 + ftmp3);

}

/* ---------------------------------------------------------------------- */

void PairCoulStreitz::coulomb_integral_ewald(double zei, double zej, double r,
                  double &ci_jfi, double &dci_jfi, double &ci_fifj,
                  double &dci_fifj)
{
  double rinv = 1.0/r;
  double rinv2 = rinv*rinv;

  double exp2zir = exp(-2.0*zei*r);
  double zei2 = zei*zei;
  double zei4 = zei2*zei2;
  double zei6 = zei2*zei4;

  double exp2zjr = exp(-2.0*zej*r);
  double zej2 = zej*zej;
  double zej4 = zej2*zej2;
  double zej6 = zej2*zej4;

  double sm1 = 11.0/8.0;
  double sm2 = 3.00/4.0;
  double sm3 = 1.00/6.0;

  double e1, e2, e3, e4;
  e1 = e2 = e3 = e4 = 0.0;

  ci_jfi = -zei*exp2zir - rinv*exp2zir;
  dci_jfi = 2.0*zei2*exp2zir + rinv2*exp2zir + 2.0*zei*rinv*exp2zir;

  if (zei == zej) {

    ci_fifj = -exp2zir*(rinv + zei*(sm1 + sm2*zei*r + sm3*zei2*r*r));
    dci_fifj = exp2zir*(rinv2 + 2.0*zei*rinv +
                zei2*(2.0 + 7.0/6.0*zei*r + 1.0/3.0*zei2*r*r));

  } else {

    e1 = zei*zej4/((zei+zej)*(zei+zej)*(zei-zej)*(zei-zej));
    e2 = zej*zei4/((zei+zej)*(zei+zej)*(zej-zei)*(zej-zei));
    e3 = (3.0*zei2*zej4-zej6) /
         ((zei+zej)*(zei+zej)*(zei+zej)*(zei-zej)*(zei-zej)*(zei-zej));
    e4 = (3.0*zej2*zei4-zei6) /
         ((zei+zej)*(zei+zej)*(zei+zej)*(zej-zei)*(zej-zei)*(zej-zei));

    ci_fifj = -exp2zir*(e1+e3/r) - exp2zjr*(e2+e4/r);
    dci_fifj = (exp2zir*(2.0*zei*(e1+e3/r) + e3*rinv2)
              + exp2zjr*(2.0*zej*(e2+e4/r) + e4*rinv2));
  }

}

/* ---------------------------------------------------------------------- */

void PairCoulStreitz::ewald_sum(double qi, double qj, double zj, double r,
                double ci_jfi, double dci_jfi, double ci_fifj,
                double dci_fifj, double &etmp, double &ftmp, double fac)
{
  double etmp1, etmp2, etmp3, etmp4;
  double ftmp1, ftmp2, ftmp3, ftmp4;

  double a = g_ewald;
  double qqrd2e = force->qqrd2e;

  double erfcr = erfc(a*r);
  double derfcr = exp(-a*a*r*r);

  etmp = ftmp = 0.0;
  etmp1 = etmp2 = etmp3 = etmp4 = 0.0;
  ftmp1 = ftmp2 = ftmp3 = ftmp4 = 0.0;

  etmp1 = qi * zj * (ci_jfi - ci_fifj);
  etmp2 = qi * qj * 0.50 * ci_fifj;
  etmp3 = qqrd2e * (etmp1 + etmp2);
  etmp4 = qqrd2e * 0.50*qi*qj/r;

  ftmp1 = qi * zj * (dci_jfi - dci_fifj);
  ftmp2 = qi * qj * 0.50 * dci_fifj;
  ftmp3 = qqrd2e * (ftmp1 + ftmp2);
  ftmp4 = etmp4 * (erfcr + 2.0/MY_PIS*a*r*derfcr);

  etmp = erfcr*etmp4;

  if (fac < 1.0) {
    etmp  -= (1.0-fac)*etmp4;
    ftmp4 -= (1.0-fac)*etmp4;
  }

  etmp += etmp3;
  ftmp = ftmp3 - ftmp4/r;

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairCoulStreitz::memory_usage()
{
  double bytes = (double)maxeatom * sizeof(double);
  bytes += (double)maxvatom*6 * sizeof(double);
  bytes += (double)nmax * sizeof(int);
  bytes += (double)MAXNEIGH * nmax * sizeof(int);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void *PairCoulStreitz::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut_coul") == 0) {
    dim = 0;
    return (void *) &cut_coul;
  }
  if (strcmp(str,"scale") == 0) {
    dim = 2;
    return (void *) scale;
  }
  if (strcmp(str,"chi") == 0 && qeq_x) {
    dim = 1;
    for (int i = 1; i <= atom->ntypes; i++)
      if (map[i] >= 0) qeq_x[i] = params[map[i]].chi;
      else qeq_x[i] = 0.0;
    return (void *) qeq_x;
  }
  if (strcmp(str,"eta") == 0 && qeq_j) {
    dim = 1;
    for (int i = 1; i <= atom->ntypes; i++)
      if (map[i] >= 0) qeq_j[i] = params[map[i]].eta;
      else qeq_j[i] = 0.0;
    return (void *) qeq_j;
  }
  if (strcmp(str,"gamma") == 0 && qeq_g) {
    dim = 1;
    for (int i = 1; i <= atom->ntypes; i++)
      if (map[i] >= 0) qeq_g[i] = params[map[i]].gamma;
      else qeq_g[i] = 0.0;
    return (void *) qeq_g;
  }
  if (strcmp(str,"zeta") == 0 && qeq_z) {
    dim = 1;
    for (int i = 1; i <= atom->ntypes; i++)
      if (map[i] >= 0) qeq_z[i] = params[map[i]].zeta;
      else qeq_z[i] = 0.0;
    return (void *) qeq_z;
  }
  if (strcmp(str,"zcore") == 0 && qeq_c) {
    dim = 1;
    for (int i = 1; i <= atom->ntypes; i++)
      if (map[i] >= 0) qeq_c[i] = params[map[i]].zcore;
      else qeq_c[i] = 0.0;
    return (void *) qeq_c;
  }
  if (strcmp(str,"kspacetype") == 0) {
    dim = 0;
    return (void *) &kspacetype;
  }
  if (strcmp(str,"alpha") == 0) {
    dim = 0;
    if (kspacetype == 1) return (void *) &g_wolf;
    if (kspacetype == 2) return (void *) &g_ewald;
  }
  return nullptr;
}
