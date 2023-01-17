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
   Contributing author: Mike Brown (SNL)
------------------------------------------------------------------------- */

#include "pair_gayberne.h"

#include <cmath>
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "citeme.h"
#include "memory.h"
#include "error.h"


using namespace LAMMPS_NS;

static const char cite_pair_gayberne[] =
  "pair gayberne command:\n\n"
  "@Article{Brown09,\n"
  " author =  {W. M. Brown, M. K. Petersen, S. J. Plimpton, and G. S. Grest},\n"
  " title =   {Liquid crystal nanodroplets in solution},\n"
  " journal = {J.~Chem.~Phys.},\n"
  " year =    2009,\n"
  " volume =  130,\n"
  " pages =   {044901}\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

PairGayBerne::PairGayBerne(LAMMPS *lmp) : Pair(lmp)
{
  if (lmp->citeme) lmp->citeme->add(cite_pair_gayberne);

  single_enable = 0;
  writedata = 1;
}

/* ----------------------------------------------------------------------
   free all arrays
------------------------------------------------------------------------- */

PairGayBerne::~PairGayBerne()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(form);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(shape1);
    memory->destroy(shape2);
    memory->destroy(well);
    memory->destroy(cut);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
    delete [] lshape;
    delete [] setwell;
  }
}

/* ---------------------------------------------------------------------- */

void PairGayBerne::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double evdwl,one_eng,rsq,r2inv,r6inv,forcelj,factor_lj;
  double fforce[3],ttor[3],rtor[3],r12[3];
  double a1[3][3],b1[3][3],g1[3][3],a2[3][3],b2[3][3],g2[3][3],temp[3][3];
  int *ilist,*jlist,*numneigh,**firstneigh;
  double *iquat,*jquat;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **x = atom->x;
  double **f = atom->f;
  double **tor = atom->torque;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];

    if (form[itype][itype] == ELLIPSE_ELLIPSE) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat_trans(iquat,a1);
      MathExtra::diag_times3(well[itype],a1,temp);
      MathExtra::transpose_times3(a1,temp,b1);
      MathExtra::diag_times3(shape2[itype],a1,temp);
      MathExtra::transpose_times3(a1,temp,g1);
    }

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      // r12 = center to center vector

      r12[0] = x[j][0]-x[i][0];
      r12[1] = x[j][1]-x[i][1];
      r12[2] = x[j][2]-x[i][2];
      rsq = MathExtra::dot3(r12,r12);
      jtype = type[j];

      // compute if less than cutoff

      if (rsq < cutsq[itype][jtype]) {

        switch (form[itype][jtype]) {
        case SPHERE_SPHERE:
          r2inv = 1.0/rsq;
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
          forcelj *= -r2inv;
          if (eflag) one_eng =
                       r6inv*(r6inv*lj3[itype][jtype]-lj4[itype][jtype]) -
                       offset[itype][jtype];
          fforce[0] = r12[0]*forcelj;
          fforce[1] = r12[1]*forcelj;
          fforce[2] = r12[2]*forcelj;
          ttor[0] = ttor[1] = ttor[2] = 0.0;
          rtor[0] = rtor[1] = rtor[2] = 0.0;
          break;

        case SPHERE_ELLIPSE:
          jquat = bonus[ellipsoid[j]].quat;
          MathExtra::quat_to_mat_trans(jquat,a2);
          MathExtra::diag_times3(well[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,b2);
          MathExtra::diag_times3(shape2[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,g2);
          one_eng = gayberne_lj(j,i,a2,b2,g2,r12,rsq,fforce,rtor);
          ttor[0] = ttor[1] = ttor[2] = 0.0;
          break;

        case ELLIPSE_SPHERE:
          one_eng = gayberne_lj(i,j,a1,b1,g1,r12,rsq,fforce,ttor);
          rtor[0] = rtor[1] = rtor[2] = 0.0;
          break;

        default:
          jquat = bonus[ellipsoid[j]].quat;
          MathExtra::quat_to_mat_trans(jquat,a2);
          MathExtra::diag_times3(well[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,b2);
          MathExtra::diag_times3(shape2[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,g2);
          one_eng = gayberne_analytic(i,j,a1,a2,b1,b2,g1,g2,r12,rsq,
                                      fforce,ttor,rtor);
          break;
        }

        fforce[0] *= factor_lj;
        fforce[1] *= factor_lj;
        fforce[2] *= factor_lj;
        ttor[0] *= factor_lj;
        ttor[1] *= factor_lj;
        ttor[2] *= factor_lj;

        f[i][0] += fforce[0];
        f[i][1] += fforce[1];
        f[i][2] += fforce[2];
        tor[i][0] += ttor[0];
        tor[i][1] += ttor[1];
        tor[i][2] += ttor[2];

        if (newton_pair || j < nlocal) {
          rtor[0] *= factor_lj;
          rtor[1] *= factor_lj;
          rtor[2] *= factor_lj;
          f[j][0] -= fforce[0];
          f[j][1] -= fforce[1];
          f[j][2] -= fforce[2];
          tor[j][0] += rtor[0];
          tor[j][1] += rtor[1];
          tor[j][2] += rtor[2];
        }

        if (eflag) evdwl = factor_lj*one_eng;

        if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
                                 evdwl,0.0,fforce[0],fforce[1],fforce[2],
                                 -r12[0],-r12[1],-r12[2]);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGayBerne::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(form,n+1,n+1,"pair:form");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(shape1,n+1,3,"pair:shape1");
  memory->create(shape2,n+1,3,"pair:shape2");
  memory->create(well,n+1,3,"pair:well");
  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");

  lshape = new double[n+1];
  setwell = new int[n+1];
  for (int i = 1; i <= n; i++) setwell[i] = 0;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGayBerne::settings(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR,"Illegal pair_style command");

  gamma = utils::numeric(FLERR,arg[0],false,lmp);
  upsilon = utils::numeric(FLERR,arg[1],false,lmp)/2.0;
  mu = utils::numeric(FLERR,arg[2],false,lmp);
  cut_global = utils::numeric(FLERR,arg[3],false,lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairGayBerne::coeff(int narg, char **arg)
{
  if (narg < 10 || narg > 11)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double epsilon_one = utils::numeric(FLERR,arg[2],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[3],false,lmp);
  double eia_one = utils::numeric(FLERR,arg[4],false,lmp);
  double eib_one = utils::numeric(FLERR,arg[5],false,lmp);
  double eic_one = utils::numeric(FLERR,arg[6],false,lmp);
  double eja_one = utils::numeric(FLERR,arg[7],false,lmp);
  double ejb_one = utils::numeric(FLERR,arg[8],false,lmp);
  double ejc_one = utils::numeric(FLERR,arg[9],false,lmp);

  double cut_one = cut_global;
  if (narg == 11) cut_one = utils::numeric(FLERR,arg[10],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      if (eia_one != 0.0 || eib_one != 0.0 || eic_one != 0.0) {
        well[i][0] = pow(eia_one,-1.0/mu);
        well[i][1] = pow(eib_one,-1.0/mu);
        well[i][2] = pow(eic_one,-1.0/mu);
        if (eia_one == eib_one && eib_one == eic_one) setwell[i] = 2;
        else setwell[i] = 1;
      }
      if (eja_one != 0.0 || ejb_one != 0.0 || ejc_one != 0.0) {
        well[j][0] = pow(eja_one,-1.0/mu);
        well[j][1] = pow(ejb_one,-1.0/mu);
        well[j][2] = pow(ejc_one,-1.0/mu);
        if (eja_one == ejb_one && ejb_one == ejc_one) setwell[j] = 2;
        else setwell[j] = 1;
      }
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairGayBerne::init_style()
{
  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec) error->all(FLERR,"Pair gayberne requires atom style ellipsoid");

  neighbor->request(this,instance_me);

  // per-type shape precalculations
  // require that atom shapes are identical within each type
  // if shape = 0 for point particle, set shape = 1 as required by Gay-Berne

  for (int i = 1; i <= atom->ntypes; i++) {
    if (!atom->shape_consistency(i,shape1[i][0],shape1[i][1],shape1[i][2]))
      error->all(FLERR,
                 "Pair gayberne requires atoms with same type have same shape");
    if (shape1[i][0] == 0.0)
      shape1[i][0] = shape1[i][1] = shape1[i][2] = 1.0;
    shape2[i][0] = shape1[i][0]*shape1[i][0];
    shape2[i][1] = shape1[i][1]*shape1[i][1];
    shape2[i][2] = shape1[i][2]*shape1[i][2];
    lshape[i] = (shape1[i][0]*shape1[i][1]+shape1[i][2]*shape1[i][2]) *
      sqrt(shape1[i][0]*shape1[i][1]);
  }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGayBerne::init_one(int i, int j)
{
  if (setwell[i] == 0 || setwell[j] == 0)
    error->all(FLERR,"Pair gayberne epsilon a,b,c coeffs are not all set");

  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag && (cut[i][j] > 0.0)) {
    double ratio = sigma[i][j] / cut[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  int ishape = 0;
  if (shape1[i][0] != shape1[i][1] ||
      shape1[i][0] != shape1[i][2] ||
      shape1[i][1] != shape1[i][2]) ishape = 1;
  if (setwell[i] == 1) ishape = 1;
  int jshape = 0;
  if (shape1[j][0] != shape1[j][1] ||
      shape1[j][0] != shape1[j][2] ||
      shape1[j][1] != shape1[j][2]) jshape = 1;
  if (setwell[j] == 1) jshape = 1;

  if (ishape == 0 && jshape == 0)
    form[i][i] = form[j][j] = form[i][j] = form[j][i] = SPHERE_SPHERE;
  else if (ishape == 0) {
    form[i][i] = SPHERE_SPHERE; form[j][j] = ELLIPSE_ELLIPSE;
    form[i][j] = SPHERE_ELLIPSE; form[j][i] = ELLIPSE_SPHERE;
  } else if (jshape == 0) {
    form[j][j] = SPHERE_SPHERE; form[i][i] = ELLIPSE_ELLIPSE;
    form[j][i] = SPHERE_ELLIPSE; form[i][j] = ELLIPSE_SPHERE;
  } else
    form[i][i] = form[j][j] = form[i][j] = form[j][i] = ELLIPSE_ELLIPSE;

  epsilon[j][i] = epsilon[i][j];
  sigma[j][i] = sigma[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGayBerne::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++) {
    fwrite(&setwell[i],sizeof(int),1,fp);
    if (setwell[i]) fwrite(&well[i][0],sizeof(double),3,fp);
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGayBerne::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++) {
    if (me == 0) utils::sfread(FLERR,&setwell[i],sizeof(int),1,fp,nullptr,error);
    MPI_Bcast(&setwell[i],1,MPI_INT,0,world);
    if (setwell[i]) {
      if (me == 0) utils::sfread(FLERR,&well[i][0],sizeof(double),3,fp,nullptr,error);
      MPI_Bcast(&well[i][0],3,MPI_DOUBLE,0,world);
    }
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&epsilon[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigma[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGayBerne::write_restart_settings(FILE *fp)
{
  fwrite(&gamma,sizeof(double),1,fp);
  fwrite(&upsilon,sizeof(double),1,fp);
  fwrite(&mu,sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGayBerne::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR,&gamma,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&upsilon,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&mu,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&gamma,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&upsilon,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mu,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairGayBerne::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %g %g %g\n",i,
            epsilon[i][i],sigma[i][i],
            pow(well[i][0],-mu),pow(well[i][1],-mu),pow(well[i][2],-mu),
            pow(well[i][0],-mu),pow(well[i][1],-mu),pow(well[i][2],-mu));
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairGayBerne::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g %g %g %g %g\n",i,j,
              epsilon[i][i],sigma[i][i],
              pow(well[i][0],-mu),pow(well[i][1],-mu),pow(well[i][2],-mu),
              pow(well[j][0],-mu),pow(well[j][1],-mu),pow(well[j][2],-mu),
              cut[i][j]);
}

/* ----------------------------------------------------------------------
   compute analytic energy, force (fforce), and torque (ttor & rtor)
   based on rotation matrices a and precomputed matrices b and g
   if newton is off, rtor is not calculated for ghost atoms
------------------------------------------------------------------------- */

double PairGayBerne::gayberne_analytic(const int i,const int j,double a1[3][3],
                                       double a2[3][3], double b1[3][3],
                                       double b2[3][3], double g1[3][3],
                                       double g2[3][3], double *r12,
                                       const double rsq, double *fforce,
                                       double *ttor, double *rtor)
{
  double tempv[3], tempv2[3];
  double temp[3][3];
  double temp1,temp2,temp3;

  int *type = atom->type;
  int newton_pair = force->newton_pair;
  int nlocal = atom->nlocal;

  double r12hat[3];
  MathExtra::normalize3(r12,r12hat);
  double r = sqrt(rsq);

  // compute distance of closest approach

  double g12[3][3];
  MathExtra::plus3(g1,g2,g12);
  double kappa[3];
  int ierror = MathExtra::mldivide3(g12,r12,kappa);
  if (ierror) error->all(FLERR,"Bad matrix inversion in mldivide3");

  // tempv = G12^-1*r12hat

  tempv[0] = kappa[0]/r;
  tempv[1] = kappa[1]/r;
  tempv[2] = kappa[2]/r;
  double sigma12 = MathExtra::dot3(r12hat,tempv);
  sigma12 = pow(0.5*sigma12,-0.5);
  double h12 = r-sigma12;

  // energy
  // compute u_r

  double varrho = sigma[type[i]][type[j]]/(h12+gamma*sigma[type[i]][type[j]]);
  double varrho6 = pow(varrho,6.0);
  double varrho12 = varrho6*varrho6;
  double u_r = 4.0*epsilon[type[i]][type[j]]*(varrho12-varrho6);

  // compute eta_12

  double eta = 2.0*lshape[type[i]]*lshape[type[j]];
  double det_g12 = MathExtra::det3(g12);
  eta = pow(eta/det_g12,upsilon);

  // compute chi_12

  double b12[3][3];
  double iota[3];
  MathExtra::plus3(b1,b2,b12);
  ierror = MathExtra::mldivide3(b12,r12,iota);
  if (ierror) error->all(FLERR,"Bad matrix inversion in mldivide3");

  // tempv = G12^-1*r12hat

  tempv[0] = iota[0]/r;
  tempv[1] = iota[1]/r;
  tempv[2] = iota[2]/r;
  double chi = MathExtra::dot3(r12hat,tempv);
  chi = pow(chi*2.0,mu);

  // force
  // compute dUr/dr

  temp1 = (2.0*varrho12*varrho-varrho6*varrho)/sigma[type[i]][type[j]];
  temp1 = temp1*24.0*epsilon[type[i]][type[j]];
  double u_slj = temp1*pow(sigma12,3.0)/2.0;
  double dUr[3];
  temp2 = MathExtra::dot3(kappa,r12hat);
  double uslj_rsq = u_slj/rsq;
  dUr[0] = temp1*r12hat[0]+uslj_rsq*(kappa[0]-temp2*r12hat[0]);
  dUr[1] = temp1*r12hat[1]+uslj_rsq*(kappa[1]-temp2*r12hat[1]);
  dUr[2] = temp1*r12hat[2]+uslj_rsq*(kappa[2]-temp2*r12hat[2]);

  // compute dChi_12/dr

  double dchi[3];
  temp1 = MathExtra::dot3(iota,r12hat);
  temp2 = -4.0/rsq*mu*pow(chi,(mu-1.0)/mu);
  dchi[0] = temp2*(iota[0]-temp1*r12hat[0]);
  dchi[1] = temp2*(iota[1]-temp1*r12hat[1]);
  dchi[2] = temp2*(iota[2]-temp1*r12hat[2]);

  temp1 = -eta*u_r;
  temp3 = eta*chi;
  fforce[0] = temp1*dchi[0]-temp3*dUr[0];
  fforce[1] = temp1*dchi[1]-temp3*dUr[1];
  fforce[2] = temp1*dchi[2]-temp3*dUr[2];

  // torque for particle 1 and 2
  // compute dUr

  tempv[0] = -uslj_rsq*kappa[0];
  tempv[1] = -uslj_rsq*kappa[1];
  tempv[2] = -uslj_rsq*kappa[2];
  MathExtra::vecmat(kappa,g1,tempv2);
  MathExtra::cross3(tempv,tempv2,dUr);
  double dUr2[3];

  if (newton_pair || j < nlocal) {
    MathExtra::vecmat(kappa,g2,tempv2);
    MathExtra::cross3(tempv,tempv2,dUr2);
  }

  // compute d_chi

  MathExtra::vecmat(iota,b1,tempv);
  MathExtra::cross3(tempv,iota,dchi);
  dchi[0] *= temp2;
  dchi[1] *= temp2;
  dchi[2] *= temp2;
  double dchi2[3];

  if (newton_pair || j < nlocal) {
    MathExtra::vecmat(iota,b2,tempv);
    MathExtra::cross3(tempv,iota,dchi2);
    dchi2[0] *= temp2;
    dchi2[1] *= temp2;
    dchi2[2] *= temp2;
  }

  // compute d_eta

  double deta[3];
  deta[0] = deta[1] = deta[2] = 0.0;
  compute_eta_torque(g12,a1,shape2[type[i]],temp);
  temp1 = -eta*upsilon;
  for (int m = 0; m < 3; m++) {
    for (int y = 0; y < 3; y++) tempv[y] = temp1*temp[m][y];
    MathExtra::cross3(a1[m],tempv,tempv2);
    deta[0] += tempv2[0];
    deta[1] += tempv2[1];
    deta[2] += tempv2[2];
  }

  // compute d_eta for particle 2

  double deta2[3];
  if (newton_pair || j < nlocal) {
    deta2[0] = deta2[1] = deta2[2] = 0.0;
    compute_eta_torque(g12,a2,shape2[type[j]],temp);
    for (int m = 0; m < 3; m++) {
      for (int y = 0; y < 3; y++) tempv[y] = temp1*temp[m][y];
      MathExtra::cross3(a2[m],tempv,tempv2);
      deta2[0] += tempv2[0];
      deta2[1] += tempv2[1];
      deta2[2] += tempv2[2];
    }
  }

  // torque

  temp1 = u_r*eta;
  temp2 = u_r*chi;
  temp3 = chi*eta;

  ttor[0] = (temp1*dchi[0]+temp2*deta[0]+temp3*dUr[0]) * -1.0;
  ttor[1] = (temp1*dchi[1]+temp2*deta[1]+temp3*dUr[1]) * -1.0;
  ttor[2] = (temp1*dchi[2]+temp2*deta[2]+temp3*dUr[2]) * -1.0;

  if (newton_pair || j < nlocal) {
    rtor[0] = (temp1*dchi2[0]+temp2*deta2[0]+temp3*dUr2[0]) * -1.0;
    rtor[1] = (temp1*dchi2[1]+temp2*deta2[1]+temp3*dUr2[1]) * -1.0;
    rtor[2] = (temp1*dchi2[2]+temp2*deta2[2]+temp3*dUr2[2]) * -1.0;
  }

  return temp1*chi;
}

/* ----------------------------------------------------------------------
   compute analytic energy, force (fforce), and torque (ttor)
   between ellipsoid and lj particle
------------------------------------------------------------------------- */

double PairGayBerne::gayberne_lj(const int i,const int j,double a1[3][3],
                                 double b1[3][3],double g1[3][3],
                                 double *r12,const double rsq,double *fforce,
                                 double *ttor)
{
  double tempv[3], tempv2[3];
  double temp[3][3];
  double temp1,temp2,temp3;

  int *type = atom->type;

  double r12hat[3];
  MathExtra::normalize3(r12,r12hat);
  double r = sqrt(rsq);

  // compute distance of closest approach

  double g12[3][3];
  g12[0][0] = g1[0][0]+shape2[type[j]][0];
  g12[1][1] = g1[1][1]+shape2[type[j]][0];
  g12[2][2] = g1[2][2]+shape2[type[j]][0];
  g12[0][1] = g1[0][1]; g12[1][0] = g1[1][0];
  g12[0][2] = g1[0][2]; g12[2][0] = g1[2][0];
  g12[1][2] = g1[1][2]; g12[2][1] = g1[2][1];
  double kappa[3];
  int ierror = MathExtra::mldivide3(g12,r12,kappa);
  if (ierror) error->all(FLERR,"Bad matrix inversion in mldivide3");

  // tempv = G12^-1*r12hat

  tempv[0] = kappa[0]/r;
  tempv[1] = kappa[1]/r;
  tempv[2] = kappa[2]/r;
  double sigma12 = MathExtra::dot3(r12hat,tempv);
  sigma12 = pow(0.5*sigma12,-0.5);
  double h12 = r-sigma12;

  // energy
  // compute u_r

  double varrho = sigma[type[i]][type[j]]/(h12+gamma*sigma[type[i]][type[j]]);
  double varrho6 = pow(varrho,6.0);
  double varrho12 = varrho6*varrho6;
  double u_r = 4.0*epsilon[type[i]][type[j]]*(varrho12-varrho6);

  // compute eta_12

  double eta = 2.0*lshape[type[i]]*lshape[type[j]];
  double det_g12 = MathExtra::det3(g12);
  eta = pow(eta/det_g12,upsilon);

  // compute chi_12

  double b12[3][3];
  double iota[3];
  b12[0][0] = b1[0][0] + well[type[j]][0];
  b12[1][1] = b1[1][1] + well[type[j]][0];
  b12[2][2] = b1[2][2] + well[type[j]][0];
  b12[0][1] = b1[0][1]; b12[1][0] = b1[1][0];
  b12[0][2] = b1[0][2]; b12[2][0] = b1[2][0];
  b12[1][2] = b1[1][2]; b12[2][1] = b1[2][1];
  ierror = MathExtra::mldivide3(b12,r12,iota);
  if (ierror) error->all(FLERR,"Bad matrix inversion in mldivide3");

  // tempv = G12^-1*r12hat

  tempv[0] = iota[0]/r;
  tempv[1] = iota[1]/r;
  tempv[2] = iota[2]/r;
  double chi = MathExtra::dot3(r12hat,tempv);
  chi = pow(chi*2.0,mu);

  // force
  // compute dUr/dr

  temp1 = (2.0*varrho12*varrho-varrho6*varrho)/sigma[type[i]][type[j]];
  temp1 = temp1*24.0*epsilon[type[i]][type[j]];
  double u_slj = temp1*pow(sigma12,3.0)/2.0;
  double dUr[3];
  temp2 = MathExtra::dot3(kappa,r12hat);
  double uslj_rsq = u_slj/rsq;
  dUr[0] = temp1*r12hat[0]+uslj_rsq*(kappa[0]-temp2*r12hat[0]);
  dUr[1] = temp1*r12hat[1]+uslj_rsq*(kappa[1]-temp2*r12hat[1]);
  dUr[2] = temp1*r12hat[2]+uslj_rsq*(kappa[2]-temp2*r12hat[2]);

  // compute dChi_12/dr

  double dchi[3];
  temp1 = MathExtra::dot3(iota,r12hat);
  temp2 = -4.0/rsq*mu*pow(chi,(mu-1.0)/mu);
  dchi[0] = temp2*(iota[0]-temp1*r12hat[0]);
  dchi[1] = temp2*(iota[1]-temp1*r12hat[1]);
  dchi[2] = temp2*(iota[2]-temp1*r12hat[2]);

  temp1 = -eta*u_r;
  temp2 = eta*chi;
  fforce[0] = temp1*dchi[0]-temp2*dUr[0];
  fforce[1] = temp1*dchi[1]-temp2*dUr[1];
  fforce[2] = temp1*dchi[2]-temp2*dUr[2];

  // torque for particle 1 and 2
  // compute dUr

  tempv[0] = -uslj_rsq*kappa[0];
  tempv[1] = -uslj_rsq*kappa[1];
  tempv[2] = -uslj_rsq*kappa[2];
  MathExtra::vecmat(kappa,g1,tempv2);
  MathExtra::cross3(tempv,tempv2,dUr);

  // compute d_chi

  MathExtra::vecmat(iota,b1,tempv);
  MathExtra::cross3(tempv,iota,dchi);
  temp1 = -4.0/rsq;
  dchi[0] *= temp1;
  dchi[1] *= temp1;
  dchi[2] *= temp1;

  // compute d_eta

  double deta[3];
  deta[0] = deta[1] = deta[2] = 0.0;
  compute_eta_torque(g12,a1,shape2[type[i]],temp);
  temp1 = -eta*upsilon;
  for (int m = 0; m < 3; m++) {
    for (int y = 0; y < 3; y++) tempv[y] = temp1*temp[m][y];
    MathExtra::cross3(a1[m],tempv,tempv2);
    deta[0] += tempv2[0];
    deta[1] += tempv2[1];
    deta[2] += tempv2[2];
  }

  // torque

  temp1 = u_r*eta;
  temp2 = u_r*chi;
  temp3 = chi*eta;

  ttor[0] = (temp1*dchi[0]+temp2*deta[0]+temp3*dUr[0]) * -1.0;
  ttor[1] = (temp1*dchi[1]+temp2*deta[1]+temp3*dUr[1]) * -1.0;
  ttor[2] = (temp1*dchi[2]+temp2*deta[2]+temp3*dUr[2]) * -1.0;

  return temp1*chi;
}

/* ----------------------------------------------------------------------
   torque contribution from eta
   computes trace in the last doc equation for the torque derivative
   code comes from symbolic solver dump
   m is g12, m2 is a_i, s is the shape for the particle
------------------------------------------------------------------------- */

void PairGayBerne::compute_eta_torque(double m[3][3], double m2[3][3],
                                      double *s, double ans[3][3])
{
  double den = m[1][0]*m[0][2]*m[2][1]-m[0][0]*m[1][2]*m[2][1]-
    m[0][2]*m[2][0]*m[1][1]+m[0][1]*m[2][0]*m[1][2]-
    m[1][0]*m[0][1]*m[2][2]+m[0][0]*m[1][1]*m[2][2];

  ans[0][0] = s[0]*(m[1][2]*m[0][1]*m2[0][2]+2.0*m[1][1]*m[2][2]*m2[0][0]-
                    m[1][1]*m2[0][2]*m[0][2]-2.0*m[1][2]*m2[0][0]*m[2][1]+
                    m2[0][1]*m[0][2]*m[2][1]-m2[0][1]*m[0][1]*m[2][2]-
                    m[1][0]*m[2][2]*m2[0][1]+m[2][0]*m[1][2]*m2[0][1]+
                    m[1][0]*m2[0][2]*m[2][1]-m2[0][2]*m[2][0]*m[1][1])/den;

  ans[0][1] = s[0]*(m[0][2]*m2[0][0]*m[2][1]-m[2][2]*m2[0][0]*m[0][1]+
                    2.0*m[0][0]*m[2][2]*m2[0][1]-m[0][0]*m2[0][2]*m[1][2]-
                    2.0*m[2][0]*m[0][2]*m2[0][1]+m2[0][2]*m[1][0]*m[0][2]-
                    m[2][2]*m[1][0]*m2[0][0]+m[2][0]*m2[0][0]*m[1][2]+
                    m[2][0]*m2[0][2]*m[0][1]-m2[0][2]*m[0][0]*m[2][1])/den;

  ans[0][2] = s[0]*(m[0][1]*m[1][2]*m2[0][0]-m[0][2]*m2[0][0]*m[1][1]-
                    m[0][0]*m[1][2]*m2[0][1]+m[1][0]*m[0][2]*m2[0][1]-
                    m2[0][1]*m[0][0]*m[2][1]-m[2][0]*m[1][1]*m2[0][0]+
                    2.0*m[1][1]*m[0][0]*m2[0][2]-2.0*m[1][0]*m2[0][2]*m[0][1]+
                    m[1][0]*m[2][1]*m2[0][0]+m[2][0]*m2[0][1]*m[0][1])/den;

  ans[1][0] = s[1]*(-m[1][1]*m2[1][2]*m[0][2]+2.0*m[1][1]*m[2][2]*m2[1][0]+
                    m[1][2]*m[0][1]*m2[1][2]-2.0*m[1][2]*m2[1][0]*m[2][1]+
                    m2[1][1]*m[0][2]*m[2][1]-m2[1][1]*m[0][1]*m[2][2]-
                    m[1][0]*m[2][2]*m2[1][1]+m[2][0]*m[1][2]*m2[1][1]-
                    m2[1][2]*m[2][0]*m[1][1]+m[1][0]*m2[1][2]*m[2][1])/den;

  ans[1][1] = s[1]*(m[0][2]*m2[1][0]*m[2][1]-m[0][1]*m[2][2]*m2[1][0]+
                    2.0*m[2][2]*m[0][0]*m2[1][1]-m2[1][2]*m[0][0]*m[1][2]-
                    2.0*m[2][0]*m2[1][1]*m[0][2]-m[1][0]*m[2][2]*m2[1][0]+
                    m[2][0]*m[1][2]*m2[1][0]+m[1][0]*m2[1][2]*m[0][2]-
                    m[0][0]*m2[1][2]*m[2][1]+m2[1][2]*m[0][1]*m[2][0])/den;

  ans[1][2] = s[1]*(m[0][1]*m[1][2]*m2[1][0]-m[0][2]*m2[1][0]*m[1][1]-
                    m[0][0]*m[1][2]*m2[1][1]+m[1][0]*m[0][2]*m2[1][1]+
                    2.0*m[1][1]*m[0][0]*m2[1][2]-m[0][0]*m2[1][1]*m[2][1]+
                    m[0][1]*m[2][0]*m2[1][1]-m2[1][0]*m[2][0]*m[1][1]-
                    2.0*m[1][0]*m[0][1]*m2[1][2]+m[1][0]*m2[1][0]*m[2][1])/den;

  ans[2][0] = s[2]*(-m[1][1]*m[0][2]*m2[2][2]+m[0][1]*m[1][2]*m2[2][2]+
                    2.0*m[1][1]*m2[2][0]*m[2][2]-m[0][1]*m2[2][1]*m[2][2]+
                    m[0][2]*m[2][1]*m2[2][1]-2.0*m2[2][0]*m[2][1]*m[1][2]-
                    m[1][0]*m2[2][1]*m[2][2]+m[1][2]*m[2][0]*m2[2][1]-
                    m[1][1]*m[2][0]*m2[2][2]+m[2][1]*m[1][0]*m2[2][2])/den;

  ans[2][1] = s[2]*-(m[0][1]*m[2][2]*m2[2][0]-m[0][2]*m2[2][0]*m[2][1]-
                     2.0*m2[2][1]*m[0][0]*m[2][2]+m[1][2]*m2[2][2]*m[0][0]+
                     2.0*m2[2][1]*m[0][2]*m[2][0]+m[1][0]*m2[2][0]*m[2][2]-
                     m[1][0]*m[0][2]*m2[2][2]-m[1][2]*m[2][0]*m2[2][0]+
                     m[0][0]*m2[2][2]*m[2][1]-m2[2][2]*m[0][1]*m[2][0])/den;

  ans[2][2] = s[2]*(m[0][1]*m[1][2]*m2[2][0]-m[0][2]*m2[2][0]*m[1][1]-
                    m[0][0]*m[1][2]*m2[2][1]+m[1][0]*m[0][2]*m2[2][1]-
                    m[1][1]*m[2][0]*m2[2][0]-m[2][1]*m2[2][1]*m[0][0]+
                    2.0*m[1][1]*m2[2][2]*m[0][0]+m[2][1]*m[1][0]*m2[2][0]+
                    m[2][0]*m[0][1]*m2[2][1]-2.0*m2[2][2]*m[1][0]*m[0][1])/den;
}
