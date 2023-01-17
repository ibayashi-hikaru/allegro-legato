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
   Contributing author: Chuanfu Luo (luochuanfu@gmail.com)
------------------------------------------------------------------------- */

#include "angle_table.h"

#include <cmath>

#include <cstring>
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

#include "tokenizer.h"
#include "table_file_reader.h"


using namespace LAMMPS_NS;
using namespace MathConst;

enum{LINEAR,SPLINE};

#define SMALL 0.001
#define TINY  1.E-10

/* ---------------------------------------------------------------------- */

AngleTable::AngleTable(LAMMPS *lmp) : Angle(lmp)
{
  writedata = 0;
  ntables = 0;
  tables = nullptr;
}

/* ---------------------------------------------------------------------- */

AngleTable::~AngleTable()
{
  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(theta0);
    memory->destroy(tabindex);
  }
}

/* ---------------------------------------------------------------------- */

void AngleTable::compute(int eflag, int vflag)
{
  int i1,i2,i3,n,type;
  double eangle,f1[3],f3[3];
  double delx1,dely1,delz1,delx2,dely2,delz2;
  double rsq1,rsq2,r1,r2,c,s,a,a11,a12,a22;
  double theta,u,mdu; //mdu: minus du, -du/dx=f

  eangle = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];

    // 1st bond

    delx1 = x[i1][0] - x[i2][0];
    dely1 = x[i1][1] - x[i2][1];
    delz1 = x[i1][2] - x[i2][2];

    rsq1 = delx1*delx1 + dely1*dely1 + delz1*delz1;
    r1 = sqrt(rsq1);

    // 2nd bond

    delx2 = x[i3][0] - x[i2][0];
    dely2 = x[i3][1] - x[i2][1];
    delz2 = x[i3][2] - x[i2][2];

    rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
    r2 = sqrt(rsq2);

    // angle (cos and sin)

    c = delx1*delx2 + dely1*dely2 + delz1*delz2;
    c /= r1*r2;

    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    s = sqrt(1.0 - c*c);
    if (s < SMALL) s = SMALL;
    s = 1.0/s;

    // tabulated force & energy

    theta = acos(c);
    uf_lookup(type,theta,u,mdu);

    if (eflag) eangle = u;

    a = mdu * s;
    a11 = a*c / rsq1;
    a12 = -a / (r1*r2);
    a22 = a*c / rsq2;

    f1[0] = a11*delx1 + a12*delx2;
    f1[1] = a11*dely1 + a12*dely2;
    f1[2] = a11*delz1 + a12*delz2;
    f3[0] = a22*delx2 + a12*delx1;
    f3[1] = a22*dely2 + a12*dely1;
    f3[2] = a22*delz2 + a12*delz1;

    // apply force to each of 3 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= f1[0] + f3[0];
      f[i2][1] -= f1[1] + f3[1];
      f[i2][2] -= f1[2] + f3[2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }

    if (evflag) ev_tally(i1,i2,i3,nlocal,newton_bond,eangle,f1,f3,
                         delx1,dely1,delz1,delx2,dely2,delz2);
  }
}

/* ---------------------------------------------------------------------- */

void AngleTable::allocate()
{
  allocated = 1;
  int n = atom->nangletypes;

  memory->create(theta0,n+1,"angle:theta0");
  memory->create(tabindex,n+1,"angle:tabindex");

  memory->create(setflag,n+1,"angle:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void AngleTable::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Illegal angle_style command");

  if (strcmp(arg[0],"linear") == 0) tabstyle = LINEAR;
  else if (strcmp(arg[0],"spline") == 0) tabstyle = SPLINE;
  else error->all(FLERR,"Unknown table style in angle style table");

  tablength = utils::inumeric(FLERR,arg[1],false,lmp);
  if (tablength < 2) error->all(FLERR,"Illegal number of angle table entries");

  // delete old tables, since cannot just change settings

  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);

  if (allocated) {
     memory->destroy(setflag);
     memory->destroy(tabindex);
  }
  allocated = 0;

  ntables = 0;
  tables = nullptr;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void AngleTable::coeff(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Illegal angle_coeff command");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nangletypes,ilo,ihi,error);

  int me;
  MPI_Comm_rank(world,&me);
  tables = (Table *)
    memory->srealloc(tables,(ntables+1)*sizeof(Table),"angle:tables");
  Table *tb = &tables[ntables];
  null_table(tb);
  if (me == 0) read_table(tb,arg[1],arg[2]);
  bcast_table(tb);

  // error check on table parameters

  if (tb->ninput <= 1) error->one(FLERR,"Invalid angle table length");

  double alo,ahi;
  alo = tb->afile[0];
  ahi = tb->afile[tb->ninput-1];
  if (fabs(alo-0.0) > TINY || fabs(ahi-180.0) > TINY)
    error->all(FLERR,"Angle table must range from 0 to 180 degrees");

  // convert theta from degrees to radians

  for (int i = 0; i < tb->ninput; i++) {
    tb->afile[i] *= MY_PI/180.0;
    tb->ffile[i] *= 180.0/MY_PI;
  }

  // spline read-in and compute a,e,f vectors within table

  spline_table(tb);
  compute_table(tb);

  // store ptr to table in tabindex

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    tabindex[i] = ntables;
    setflag[i] = 1;
    theta0[i] = tb->theta0;
    count++;
  }
  ntables++;

  if (count == 0) error->all(FLERR,"Illegal angle_coeff command");
}

/* ----------------------------------------------------------------------
   return an equilbrium angle length
   should not be used, since don't know minimum of tabulated function
------------------------------------------------------------------------- */

double AngleTable::equilibrium_angle(int i)
{
  return theta0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void AngleTable::write_restart(FILE *fp)
{
  write_restart_settings(fp);
}

/* ----------------------------------------------------------------------
    proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void AngleTable::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void AngleTable::write_restart_settings(FILE *fp)
{
  fwrite(&tabstyle,sizeof(int),1,fp);
  fwrite(&tablength,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
    proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void AngleTable::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&tabstyle,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&tablength,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&tabstyle,1,MPI_INT,0,world);
  MPI_Bcast(&tablength,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double AngleTable::single(int type, int i1, int i2, int i3)
{
  double **x = atom->x;

  double delx1 = x[i1][0] - x[i2][0];
  double dely1 = x[i1][1] - x[i2][1];
  double delz1 = x[i1][2] - x[i2][2];
  domain->minimum_image(delx1,dely1,delz1);
  double r1 = sqrt(delx1*delx1 + dely1*dely1 + delz1*delz1);

  double delx2 = x[i3][0] - x[i2][0];
  double dely2 = x[i3][1] - x[i2][1];
  double delz2 = x[i3][2] - x[i2][2];
  domain->minimum_image(delx2,dely2,delz2);
  double r2 = sqrt(delx2*delx2 + dely2*dely2 + delz2*delz2);

  double c = delx1*delx2 + dely1*dely2 + delz1*delz2;
  c /= r1*r2;
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;

  double theta = acos(c);
  double u=0.0;
  u_lookup(type,theta,u);
  return u;
}

/* ---------------------------------------------------------------------- */

void AngleTable::null_table(Table *tb)
{
  tb->afile = tb->efile = tb->ffile = nullptr;
  tb->e2file = tb->f2file = nullptr;
  tb->ang = tb->e = tb->de = nullptr;
  tb->f = tb->df = tb->e2 = tb->f2 = nullptr;
}

/* ---------------------------------------------------------------------- */

void AngleTable::free_table(Table *tb)
{
  memory->destroy(tb->afile);
  memory->destroy(tb->efile);
  memory->destroy(tb->ffile);
  memory->destroy(tb->e2file);
  memory->destroy(tb->f2file);

  memory->destroy(tb->ang);
  memory->destroy(tb->e);
  memory->destroy(tb->de);
  memory->destroy(tb->f);
  memory->destroy(tb->df);
  memory->destroy(tb->e2);
  memory->destroy(tb->f2);
}

/* ----------------------------------------------------------------------
   read table file, only called by proc 0
------------------------------------------------------------------------- */

void AngleTable::read_table(Table *tb, char *file, char *keyword)
{
  TableFileReader reader(lmp, file, "angle");

  char * line = reader.find_section_start(keyword);

  if (!line) {
    error->one(FLERR,"Did not find keyword in table file");
  }

  // read args on 2nd line of section
  // allocate table arrays for file values

  line = reader.next_line();
  param_extract(tb, line);
  memory->create(tb->afile, tb->ninput, "angle:afile");
  memory->create(tb->efile, tb->ninput, "angle:efile");
  memory->create(tb->ffile, tb->ninput, "angle:ffile");

  // read a,e,f table values from file

  int cerror = 0;
  reader.skip_line();
  for (int i = 0; i < tb->ninput; i++) {
    line = reader.next_line(4);
    try {
      ValueTokenizer values(line);
      values.next_int();
      tb->afile[i] = values.next_double();
      tb->efile[i] = values.next_double();
      tb->ffile[i] = values.next_double();
    } catch (TokenizerException &e) {
      ++cerror;
    }
  }

  // warn if data was read incompletely, e.g. columns were missing

  if (cerror) {
    std::string str = fmt::format("{} of {} lines in table were incomplete or could not be parsed completely", cerror, tb->ninput);
    error->warning(FLERR,str.c_str());
  }
}

/* ----------------------------------------------------------------------
   build spline representation of e,f over entire range of read-in table
   this function sets these values in e2file,f2file
------------------------------------------------------------------------- */

void AngleTable::spline_table(Table *tb)
{
  memory->create(tb->e2file,tb->ninput,"angle:e2file");
  memory->create(tb->f2file,tb->ninput,"angle:f2file");

  double ep0 = - tb->ffile[0];
  double epn = - tb->ffile[tb->ninput-1];
  spline(tb->afile,tb->efile,tb->ninput,ep0,epn,tb->e2file);

  if (tb->fpflag == 0) {
    tb->fplo = (tb->ffile[1] - tb->ffile[0]) / (tb->afile[1] - tb->afile[0]);
    tb->fphi = (tb->ffile[tb->ninput-1] - tb->ffile[tb->ninput-2]) /
      (tb->afile[tb->ninput-1] - tb->afile[tb->ninput-2]);
  }

  double fp0 = tb->fplo;
  double fpn = tb->fphi;
  spline(tb->afile,tb->ffile,tb->ninput,fp0,fpn,tb->f2file);
}

/* ----------------------------------------------------------------------
   compute a,e,f vectors from splined values
------------------------------------------------------------------------- */

void AngleTable::compute_table(Table *tb)
{
  // delta = table spacing in angle for N-1 bins

  int tlm1 = tablength-1;
  tb->delta = MY_PI / tlm1;
  tb->invdelta = 1.0/tb->delta;
  tb->deltasq6 = tb->delta*tb->delta / 6.0;

  // N-1 evenly spaced bins in angle from 0 to PI
  // ang,e,f = value at lower edge of bin
  // de,df values = delta values of e,f
  // ang,e,f are N in length so de,df arrays can compute difference

  memory->create(tb->ang,tablength,"angle:ang");
  memory->create(tb->e,tablength,"angle:e");
  memory->create(tb->de,tablength,"angle:de");
  memory->create(tb->f,tablength,"angle:f");
  memory->create(tb->df,tablength,"angle:df");
  memory->create(tb->e2,tablength,"angle:e2");
  memory->create(tb->f2,tablength,"angle:f2");

  double a;
  for (int i = 0; i < tablength; i++) {
    a = i*tb->delta;
    tb->ang[i] = a;
          tb->e[i] = splint(tb->afile,tb->efile,tb->e2file,tb->ninput,a);
          tb->f[i] = splint(tb->afile,tb->ffile,tb->f2file,tb->ninput,a);
  }

  for (int i = 0; i < tlm1; i++) {
    tb->de[i] = tb->e[i+1] - tb->e[i];
    tb->df[i] = tb->f[i+1] - tb->f[i];
  }
  // get final elements from linear extrapolation
  tb->de[tlm1] = 2.0*tb->de[tlm1-1] - tb->de[tlm1-2];
  tb->df[tlm1] = 2.0*tb->df[tlm1-1] - tb->df[tlm1-2];

  double ep0 = - tb->f[0];
  double epn = - tb->f[tlm1];
  spline(tb->ang,tb->e,tablength,ep0,epn,tb->e2);
  spline(tb->ang,tb->f,tablength,tb->fplo,tb->fphi,tb->f2);
}

/* ----------------------------------------------------------------------
   extract attributes from parameter line in table section
   format of line: N value FP fplo fphi EQ theta0
   N is required, other params are optional
------------------------------------------------------------------------- */

void AngleTable::param_extract(Table *tb, char *line)
{
  tb->ninput = 0;
  tb->fpflag = 0;
  tb->theta0 = MY_PI;

  try {
    ValueTokenizer values(line);

    while (values.has_next()) {
      std::string word = values.next_string();

      if (word == "N") {
        tb->ninput = values.next_int();
      } else if (word == "FP") {
        tb->fpflag = 1;
        tb->fplo = values.next_double();
        tb->fphi = values.next_double();
        tb->fplo *= (180.0/MY_PI)*(180.0/MY_PI);
        tb->fphi *= (180.0/MY_PI)*(180.0/MY_PI);
      } else if (word == "EQ") {
        tb->theta0 = values.next_double()/180.0*MY_PI;
      } else {
        error->one(FLERR,"Invalid keyword in angle table parameters");
      }
    }
  } catch(TokenizerException &e) {
    error->one(FLERR, e.what());
  }

  if (tb->ninput == 0) error->one(FLERR,"Angle table parameters did not set N");
}

/* ----------------------------------------------------------------------
   broadcast read-in table info from proc 0 to other procs
   this function communicates these values in Table:
     ninput,afile,efile,ffile,fpflag,fplo,fphi,theta0
------------------------------------------------------------------------- */

void AngleTable::bcast_table(Table *tb)
{
  MPI_Bcast(&tb->ninput,1,MPI_INT,0,world);

  int me;
  MPI_Comm_rank(world,&me);
  if (me > 0) {
    memory->create(tb->afile,tb->ninput,"angle:afile");
    memory->create(tb->efile,tb->ninput,"angle:efile");
    memory->create(tb->ffile,tb->ninput,"angle:ffile");
  }

  MPI_Bcast(tb->afile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->efile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->ffile,tb->ninput,MPI_DOUBLE,0,world);

  MPI_Bcast(&tb->fpflag,1,MPI_INT,0,world);
  if (tb->fpflag) {
    MPI_Bcast(&tb->fplo,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&tb->fphi,1,MPI_DOUBLE,0,world);
  }
  MPI_Bcast(&tb->theta0,1,MPI_DOUBLE,0,world);
}

/* ----------------------------------------------------------------------
   spline and splint routines modified from Numerical Recipes
------------------------------------------------------------------------- */

void AngleTable::spline(double *x, double *y, int n,
                       double yp1, double ypn, double *y2)
{
  int i,k;
  double p,qn,sig,un;
  double *u = new double[n];

  if (yp1 > 0.99e30) y2[0] = u[0] = 0.0;
  else {
    y2[0] = -0.5;
    u[0] = (3.0/(x[1]-x[0])) * ((y[1]-y[0]) / (x[1]-x[0]) - yp1);
  }
  for (i = 1; i < n-1; i++) {
    sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);
    p = sig*y2[i-1] + 2.0;
    y2[i] = (sig-1.0) / p;
    u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1]);
    u[i] = (6.0*u[i] / (x[i+1]-x[i-1]) - sig*u[i-1]) / p;
  }
  if (ypn > 0.99e30) qn = un = 0.0;
  else {
    qn = 0.5;
    un = (3.0/(x[n-1]-x[n-2])) * (ypn - (y[n-1]-y[n-2]) / (x[n-1]-x[n-2]));
  }
  y2[n-1] = (un-qn*u[n-2]) / (qn*y2[n-2] + 1.0);
  for (k = n-2; k >= 0; k--) y2[k] = y2[k]*y2[k+1] + u[k];

  delete [] u;
}

/* ---------------------------------------------------------------------- */

double AngleTable::splint(double *xa, double *ya, double *y2a, int n, double x)
{
  int klo,khi,k;
  double h,b,a,y;

  klo = 0;
  khi = n-1;
  while (khi-klo > 1) {
    k = (khi+klo) >> 1;
    if (xa[k] > x) khi = k;
    else klo = k;
  }
  h = xa[khi]-xa[klo];
  a = (xa[khi]-x) / h;
  b = (x-xa[klo]) / h;
  y = a*ya[klo] + b*ya[khi] +
    ((a*a*a-a)*y2a[klo] + (b*b*b-b)*y2a[khi]) * (h*h)/6.0;
  return y;
}

/* ----------------------------------------------------------------------
   calculate potential u and force f at angle x
------------------------------------------------------------------------- */

void AngleTable::uf_lookup(int type, double x, double &u, double &f)
{
  if (!std::isfinite(x)) {
    error->one(FLERR,"Illegal angle in angle style table");
  }

  double fraction,a,b;
  const Table *tb = &tables[tabindex[type]];
  int itable = static_cast<int> (x * tb->invdelta);

  if (itable < 0) itable = 0;
  if (itable >= tablength) itable = tablength-1;

  if (tabstyle == LINEAR) {
    fraction = (x - tb->ang[itable]) * tb->invdelta;
    u = tb->e[itable] + fraction*tb->de[itable];
    f = tb->f[itable] + fraction*tb->df[itable];
  } else if (tabstyle == SPLINE) {
    fraction = (x - tb->ang[itable]) * tb->invdelta;

    b = (x - tb->ang[itable]) * tb->invdelta;
    a = 1.0 - b;
    u = a * tb->e[itable] + b * tb->e[itable+1] +
      ((a*a*a-a)*tb->e2[itable] + (b*b*b-b)*tb->e2[itable+1]) *
      tb->deltasq6;
    f = a * tb->f[itable] + b * tb->f[itable+1] +
      ((a*a*a-a)*tb->f2[itable] + (b*b*b-b)*tb->f2[itable+1]) *
      tb->deltasq6;
  }
}

/* ----------------------------------------------------------------------
   calculate potential u at angle x
------------------------------------------------------------------------- */

void AngleTable::u_lookup(int type, double x, double &u)
{
  if (!std::isfinite(x)) {
    error->one(FLERR,"Illegal angle in angle style table");
  }

  double fraction,a,b;
  const Table *tb = &tables[tabindex[type]];
  int itable = static_cast<int> ( x * tb->invdelta);

  if (itable < 0) itable = 0;
  if (itable >= tablength) itable = tablength-1;

  if (tabstyle == LINEAR) {
    fraction = (x - tb->ang[itable]) * tb->invdelta;
    u = tb->e[itable] + fraction*tb->de[itable];
  } else if (tabstyle == SPLINE) {
    fraction = (x - tb->ang[itable]) * tb->invdelta;

    b = (x - tb->ang[itable]) * tb->invdelta;
    a = 1.0 - b;
    u = a * tb->e[itable] + b * tb->e[itable+1] +
      ((a*a*a-a)*tb->e2[itable] + (b*b*b-b)*tb->e2[itable+1]) *
      tb->deltasq6;
  }
}
