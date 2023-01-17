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

/* ------------------------------------------------------------------------
   Contributing authors: Julien Tranchida (SNL)

   Please cite the related publication:
   Bessarab, P. F., Uzdin, V. M., & Jónsson, H. (2015).
   Method for finding mechanism and activation energy of magnetic transitions,
   applied to skyrmion and antivortex annihilation.
   Computer Physics Communications, 196, 335-347.
------------------------------------------------------------------------- */

#include "neb_spin.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "finish.h"
#include "fix.h"
#include "fix_neb_spin.h"
#include "memory.h"
#include "min.h"
#include "modify.h"
#include "output.h"
#include "thermo.h"
#include "timer.h"
#include "universe.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static const char cite_neb_spin[] =
  "neb/spin command:\n\n"
  "@article{bessarab2015method,\n"
  "title={Method for finding mechanism and activation energy of "
  "magnetic transitions, applied to skyrmion and antivortex "
  "annihilation},\n"
  "author={Bessarab, P.F. and Uzdin, V.M. and J{\'o}nsson, H.},\n"
  "journal={Computer Physics Communications},\n"
  "volume={196},\n"
  "pages={335--347},\n"
  "year={2015},\n"
  "publisher={Elsevier}\n"
  "doi={10.1016/j.cpc.2015.07.001}\n"
  "}\n\n";

#define MAXLINE 256
#define CHUNK 1024
// 8 attributes: tag, spin norm, position (3), spin direction (3)
#define ATTRIBUTE_PERLINE 8

/* ---------------------------------------------------------------------- */

NEBSpin::NEBSpin(LAMMPS *lmp) : Command(lmp), fp(nullptr) {
  if (lmp->citeme) lmp->citeme->add(cite_neb_spin);
}

/* ---------------------------------------------------------------------- */

NEBSpin::~NEBSpin()
{
  MPI_Comm_free(&roots);
  memory->destroy(all);
  delete[] rdist;
  if (fp) fclose(fp);
}

/* ----------------------------------------------------------------------
   perform NEBSpin on multiple replicas
------------------------------------------------------------------------- */

void NEBSpin::command(int narg, char **arg)
{
  if (domain->box_exist == 0)
    error->all(FLERR,"NEBSpin command before simulation box is defined");

  if (narg < 6) error->universe_all(FLERR,"Illegal NEBSpin command");

  etol = utils::numeric(FLERR,arg[0],false,lmp);
  ttol = utils::numeric(FLERR,arg[1],false,lmp);
  n1steps = utils::inumeric(FLERR,arg[2],false,lmp);
  n2steps = utils::inumeric(FLERR,arg[3],false,lmp);
  nevery = utils::inumeric(FLERR,arg[4],false,lmp);

  // error checks

  if (etol < 0.0) error->all(FLERR,"Illegal NEBSpin command");
  if (ttol < 0.0) error->all(FLERR,"Illegal NEBSpin command");
  if (nevery <= 0) error->universe_all(FLERR,"Illegal NEBSpin command");
  if (n1steps % nevery || n2steps % nevery)
    error->universe_all(FLERR,"Illegal NEBSpin command");

  // replica info

  nreplica = universe->nworlds;
  ireplica = universe->iworld;
  me_universe = universe->me;
  uworld = universe->uworld;
  MPI_Comm_rank(world,&me);

  // check metal units and spin atom/style

  if (!atom->sp_flag)
    error->all(FLERR,"neb/spin requires atom/spin style");
  if (strcmp(update->unit_style,"metal") != 0)
    error->all(FLERR,"neb/spin simulation requires metal unit style");

  // error checks

  if (nreplica == 1) error->all(FLERR,"Cannot use NEBSpin with a single replica");
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR,"Cannot use NEBSpin unless atom map exists");

  // process file-style setting to setup initial configs for all replicas

  if (strcmp(arg[5],"final") == 0) {
    if (narg != 7 && narg !=8) error->universe_all(FLERR,"Illegal NEBSpin command");
    inpfile = arg[6];
    readfile(inpfile,0);
  } else if (strcmp(arg[5],"each") == 0) {
    if (narg != 7 && narg !=8) error->universe_all(FLERR,"Illegal NEBSpin command");
    inpfile = arg[6];
    readfile(inpfile,1);
  } else if (strcmp(arg[5],"none") == 0) {
    if (narg != 6 && narg !=7) error->universe_all(FLERR,"Illegal NEBSpin command");
  } else error->universe_all(FLERR,"Illegal NEBSpin command");

  verbose=false;
  if (strcmp(arg[narg-1],"verbose") == 0) verbose=true;

  run();
}

/* ----------------------------------------------------------------------
   run NEBSpin on multiple replicas
------------------------------------------------------------------------- */

void NEBSpin::run()
{
  // create MPI communicator for root proc from each world

  int color;
  if (me == 0) color = 0;
  else color = 1;
  MPI_Comm_split(uworld,color,0,&roots);

  // search for neb_spin fix, allocate it

  int ineb;
  for (ineb = 0; ineb < modify->nfix; ineb++)
    if (strcmp(modify->fix[ineb]->style,"neb/spin") == 0) break;
  if (ineb == modify->nfix) error->all(FLERR,"NEBSpin requires use of fix neb/spin");

  fneb = (FixNEBSpin *) modify->fix[ineb];
  if (verbose) numall =7;
  else  numall = 4;
  memory->create(all,nreplica,numall,"neb:all");
  rdist = new double[nreplica];

  // initialize LAMMPS

  update->whichflag = 2;
  update->etol = etol;
  update->ftol = ttol;          // update->ftol is a torque tolerance
  update->multireplica = 1;

  lmp->init();

  // check if correct minimizer is setup

  if (update->minimize->searchflag)
    error->all(FLERR,"NEBSpin requires damped dynamics minimizer");
  if (!utils::strmatch(update->minimize_style,"^spin"))
    error->all(FLERR,"NEBSpin requires a spin minimizer");

  // setup regular NEBSpin minimization

  FILE *uscreen = universe->uscreen;
  FILE *ulogfile = universe->ulogfile;

  if (me_universe == 0 && uscreen)
    fprintf(uscreen,"Setting up regular NEBSpin ...\n");

  update->beginstep = update->firststep = update->ntimestep;
  update->endstep = update->laststep = update->firststep + n1steps;
  update->nsteps = n1steps;
  update->max_eval = n1steps;
  if (update->laststep < 0)
    error->all(FLERR,"Too many timesteps for NEBSpin");

  update->minimize->setup();

  if (me_universe == 0) {
    if (uscreen) {
      if (verbose) {
        fprintf(uscreen,"Step MaxReplicaTorque MaxAtomTorque "
                "GradV0 GradV1 GradVc EBF EBR RDT "
                "RD1 PE1 RD2 PE2 ... RDN PEN "
                "GradV0dottan DN0 ... GradVNdottan DNN\n");
      } else {
        fprintf(uscreen,"Step MaxReplicaTorque MaxAtomTorque "
                "GradV0 GradV1 GradVc EBF EBR RDT RD1 PE1 RD2 PE2 ... "
                "RDN PEN\n");
      }
    }

    if (ulogfile) {
      if (verbose) {
        fprintf(ulogfile,"Step MaxReplicaTorque MaxAtomTorque "
            "GradV0 GradV1 GradVc EBF EBR RDT "
            "RD1 PE1 RD2 PE2 ... RDN PEN "
            "GradV0dottan DN0 ... GradVNdottan DNN\n");
      } else {
        fprintf(ulogfile,"Step MaxReplicaTorque MaxAtomTorque "
                "GradV0 GradV1 GradVc EBF EBR RDT RD1 PE1 RD2 PE2 ... "
                "RDN PEN\n");
      }
    }
  }
  print_status();

  // perform regular NEBSpin for n1steps or until replicas converge
  // retrieve PE values from fix NEBSpin and print every nevery iterations
  // break out of while loop early if converged
  // damped dynamic min styles insure all replicas converge together

  timer->init();
  timer->barrier_start();

  // if (ireplica != 0 && ireplica != nreplica -1)

  while (update->minimize->niter < n1steps) {
    update->minimize->run(nevery);
    print_status();
    if (update->minimize->stop_condition) break;
  }

  timer->barrier_stop();

  update->minimize->cleanup();

  Finish finish(lmp);
  finish.end(1);

  // switch fix NEBSpin to climbing mode
  // top = replica that becomes hill climber

  double vmax = all[0][0];
  int top = 0;
  for (int m = 1; m < nreplica; m++)
    if (vmax < all[m][0]) {
      vmax = all[m][0];
      top = m;
    }

  // setup climbing NEBSpin minimization
  // must reinitialize minimizer so it re-creates its fix MINIMIZE

  if (me_universe == 0 && uscreen)
    fprintf(uscreen,"Setting up climbing ...\n");

  if (me_universe == 0) {
    if (uscreen)
      fprintf(uscreen,"Climbing replica = %d\n",top+1);
    if (ulogfile)
      fprintf(ulogfile,"Climbing replica = %d\n",top+1);
  }

  update->beginstep = update->firststep = update->ntimestep;
  update->endstep = update->laststep = update->firststep + n2steps;
  update->nsteps = n2steps;
  update->max_eval = n2steps;
  if (update->laststep < 0)
    error->all(FLERR,"Too many timesteps");

  update->minimize->init();
  fneb->rclimber = top;
  update->minimize->setup();

  if (me_universe == 0) {
    if (uscreen) {
      if (verbose) {
        fprintf(uscreen,"Step MaxReplicaTorque MaxAtomTorque "
                "GradV0 GradV1 GradVc EBF EBR RDT "
                "RD1 PE1 RD2 PE2 ... RDN PEN "
                "GradV0dottan DN0 ... GradVNdottan DNN\n");
      } else {
        fprintf(uscreen,"Step MaxReplicaTorque MaxAtomTorque "
                "GradV0 GradV1 GradVc "
                "EBF EBR RDT "
                "RD1 PE1 RD2 PE2 ... RDN PEN\n");
      }
    }
    if (ulogfile) {
      if (verbose) {
        fprintf(ulogfile,"Step MaxReplicaTorque MaxAtomTorque "
            "GradV0 GradV1 GradVc EBF EBR RDT "
            "RD1 PE1 RD2 PE2 ... RDN PEN "
            "GradV0dottan DN0 ... GradVNdottan DNN\n");
      } else {
        fprintf(ulogfile,"Step MaxReplicaTorque MaxAtomTorque "
                "GradV0 GradV1 GradVc "
                "EBF EBR RDT "
                "RD1 PE1 RD2 PE2 ... RDN PEN\n");
      }
    }
  }
  print_status();

  // perform climbing NEBSpin for n2steps or until replicas converge
  // retrieve PE values from fix NEBSpin and print every nevery iterations
  // break induced if converged
  // damped dynamic min styles insure all replicas converge together

  timer->init();
  timer->barrier_start();

  while (update->minimize->niter < n2steps) {
    update->minimize->run(nevery);
    print_status();
    if (update->minimize->stop_condition) break;
  }

  timer->barrier_stop();

  update->minimize->cleanup();

  finish.end(1);

  update->whichflag = 0;
  update->multireplica = 0;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;
}

/* ----------------------------------------------------------------------
   read initial config atom coords from file
   flag = 0
   only first replica opens file and reads it
   first replica bcasts lines to all replicas
   final replica stores coords
   intermediate replicas interpolate from coords
   new coord = replica fraction between current and final state
   initial replica does nothing
   flag = 1
   each replica (except first) opens file and reads it
   each replica stores coords
   initial replica does nothing
------------------------------------------------------------------------- */

void NEBSpin::readfile(char *file, int flag)
{
  int i,j,m,nchunk,eofflag,nlines;
  tagint tag;
  char *eof,*start,*next,*buf;
  char line[MAXLINE];
  double xx,yy,zz;
  double musp,spx,spy,spz;

  if (me_universe == 0 && universe->uscreen)
    fprintf(universe->uscreen,"Reading NEBSpin coordinate file(s) ...\n");

  // flag = 0, universe root reads header of file, bcast to universe
  // flag = 1, each replica's root reads header of file, bcast to world
  //   but explicitly skip first replica

  if (flag == 0) {
    if (me_universe == 0) {
      open(file);
      while (1) {
        eof = fgets(line,MAXLINE,fp);
        if (eof == nullptr) error->one(FLERR,"Unexpected end of neb/spin file");
        start = &line[strspn(line," \t\n\v\f\r")];
        if (*start != '\0' && *start != '#') break;
      }
      sscanf(line,"%d",&nlines);
    }
    MPI_Bcast(&nlines,1,MPI_INT,0,uworld);

  } else {
    if (me == 0) {
      if (ireplica) {
        open(file);
        while (1) {
          eof = fgets(line,MAXLINE,fp);
          if (eof == nullptr) error->one(FLERR,"Unexpected end of neb/spin file");
          start = &line[strspn(line," \t\n\v\f\r")];
          if (*start != '\0' && *start != '#') break;
        }
        sscanf(line,"%d",&nlines);
      } else nlines = 0;
    }
    MPI_Bcast(&nlines,1,MPI_INT,0,world);
  }

  char *buffer = new char[CHUNK*MAXLINE];
  char **values = new char*[ATTRIBUTE_PERLINE];

  double fraction = ireplica/(nreplica-1.0);

  double **x = atom->x;
  double **sp = atom->sp;
  double spinit[3],spfinal[3];
  int nlocal = atom->nlocal;

  // loop over chunks of lines read from file
  // two versions of read_lines_from_file() for world vs universe bcast
  // count # of atom coords changed so can check for invalid atom IDs in file

  int ncount = 0;

  int temp_flag,rot_flag;
  temp_flag = rot_flag = 0;
  int nread = 0;
  while (nread < nlines) {
    nchunk = MIN(nlines-nread,CHUNK);
    if (flag == 0)
      eofflag = utils::read_lines_from_file(fp,nchunk,MAXLINE,buffer,
                                            universe->me,universe->uworld);
    else
      eofflag = utils::read_lines_from_file(fp,nchunk,MAXLINE,buffer,me,world);
    if (eofflag) error->all(FLERR,"Unexpected end of neb/spin file");

    buf = buffer;
    next = strchr(buf,'\n');
    *next = '\0';
    int nwords = utils::trim_and_count_words(buf);
    *next = '\n';

    if (nwords != ATTRIBUTE_PERLINE)
      error->all(FLERR,"Incorrect atom format in neb/spin file");

    // loop over lines of atom coords
    // tokenize the line into values

    for (i = 0; i < nchunk; i++) {
      next = strchr(buf,'\n');

      values[0] = strtok(buf," \t\n\r\f");
      for (j = 1; j < nwords; j++)
        values[j] = strtok(nullptr," \t\n\r\f");

      // adjust spin coord based on replica fraction
      // for flag = 0, interpolate for intermediate and final replicas
      // for flag = 1, replace existing coord with new coord
      // ignore image flags of final x
      // for interpolation:
      //   new x is displacement from old x via minimum image convention
      //   if final x is across periodic boundary:
      //     new x may be outside box
      //     will be remapped back into box when simulation starts
      //     its image flags will then be adjusted

      tag = ATOTAGINT(values[0]);
      m = atom->map(tag);
      if (m >= 0 && m < nlocal) {
        ncount++;
        musp = atof(values[1]);
        xx = atof(values[2]);
        yy = atof(values[3]);
        zz = atof(values[4]);
        spx = atof(values[5]);
        spy = atof(values[6]);
        spz = atof(values[7]);

        if (flag == 0) {

          spinit[0] = sp[m][0];
          spinit[1] = sp[m][1];
          spinit[2] = sp[m][2];
          spfinal[0] = spx;
          spfinal[1] = spy;
          spfinal[2] = spz;

          // interpolate intermediate spin states

          sp[m][3] = musp;
          if (fraction == 0.0) {
            sp[m][0] = spinit[0];
            sp[m][1] = spinit[1];
            sp[m][2] = spinit[2];
          } else if (fraction == 1.0) {
            sp[m][0] = spfinal[0];
            sp[m][1] = spfinal[1];
            sp[m][2] = spfinal[2];
          } else {
            temp_flag = initial_rotation(spinit,spfinal,fraction);
            rot_flag = MAX(temp_flag,rot_flag);
            sp[m][0] = spfinal[0];
            sp[m][1] = spfinal[1];
            sp[m][2] = spfinal[2];
          }
        } else {
          sp[m][3] = musp;
          x[m][0] = xx;
          x[m][1] = yy;
          x[m][2] = zz;
          sp[m][0] = spx;
          sp[m][1] = spy;
          sp[m][2] = spz;
        }
      }

      buf = next + 1;
    }

    nread += nchunk;
  }

  // warning message if one or more couples (spi,spf) were aligned
  // this breaks Rodrigues' formula, and an arbitrary rotation
  // vector has to be chosen

  if ((rot_flag > 0) && (comm->me == 0))
    error->warning(FLERR,"arbitrary initial rotation of one or more spin(s)");

  // check that all atom IDs in file were found by a proc

  if (flag == 0) {
    int ntotal;
    MPI_Allreduce(&ncount,&ntotal,1,MPI_INT,MPI_SUM,uworld);
    if (ntotal != nreplica*nlines)
      error->universe_all(FLERR,"Invalid atom IDs in neb/spin file");
  } else {
    int ntotal;
    MPI_Allreduce(&ncount,&ntotal,1,MPI_INT,MPI_SUM,world);
    if (ntotal != nlines)
      error->all(FLERR,"Invalid atom IDs in neb/spin file");
  }

  // clean up

  delete[] buffer;
  delete[] values;

  if (flag == 0) {
    if (me_universe == 0) {
      if (compressed) pclose(fp);
      else fclose(fp);
    }
  } else {
    if (me == 0 && ireplica) {
      if (compressed) pclose(fp);
      else fclose(fp);
    }
  }
  fp = nullptr;
}

/* ----------------------------------------------------------------------
   initial configuration of intermediate spins using Rodrigues' formula
   interpolates between initial (spi) and final (stored in sploc)
------------------------------------------------------------------------- */

int NEBSpin::initial_rotation(double *spi, double *sploc, double fraction)
{

  // no interpolation for initial and final replica

  if (fraction == 0.0 || fraction == 1.0) return 0;

  int rot_flag = 0;
  double kx,ky,kz;
  double spix,spiy,spiz,spfx,spfy,spfz;
  double kcrossx,kcrossy,kcrossz,knormsq;
  double kdots;
  double spkx,spky,spkz;
  double sidotsf,omega,iknorm,isnorm;

  spix = spi[0];
  spiy = spi[1];
  spiz = spi[2];

  spfx = sploc[0];
  spfy = sploc[1];
  spfz = sploc[2];

  kx = spiy*spfz - spiz*spfy;
  ky = spiz*spfx - spix*spfz;
  kz = spix*spfy - spiy*spfx;

  knormsq = kx*kx+ky*ky+kz*kz;
  sidotsf = spix*spfx + spiy*spfy + spiz*spfz;

  // if knormsq == 0.0, init and final spins are aligned
  // Rodrigues' formula breaks, needs to define another axis k

  if (knormsq == 0.0) {
    if (sidotsf > 0.0) {        // spins aligned and in same direction
      return 0;
    } else if (sidotsf < 0.0) { // spins aligned and in opposite directions

      // defining a rotation axis
      // first guess, k = spi x [100]
      // second guess, k = spi x [010]

      if (spiy*spiy + spiz*spiz != 0.0) { // spin not along [100]
        kx = 0.0;
        ky = spiz;
        kz = -spiy;
        knormsq = ky*ky + kz*kz;
      } else if (spix*spix + spiz*spiz != 0.0) { // spin not along [010]
        kx = -spiz;
        ky = 0.0;
        kz = spix;
        knormsq = kx*kx + kz*kz;
      } else error->all(FLERR,"Incorrect initial rotation operation");
      rot_flag = 1;
    }
  }

  // knormsq should not be 0

  if (knormsq == 0.0)
    error->all(FLERR,"Incorrect initial rotation operation");

  // normalize k vector

  iknorm = 1.0/sqrt(knormsq);
  kx *= iknorm;
  ky *= iknorm;
  kz *= iknorm;

  // calc. k x spi and total rotation angle

  kcrossx = ky*spiz - kz*spiy;
  kcrossy = kz*spix - kx*spiz;
  kcrossz = kx*spiy - ky*spix;

  kdots = kx*spix + ky*spiy + kz*spiz;

  omega = acos(sidotsf);
  omega *= fraction;

  // apply Rodrigues' formula

  spkx = spix*cos(omega);
  spky = spiy*cos(omega);
  spkz = spiz*cos(omega);

  spkx += kcrossx*sin(omega);
  spky += kcrossy*sin(omega);
  spkz += kcrossz*sin(omega);

  spkx += kx*kdots*(1.0-cos(omega));
  spky += ky*kdots*(1.0-cos(omega));
  spkz += kz*kdots*(1.0-cos(omega));

  // normalizing resulting spin vector

  isnorm = 1.0/sqrt(spkx*spkx+spky*spky+spkz*spkz);
  if (isnorm == 0.0)
    error->all(FLERR,"Incorrect initial rotation operation");

  spkx *= isnorm;
  spky *= isnorm;
  spkz *= isnorm;

  // returns rotated spin

  sploc[0] = spkx;
  sploc[1] = spky;
  sploc[2] = spkz;

  return rot_flag;
}

/* ----------------------------------------------------------------------
   universe proc 0 opens NEBSpin data file
   test if gzipped
------------------------------------------------------------------------- */

void NEBSpin::open(char *file)
{
  compressed = 0;
  char *suffix = file + strlen(file) - 3;
  if (suffix > file && strcmp(suffix,".gz") == 0) compressed = 1;
  if (!compressed) fp = fopen(file,"r");
  else {
#ifdef LAMMPS_GZIP
    auto gunzip = std::string("gzip -c -d ") + file;
#ifdef _WIN32
    fp = _popen(gunzip.c_str(),"rb");
#else
    fp = popen(gunzip.c_str(),"r");
#endif

#else
    error->one(FLERR,"Cannot open gzipped file");
#endif
  }

  if (fp == nullptr)
    error->one(FLERR,"Cannot open file {}: {}",file,utils::getsyserror());
}

/* ----------------------------------------------------------------------
   query fix NEBSpin for info on each replica
   universe proc 0 prints current NEBSpin status
------------------------------------------------------------------------- */

void NEBSpin::print_status()
{
  int nlocal = atom->nlocal;
  double tx,ty,tz;
  double tnorm2,local_norm_inf,temp_inf;
  double **sp = atom->sp;
  double **fm = atom->fm;

  // calc. magnetic torques

  tnorm2 = local_norm_inf = temp_inf = 0.0;
  for (int i = 0; i < nlocal; i++) {
    tx = (fm[i][1]*sp[i][2] - fm[i][2]*sp[i][1]);
    ty = (fm[i][2]*sp[i][0] - fm[i][0]*sp[i][2]);
    tz = (fm[i][0]*sp[i][1] - fm[i][1]*sp[i][0]);
    tnorm2 += tx*tx + ty*ty + tz*tz;
    temp_inf = MAX(fabs(tx),fabs(ty));
    temp_inf = MAX(fabs(tz),temp_inf);
    local_norm_inf = MAX(temp_inf,local_norm_inf);
  }

  double fmaxreplica;
  MPI_Allreduce(&tnorm2,&fmaxreplica,1,MPI_DOUBLE,MPI_MAX,roots);

  double fnorminf = 0.0;
  MPI_Allreduce(&local_norm_inf,&fnorminf,1,MPI_DOUBLE,MPI_MAX,world);
  double fmaxatom;
  MPI_Allreduce(&fnorminf,&fmaxatom,1,MPI_DOUBLE,MPI_MAX,roots);

  if (verbose) {
    freplica = new double[nreplica];
    MPI_Allgather(&tnorm2,1,MPI_DOUBLE,&freplica[0],1,MPI_DOUBLE,roots);
    fmaxatomInRepl = new double[nreplica];
    MPI_Allgather(&fnorminf,1,MPI_DOUBLE,&fmaxatomInRepl[0],1,MPI_DOUBLE,roots);
  }

  double one[7];
  one[0] = fneb->veng;
  one[1] = fneb->plen;
  one[2] = fneb->nlen;
  one[3] = fneb->gradlen;

  if (verbose) {
    one[4] = fneb->dotpath;
    one[5] = fneb->dottangrad;
    one[6] = fneb->dotgrad;
  }

  if (output->thermo->normflag) one[0] /= atom->natoms;
  if (me == 0)
    MPI_Allgather(one,numall,MPI_DOUBLE,&all[0][0],numall,MPI_DOUBLE,roots);
  MPI_Bcast(&all[0][0],numall*nreplica,MPI_DOUBLE,0,world);

  rdist[0] = 0.0;
  for (int i = 1; i < nreplica; i++)
    rdist[i] = rdist[i-1] + all[i][1];
  double endpt = rdist[nreplica-1] = rdist[nreplica-2] + all[nreplica-2][2];
  for (int i = 1; i < nreplica; i++)
    rdist[i] /= endpt;

  // look up GradV for the initial, final, and climbing replicas
  // these are identical to fnorm2, but to be safe we
  // take them straight from fix_neb

  double gradvnorm0, gradvnorm1, gradvnormc;

  int irep;
  irep = 0;
  gradvnorm0 = all[irep][3];
  irep = nreplica-1;
  gradvnorm1 = all[irep][3];
  irep = fneb->rclimber;
  if (irep > -1) {
    gradvnormc = all[irep][3];
    ebf = all[irep][0]-all[0][0];
    ebr = all[irep][0]-all[nreplica-1][0];
  } else {
    double vmax = all[0][0];
    int top = 0;
    for (int m = 1; m < nreplica; m++)
      if (vmax < all[m][0]) {
        vmax = all[m][0];
        top = m;
      }
    irep = top;
    gradvnormc = all[irep][3];
    ebf = all[irep][0]-all[0][0];
    ebr = all[irep][0]-all[nreplica-1][0];
  }

  if (me_universe == 0) {
    FILE *uscreen = universe->uscreen;
    FILE *ulogfile = universe->ulogfile;
    if (uscreen) {
      fprintf(uscreen,BIGINT_FORMAT " %12.8g %12.8g ",
              update->ntimestep,fmaxreplica,fmaxatom);
      fprintf(uscreen,"%12.8g %12.8g %12.8g ",
              gradvnorm0,gradvnorm1,gradvnormc);
      fprintf(uscreen,"%12.8g %12.8g %12.8g ",ebf,ebr,endpt);
      for (int i = 0; i < nreplica; i++)
        fprintf(uscreen,"%12.8g %12.8g ",rdist[i],all[i][0]);
      if (verbose) {
        for (int i = 0; i < nreplica-1; i++)
          fprintf(uscreen,"%12.8g %12.8g ",all[i][2],all[i][5]);
        fprintf(uscreen,"%12.8g %12.8g ",NAN,all[nreplica-1][5]);
      }
      fprintf(uscreen,"\n");
    }

    if (ulogfile) {
      fprintf(ulogfile,BIGINT_FORMAT " %12.8g %12.8g ",
              update->ntimestep,fmaxreplica,fmaxatom);
      fprintf(ulogfile,"%12.8g %12.8g %12.8g ",
              gradvnorm0,gradvnorm1,gradvnormc);
      fprintf(ulogfile,"%12.8g %12.8g %12.8g ",ebf,ebr,endpt);
      for (int i = 0; i < nreplica; i++)
        fprintf(ulogfile,"%12.8g %12.8g ",rdist[i],all[i][0]);
      if (verbose) {
        for (int i = 0; i < nreplica-1; i++)
          fprintf(ulogfile,"%12.8g %12.8g ",all[i][2],all[i][5]);
        fprintf(ulogfile,"%12.8g %12.8g ",NAN,all[nreplica-1][5]);
      }
      fprintf(ulogfile,"\n");
      fflush(ulogfile);
    }
  }
}
