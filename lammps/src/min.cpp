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
   Contributing author: Aidan Thompson (SNL)
                        improved CG and backtrack ls, added quadratic ls
   Sources: Numerical Recipes frprmn routine
            "Conjugate Gradient Method Without the Agonizing Pain" by
            JR Shewchuk, https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
------------------------------------------------------------------------- */

#include "min.h"

#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix_minimize.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "pair.h"
#include "thermo.h"
#include "timer.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

Min::Min(LAMMPS *lmp) : Pointers(lmp)
{
  dmax = 0.1;
  searchflag = 0;
  linestyle = 1;
  normstyle = TWO;

  delaystep = 20;
  dtgrow = 1.1;
  dtshrink = 0.5;
  alpha0 = 0.25;
  alphashrink = 0.99;
  tmax = 10.0;
  tmin = 0.02;
  integrator = 0;
  halfstepback_flag = 1;
  delaystep_start_flag = 1;
  max_vdotf_negatif = 2000;
  alpha_final = 0.0;

  elist_global = elist_atom = nullptr;
  vlist_global = vlist_atom = cvlist_atom = nullptr;

  nextra_global = 0;
  fextra = nullptr;

  nextra_atom = 0;
  xextra_atom = fextra_atom = nullptr;
  extra_peratom = extra_nlen = nullptr;
  extra_max = nullptr;
  requestor = nullptr;

  external_force_clear = 0;

  kokkosable = 0;
}

/* ---------------------------------------------------------------------- */

Min::~Min()
{
  delete [] elist_global;
  delete [] elist_atom;
  delete [] vlist_global;
  delete [] vlist_atom;
  delete [] cvlist_atom;

  delete [] fextra;

  memory->sfree(xextra_atom);
  memory->sfree(fextra_atom);
  memory->destroy(extra_peratom);
  memory->destroy(extra_nlen);
  memory->destroy(extra_max);
  memory->sfree(requestor);
}

/* ---------------------------------------------------------------------- */

void Min::init()
{
  if (lmp->kokkos && !kokkosable)
    error->all(FLERR,"Must use a Kokkos-enabled min style "
               "(e.g. min_style cg/kk) with Kokkos minimize");

  // create fix needed for storing atom-based quantities
  // will delete it at end of run

  fix_minimize = (FixMinimize *) modify->add_fix("MINIMIZE all MINIMIZE");

  // clear out extra global and per-atom dof
  // will receive requests for new per-atom dof during pair init()
  // can then add vectors to fix_minimize in setup()

  nextra_global = 0;
  delete [] fextra;
  fextra = nullptr;

  nextra_atom = 0;
  memory->sfree(xextra_atom);
  memory->sfree(fextra_atom);
  memory->destroy(extra_peratom);
  memory->destroy(extra_nlen);
  memory->destroy(extra_max);
  memory->sfree(requestor);
  xextra_atom = fextra_atom = nullptr;
  extra_peratom = extra_nlen = nullptr;
  extra_max = nullptr;
  requestor = nullptr;

  // virial_style:
  // VIRIAL_PAIR if computed explicitly in pair via sum over pair interactions
  // VIRIAL_FDOTR if computed implicitly in pair by
  //   virial_fdotr_compute() via sum over ghosts

  if (force->newton_pair) virial_style = VIRIAL_FDOTR;
  else virial_style = VIRIAL_PAIR;

  // setup lists of computes for global and per-atom PE and pressure

  ev_setup();

  // detect if fix omp is present for clearing force arrays

  int ifix = modify->find_fix("package_omp");
  if (ifix >= 0) external_force_clear = 1;

  // set flags for arrays to clear in force_clear()

  torqueflag = extraflag = 0;
  if (atom->torque_flag) torqueflag = 1;
  if (atom->avec->forceclearflag) extraflag = 1;

  // allow pair and Kspace compute() to be turned off via modify flags

  if (force->pair && force->pair->compute_flag) pair_compute_flag = 1;
  else pair_compute_flag = 0;
  if (force->kspace && force->kspace->compute_flag) kspace_compute_flag = 1;
  else kspace_compute_flag = 0;

  // orthogonal vs triclinic simulation box

  triclinic = domain->triclinic;

  // reset reneighboring criteria if necessary

  neigh_every = neighbor->every;
  neigh_delay = neighbor->delay;
  neigh_dist_check = neighbor->dist_check;

  if (neigh_every != 1 || neigh_delay != 0 || neigh_dist_check != 1) {
    if (comm->me == 0)
      error->warning(FLERR, "Using 'neigh_modify every 1 delay 0 check"
                     " yes' setting during minimization");
  }

  neighbor->every = 1;
  neighbor->delay = 0;
  neighbor->dist_check = 1;

  niter = neval = 0;

  // store timestep size (important for variable timestep minimizer)

  dtinit = update->dt;
}

/* ----------------------------------------------------------------------
   setup before run
------------------------------------------------------------------------- */

void Min::setup(int flag)
{
  if (comm->me == 0 && screen) {
    fmt::print(screen,"Setting up {} style minimization ...\n",
               update->minimize_style);
    if (flag) {
      fmt::print(screen,"  Unit style    : {}\n", update->unit_style);
      fmt::print(screen,"  Current step  : {}\n", update->ntimestep);
      timer->print_timeout(screen);
    }
  }
  update->setupflag = 1;

  // setup extra global dof due to fixes
  // cannot be done in init() b/c update init() is before modify init()

  nextra_global = modify->min_dof();
  if (nextra_global) {
    fextra = new double[nextra_global];
    if (comm->me == 0 && screen)
      fprintf(screen,"WARNING: Energy due to %d extra global DOFs will"
              " be included in minimizer energies\n",nextra_global);
  }

  // compute for potential energy

  int id = modify->find_compute("thermo_pe");
  if (id < 0) error->all(FLERR,"Minimization could not find thermo_pe compute");
  pe_compute = modify->compute[id];

  // style-specific setup does two tasks
  // setup extra global dof vectors
  // setup extra per-atom dof vectors due to requests from Pair classes
  // cannot be done in init() b/c update init() is before modify/pair init()

  setup_style();

  // ndoftotal = total dof for entire minimization problem
  // dof for atoms, extra per-atom, extra global

  bigint ndofme = 3 * static_cast<bigint>(atom->nlocal);
  for (int m = 0; m < nextra_atom; m++)
    ndofme += extra_peratom[m]*static_cast<bigint>(atom->nlocal);
  MPI_Allreduce(&ndofme,&ndoftotal,1,MPI_LMP_BIGINT,MPI_SUM,world);
  ndoftotal += nextra_global;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  atom->setup();
  modify->setup_pre_exchange();
  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  domain->reset_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
  comm->exchange();
  if (atom->sortfreq > 0) atom->sort();
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  domain->image_check();
  domain->box_too_small_check();
  modify->setup_pre_neighbor();
  neighbor->build(1);
  modify->setup_post_neighbor();
  neighbor->ncalls = 0;

  // remove these restriction eventually

  if (searchflag == 0) {
    if (nextra_global)
      error->all(FLERR,
                 "Cannot use a damped dynamics min style with fix box/relax");
    if (nextra_atom)
      error->all(FLERR,
                 "Cannot use a damped dynamics min style with per-atom DOF");
  }

  if (strcmp(update->minimize_style,"hftn") == 0) {
    if (nextra_global)
      error->all(FLERR, "Cannot use hftn min style with fix box/relax");
    if (nextra_atom)
      error->all(FLERR, "Cannot use hftn min style with per-atom DOF");
  }

  // atoms may have migrated in comm->exchange()

  reset_vectors();

  // compute all forces

  force->setup();
  ev_set(update->ntimestep);
  force_clear();
  modify->setup_pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
    else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  // update per-atom minimization variables stored by pair styles

  if (nextra_atom)
    for (int m = 0; m < nextra_atom; m++)
      requestor[m]->min_xf_get(m);

  modify->setup(vflag);
  output->setup(flag);
  update->setupflag = 0;

  // stats for initial thermo output

  ecurrent = pe_compute->compute_scalar();
  if (nextra_global) ecurrent += modify->min_energy(fextra);
  if (output->thermo->normflag) ecurrent /= atom->natoms;

  einitial = ecurrent;
  fnorm2_init = sqrt(fnorm_sqr());
  fnorminf_init = sqrt(fnorm_inf());
}

/* ----------------------------------------------------------------------
   setup without output or one-time post-init setup
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
------------------------------------------------------------------------- */

void Min::setup_minimal(int flag)
{
  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  if (flag) {
    modify->setup_pre_exchange();
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    domain->image_check();
    domain->box_too_small_check();
    modify->setup_pre_neighbor();
    neighbor->build(1);
    modify->setup_post_neighbor();
    neighbor->ncalls = 0;
  }

  // atoms may have migrated in comm->exchange()

  reset_vectors();

  // compute all forces

  ev_set(update->ntimestep);
  force_clear();
  modify->setup_pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
    else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  // update per-atom minimization variables stored by pair styles

  if (nextra_atom)
    for (int m = 0; m < nextra_atom; m++)
      requestor[m]->min_xf_get(m);

  modify->setup(vflag);
  update->setupflag = 0;

  // stats for Finish to print

  ecurrent = pe_compute->compute_scalar();
  if (nextra_global) ecurrent += modify->min_energy(fextra);
  if (output->thermo->normflag) ecurrent /= atom->natoms;

  einitial = ecurrent;
  fnorm2_init = sqrt(fnorm_sqr());
  fnorminf_init = sqrt(fnorm_inf());
}

/* ----------------------------------------------------------------------
   perform minimization, calling iterate() for N steps
------------------------------------------------------------------------- */

void Min::run(int n)
{
  // minimizer iterations

  stop_condition = iterate(n);
  stopstr = stopstrings(stop_condition);

  // if early exit from iterate loop:
  // set update->nsteps to niter for Finish stats to print
  // set output->next values to this timestep
  // call energy_force() to insure vflag is set when forces computed
  // output->write does final output for thermo, dump, restart files
  // add ntimestep to all computes that store invocation times
  //   since are hardwiring call to thermo/dumps and computes may not be ready

  if (stop_condition != MAXITER) {
    update->nsteps = niter;

    if (update->restrict_output == 0) {
      for (int idump = 0; idump < output->ndump; idump++)
        output->next_dump[idump] = update->ntimestep;
      output->next_dump_any = update->ntimestep;
      if (output->restart_flag) {
        output->next_restart = update->ntimestep;
        if (output->restart_every_single)
          output->next_restart_single = update->ntimestep;
        if (output->restart_every_double)
          output->next_restart_double = update->ntimestep;
      }
    }
    output->next_thermo = update->ntimestep;

    modify->addstep_compute_all(update->ntimestep);
    ecurrent = energy_force(0);
    output->write(update->ntimestep);
  }
}

/* ---------------------------------------------------------------------- */

void Min::cleanup()
{
  modify->post_run();

  // stats for Finish to print

  efinal = ecurrent;
  fnorm2_final = sqrt(fnorm_sqr());
  fnorminf_final = sqrt(fnorm_inf());

  // reset reneighboring criteria

  neighbor->every = neigh_every;
  neighbor->delay = neigh_delay;
  neighbor->dist_check = neigh_dist_check;

  // delete fix at end of run, so its atom arrays won't persist

  modify->delete_fix("MINIMIZE");
  domain->box_too_small_check();

  // reset timestep size (important for variable timestep minimizer)

  update->dt = dtinit;
}

/* ----------------------------------------------------------------------
   evaluate potential energy and forces
   may migrate atoms due to reneighboring
   return new energy, which should include nextra_global dof
   return negative gradient stored in atom->f
   return negative gradient for nextra_global dof in fextra
------------------------------------------------------------------------- */

double Min::energy_force(int resetflag)
{
  // check for reneighboring
  // always communicate since minimizer moved atoms

  int nflag = neighbor->decide();

  if (nflag == 0) {
    timer->stamp();
    comm->forward_comm();
    timer->stamp(Timer::COMM);
  } else {
    if (modify->n_min_pre_exchange) {
      timer->stamp();
      modify->min_pre_exchange();
      timer->stamp(Timer::MODIFY);
    }
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    if (domain->box_change) {
      domain->reset_box();
      comm->setup();
      if (neighbor->style) neighbor->setup_bins();
    }
    timer->stamp();
    comm->exchange();
    if (atom->sortfreq > 0 &&
        update->ntimestep >= atom->nextsort) atom->sort();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    timer->stamp(Timer::COMM);
    if (modify->n_min_pre_neighbor) {
      modify->min_pre_neighbor();
      timer->stamp(Timer::MODIFY);
    }
    neighbor->build(1);
    timer->stamp(Timer::NEIGH);
    if (modify->n_min_post_neighbor) {
      modify->min_post_neighbor();
      timer->stamp(Timer::MODIFY);
    }
  }

  ev_set(update->ntimestep);
  force_clear();

  timer->stamp();

  if (modify->n_min_pre_force) {
    modify->min_pre_force(vflag);
    timer->stamp(Timer::MODIFY);
  }

  if (pair_compute_flag) {
    force->pair->compute(eflag,vflag);
    timer->stamp(Timer::PAIR);
  }

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
    timer->stamp(Timer::BOND);
  }

  if (kspace_compute_flag) {
    force->kspace->compute(eflag,vflag);
    timer->stamp(Timer::KSPACE);
  }

  if (modify->n_min_pre_reverse) {
    modify->min_pre_reverse(eflag,vflag);
    timer->stamp(Timer::MODIFY);
  }

  if (force->newton) {
    comm->reverse_comm();
    timer->stamp(Timer::COMM);
  }

  // update per-atom minimization variables stored by pair styles

  if (nextra_atom)
    for (int m = 0; m < nextra_atom; m++)
      requestor[m]->min_xf_get(m);

  // fixes that affect minimization

  if (modify->n_min_post_force) {
     timer->stamp();
     modify->min_post_force(vflag);
     timer->stamp(Timer::MODIFY);
  }

  // compute potential energy of system
  // normalize if thermo PE does

  double energy = pe_compute->compute_scalar();
  if (nextra_global) energy += modify->min_energy(fextra);
  if (output->thermo->normflag) energy /= atom->natoms;

  // if reneighbored, atoms migrated
  // if resetflag = 1, update x0 of atoms crossing PBC
  // reset vectors used by lo-level minimizer

  if (nflag) {
    if (resetflag) fix_minimize->reset_coords();
    reset_vectors();
  }

  return energy;
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
------------------------------------------------------------------------- */

void Min::force_clear()
{
  if (external_force_clear) return;

  // clear global force array
  // if either newton flag is set, also include ghosts

  size_t nbytes = sizeof(double) * atom->nlocal;
  if (force->newton) nbytes += sizeof(double) * atom->nghost;

  if (nbytes) {
    memset(&atom->f[0][0],0,3*nbytes);
    if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
    if (extraflag) atom->avec->force_clear(0,nbytes);
  }
}

/* ----------------------------------------------------------------------
   pair style makes request to add a per-atom variables to minimization
   requestor stores callback to pair class to invoke during min
     to get current variable and forces on it and to update the variable
   return flag that pair can use if it registers multiple variables
------------------------------------------------------------------------- */

int Min::request(Pair *pair, int peratom, double maxvalue)
{
  int n = nextra_atom + 1;
  xextra_atom = (double **) memory->srealloc(xextra_atom,n*sizeof(double *),
                                             "min:xextra_atom");
  fextra_atom = (double **) memory->srealloc(fextra_atom,n*sizeof(double *),
                                             "min:fextra_atom");
  memory->grow(extra_peratom,n,"min:extra_peratom");
  memory->grow(extra_nlen,n,"min:extra_nlen");
  memory->grow(extra_max,n,"min:extra_max");
  requestor = (Pair **) memory->srealloc(requestor,n*sizeof(Pair *),
                                         "min:requestor");

  requestor[nextra_atom] = pair;
  extra_peratom[nextra_atom] = peratom;
  extra_max[nextra_atom] = maxvalue;
  nextra_atom++;
  return nextra_atom-1;
}

/* ---------------------------------------------------------------------- */

void Min::modify_params(int narg, char **arg)
{
  if (narg == 0) error->all(FLERR,"Illegal min_modify command");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"dmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      dmax = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"delaystep") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      delaystep = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"dtgrow") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      dtgrow = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"dtshrink") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      dtshrink = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"alpha0") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      alpha0 = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"alphashrink") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      alphashrink = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"tmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      tmax = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"tmin") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      tmin = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"halfstepback") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      if (strcmp(arg[iarg+1],"yes") == 0) halfstepback_flag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) halfstepback_flag = 0;
      else error->all(FLERR,"Illegal min_modify command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"initialdelay") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      if (strcmp(arg[iarg+1],"yes") == 0) delaystep_start_flag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) delaystep_start_flag = 0;
      else error->all(FLERR,"Illegal min_modify command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"vdfmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      max_vdotf_negatif = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"integrator") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      if (strcmp(arg[iarg+1],"eulerimplicit") == 0) integrator = 0;
      else if (strcmp(arg[iarg+1],"verlet") == 0) integrator = 1;
      else if (strcmp(arg[iarg+1],"leapfrog") == 0) integrator = 2;
      else if (strcmp(arg[iarg+1],"eulerexplicit") == 0) integrator = 3;
      else error->all(FLERR,"Illegal min_modify command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"line") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      if (strcmp(arg[iarg+1],"backtrack") == 0) linestyle = 0;
      else if (strcmp(arg[iarg+1],"quadratic") == 0) linestyle = 1;
      else if (strcmp(arg[iarg+1],"forcezero") == 0) linestyle = 2;
      else if (strcmp(arg[iarg+1],"spin_cubic") == 0) linestyle = 3;
      else if (strcmp(arg[iarg+1],"spin_none") == 0) linestyle = 4;
      else error->all(FLERR,"Illegal min_modify command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"norm") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal min_modify command");
      if (strcmp(arg[iarg+1],"two") == 0) normstyle = TWO;
      else if (strcmp(arg[iarg+1],"max") == 0) normstyle = MAX;
      else if (strcmp(arg[iarg+1],"inf") == 0) normstyle = INF;
      else error->all(FLERR,"Illegal min_modify command");
      iarg += 2;
    } else {
      int n = modify_param(narg-iarg,&arg[iarg]);
      if (n == 0) error->all(FLERR,"Illegal fix_modify command");
      iarg += n;
    }
  }
}

/* ----------------------------------------------------------------------
   setup lists of computes for global and per-atom PE and pressure
------------------------------------------------------------------------- */

void Min::ev_setup()
{
  delete [] elist_global;
  delete [] elist_atom;
  delete [] vlist_global;
  delete [] vlist_atom;
  delete [] cvlist_atom;
  elist_global = elist_atom = nullptr;
  vlist_global = vlist_atom = cvlist_atom = nullptr;

  nelist_global = nelist_atom = 0;
  nvlist_global = nvlist_atom = ncvlist_atom = 0;
  for (int i = 0; i < modify->ncompute; i++) {
    if (modify->compute[i]->peflag) nelist_global++;
    if (modify->compute[i]->peatomflag) nelist_atom++;
    if (modify->compute[i]->pressflag) nvlist_global++;
    if (modify->compute[i]->pressatomflag & 1) nvlist_atom++;
    if (modify->compute[i]->pressatomflag & 2) ncvlist_atom++;
  }

  if (nelist_global) elist_global = new Compute*[nelist_global];
  if (nelist_atom) elist_atom = new Compute*[nelist_atom];
  if (nvlist_global) vlist_global = new Compute*[nvlist_global];
  if (nvlist_atom) vlist_atom = new Compute*[nvlist_atom];
  if (ncvlist_atom) cvlist_atom = new Compute*[ncvlist_atom];

  nelist_global = nelist_atom = 0;
  nvlist_global = nvlist_atom = ncvlist_atom = 0;
  for (int i = 0; i < modify->ncompute; i++) {
    if (modify->compute[i]->peflag)
      elist_global[nelist_global++] = modify->compute[i];
    if (modify->compute[i]->peatomflag)
      elist_atom[nelist_atom++] = modify->compute[i];
    if (modify->compute[i]->pressflag)
      vlist_global[nvlist_global++] = modify->compute[i];
    if (modify->compute[i]->pressatomflag & 1)
      vlist_atom[nvlist_atom++] = modify->compute[i];
    if (modify->compute[i]->pressatomflag & 2)
      cvlist_atom[ncvlist_atom++] = modify->compute[i];
  }
}

/* ----------------------------------------------------------------------
   set eflag,vflag for current iteration
   invoke matchstep() on all timestep-dependent computes to clear their arrays
   eflag/vflag based on computes that need info on this ntimestep
   always set eflag_global = 1, since need energy every iteration
   eflag: set any or no bits
     ENERGY_GLOBAL bit for global energy
     ENERGY_ATOM   bit for per-atom energy
   vflag: set any or no bits, but GLOBAL/FDOTR bit cannot both be set
     VIRIAL_PAIR     bit for global virial as sum of pairwise terms
     VIRIAL_FDOTR    bit for global virial via F dot r
     VIRIAL_ATOM     bit for per-atom virial
     VIRIAL_CENTROID bit for per-atom centroid virial
   all force components (pair,bond,angle,...,kspace) use eflag/vflag
     in their ev_setup() method to set local energy/virial flags
------------------------------------------------------------------------- */

void Min::ev_set(bigint ntimestep)
{
  int i,flag;

  int eflag_global = 1;
  for (i = 0; i < nelist_global; i++)
    elist_global[i]->matchstep(ntimestep);

  flag = 0;
  int eflag_atom = 0;
  for (i = 0; i < nelist_atom; i++)
    if (elist_atom[i]->matchstep(ntimestep)) flag = 1;
  if (flag) eflag_atom = ENERGY_ATOM;

  if (eflag_global) update->eflag_global = update->ntimestep;
  if (eflag_atom) update->eflag_atom = update->ntimestep;
  eflag = eflag_global + eflag_atom;

  flag = 0;
  int vflag_global = 0;
  for (i = 0; i < nvlist_global; i++)
    if (vlist_global[i]->matchstep(ntimestep)) flag = 1;
  if (flag) vflag_global = virial_style;

  flag = 0;
  int vflag_atom = 0;
  for (i = 0; i < nvlist_atom; i++)
    if (vlist_atom[i]->matchstep(ntimestep)) flag = 1;
  if (flag) vflag_atom = VIRIAL_ATOM;

  flag = 0;
  int cvflag_atom = 0;
  for (i = 0; i < ncvlist_atom; i++)
    if (cvlist_atom[i]->matchstep(ntimestep)) flag = 1;
  if (flag) cvflag_atom = VIRIAL_CENTROID;

  if (vflag_global) update->vflag_global = update->ntimestep;
  if (vflag_atom || cvflag_atom) update->vflag_atom = update->ntimestep;
  vflag = vflag_global + vflag_atom + cvflag_atom;
}

/* ----------------------------------------------------------------------
   compute and return ||force||_2^2
------------------------------------------------------------------------- */

double Min::fnorm_sqr()
{
  int i,n;
  double *fatom;

  double local_norm2_sqr = 0.0;
  for (i = 0; i < nvec; i++) local_norm2_sqr += fvec[i]*fvec[i];
  if (nextra_atom) {
    for (int m = 0; m < nextra_atom; m++) {
      fatom = fextra_atom[m];
      n = extra_nlen[m];
      for (i = 0; i < n; i++)
        local_norm2_sqr += fatom[i]*fatom[i];
    }
  }

  double norm2_sqr = 0.0;
  MPI_Allreduce(&local_norm2_sqr,&norm2_sqr,1,MPI_DOUBLE,MPI_SUM,world);

  if (nextra_global)
    for (i = 0; i < nextra_global; i++)
      norm2_sqr += fextra[i]*fextra[i];

  return norm2_sqr;
}

/* ----------------------------------------------------------------------
   compute and return ||force||_inf
------------------------------------------------------------------------- */

double Min::fnorm_inf()
{
  int i,n;
  double *fatom;

  double local_norm_inf = 0.0;
  for (i = 0; i < nvec; i++)
    local_norm_inf = MAX(fvec[i]*fvec[i],local_norm_inf);
  if (nextra_atom) {
    for (int m = 0; m < nextra_atom; m++) {
      fatom = fextra_atom[m];
      n = extra_nlen[m];
      for (i = 0; i < n; i++)
        local_norm_inf = MAX(fatom[i]*fatom[i],local_norm_inf);
    }
  }

  double norm_inf = 0.0;
  MPI_Allreduce(&local_norm_inf,&norm_inf,1,MPI_DOUBLE,MPI_MAX,world);

  if (nextra_global)
    for (i = 0; i < nextra_global; i++)
      norm_inf = MAX(fextra[i]*fextra[i],norm_inf);

  return norm_inf;
}

/* ----------------------------------------------------------------------
   compute and return ||force||_max (inf norm per-vector)
------------------------------------------------------------------------- */

double Min::fnorm_max()
{
  int i,n;
  double fdotf,*fatom;

  double local_norm_max = 0.0;
  for (i = 0; i < nvec; i+=3) {
    fdotf = fvec[i]*fvec[i]+fvec[i+1]*fvec[i+1]+fvec[i+2]*fvec[i+2];
    local_norm_max = MAX(fdotf,local_norm_max);
  }
  if (nextra_atom) {
    for (int m = 0; m < nextra_atom; m++) {
      fatom = fextra_atom[m];
      n = extra_nlen[m];
      for (i = 0; i < n; i+=3) {
        fdotf = fatom[i]*fatom[i]+fatom[i+1]*fatom[i+1]+fatom[i+2]*fatom[i+2];
        local_norm_max = MAX(fdotf,local_norm_max);
      }
    }
  }

  double norm_max = 0.0;
  MPI_Allreduce(&local_norm_max,&norm_max,1,MPI_DOUBLE,MPI_MAX,world);

  if (nextra_global) {
    for (i = 0; i < nextra_global; i+=3) {
      fdotf = fextra[i]*fextra[i];
      norm_max = MAX(fdotf,norm_max);
    }
  }
  return norm_max;
}

/* ----------------------------------------------------------------------
   compute and return  sum_i||mag. torque_i||_2 (in eV)
------------------------------------------------------------------------- */

double Min::total_torque()
{
  double ftotsqone,ftotsqall;
  int nlocal = atom->nlocal;
  double hbar = force->hplanck/MY_2PI;
  double tx,ty,tz;
  double **sp = atom->sp;
  double **fm = atom->fm;

  ftotsqone = ftotsqall = 0.0;
  for (int i = 0; i < nlocal; i++) {
    tx = fm[i][1]*sp[i][2] - fm[i][2]*sp[i][1];
    ty = fm[i][2]*sp[i][0] - fm[i][0]*sp[i][2];
    tz = fm[i][0]*sp[i][1] - fm[i][1]*sp[i][0];
    ftotsqone += tx*tx + ty*ty + tz*tz;
  }

  // summing all fmsqtot on this replica

  MPI_Allreduce(&ftotsqone,&ftotsqall,1,MPI_DOUBLE,MPI_SUM,world);

  // multiply it by hbar so that units are in eV

  return sqrt(ftotsqall) * hbar;
}

/* ----------------------------------------------------------------------
   compute and return max_i ||mag. torque components|| (in eV)
------------------------------------------------------------------------- */

double Min::inf_torque()
{
  double fmaxsqone,fmaxsqall;
  int nlocal = atom->nlocal;
  double hbar = force->hplanck/MY_2PI;
  double tx,ty,tz;
  double **sp = atom->sp;
  double **fm = atom->fm;

  fmaxsqone = fmaxsqall = 0.0;
  for (int i = 0; i < nlocal; i++) {
    tx = fm[i][1]*sp[i][2] - fm[i][2]*sp[i][1];
    ty = fm[i][2]*sp[i][0] - fm[i][0]*sp[i][2];
    tz = fm[i][0]*sp[i][1] - fm[i][1]*sp[i][0];
    fmaxsqone = MAX(fmaxsqone,tx*tx);
    fmaxsqone = MAX(fmaxsqone,ty*ty);
    fmaxsqone = MAX(fmaxsqone,tz*tz);
  }

  // finding max fm on this replica

  fmaxsqall = fmaxsqone;
  MPI_Allreduce(&fmaxsqone,&fmaxsqall,1,MPI_DOUBLE,MPI_MAX,world);

  // multiply it by hbar so that units are in eV

  return sqrt(fmaxsqall) * hbar;
}

/* ----------------------------------------------------------------------
   compute and return max_i ||mag. torque_i|| (in eV)
------------------------------------------------------------------------- */

double Min::max_torque()
{
  double fmsq,fmaxsqone,fmaxsqall;
  int nlocal = atom->nlocal;
  double hbar = force->hplanck/MY_2PI;
  double tx,ty,tz;
  double **sp = atom->sp;
  double **fm = atom->fm;

  fmaxsqone = fmaxsqall = 0.0;
  for (int i = 0; i < nlocal; i++) {
    tx = fm[i][1]*sp[i][2] - fm[i][2]*sp[i][1];
    ty = fm[i][2]*sp[i][0] - fm[i][0]*sp[i][2];
    tz = fm[i][0]*sp[i][1] - fm[i][1]*sp[i][0];
    fmsq = tx*tx + ty*ty + tz*tz;
    fmaxsqone = MAX(fmaxsqone,fmsq);
  }

  // finding max fm on this replica

  fmaxsqall = fmaxsqone;
  MPI_Allreduce(&fmaxsqone,&fmaxsqall,1,MPI_DOUBLE,MPI_MAX,world);

  // multiply it by hbar so that units are in eV

  return sqrt(fmaxsqall) * hbar;
}

/* ----------------------------------------------------------------------
   possible stop conditions
------------------------------------------------------------------------- */

char *Min::stopstrings(int n)
{
  const char *strings[] = {"max iterations",
                           "max force evaluations",
                           "energy tolerance",
                           "force tolerance",
                           "search direction is not downhill",
                           "linesearch alpha is zero",
                           "forces are zero",
                           "quadratic factors are zero",
                           "trust region too small",
                           "HFTN minimizer error",
                           "walltime limit reached",
                           "max iterations with v.f negative"};
  return (char *) strings[n];
}
