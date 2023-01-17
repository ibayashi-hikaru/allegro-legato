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

#include "prd.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "finish.h"
#include "fix_event_prd.h"
#include "integrate.h"
#include "memory.h"
#include "min.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "random_mars.h"
#include "random_park.h"
#include "region.h"
#include "timer.h"
#include "universe.h"
#include "update.h"
#include "velocity.h"

#include <cstring>

using namespace LAMMPS_NS;

enum{SINGLE_PROC_DIRECT,SINGLE_PROC_MAP,MULTI_PROC};

/* ---------------------------------------------------------------------- */

PRD::PRD(LAMMPS *lmp) : Command(lmp) {}

/* ----------------------------------------------------------------------
   perform PRD simulation on one or more replicas
------------------------------------------------------------------------- */

void PRD::command(int narg, char **arg)
{
  int ireplica;

  // error checks

  if (domain->box_exist == 0)
    error->all(FLERR,"PRD command before simulation box is defined");
  if (universe->nworlds != universe->nprocs &&
      atom->map_style == Atom::MAP_NONE)
    error->all(FLERR,"Cannot use PRD with multi-processor replicas "
               "unless atom map exists");
  if (universe->nworlds == 1 && comm->me == 0)
    error->warning(FLERR,"Running PRD with only one replica");

  if (narg < 7) error->universe_all(FLERR,"Illegal prd command");

  // read as double so can cast to bigint

  int nsteps = utils::inumeric(FLERR,arg[0],false,lmp);
  t_event = utils::inumeric(FLERR,arg[1],false,lmp);
  n_dephase = utils::inumeric(FLERR,arg[2],false,lmp);
  t_dephase = utils::inumeric(FLERR,arg[3],false,lmp);
  t_corr = utils::inumeric(FLERR,arg[4],false,lmp);

  char *id_compute = utils::strdup(arg[5]);
  int seed = utils::inumeric(FLERR,arg[6],false,lmp);

  options(narg-7,&arg[7]);

  // total # of timesteps must be multiple of t_event

  if (t_event <= 0) error->universe_all(FLERR,"Invalid t_event in prd command");
  if (nsteps % t_event)
    error->universe_all(FLERR,"PRD nsteps must be multiple of t_event");
  if (t_corr % t_event)
    error->universe_all(FLERR,"PRD t_corr must be multiple of t_event");

  // local storage

  int me_universe = universe->me;
  int nprocs_universe = universe->nprocs;
  int nreplica = universe->nworlds;
  int iworld = universe->iworld;

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  // comm_replica = communicator between all proc 0s across replicas

  int color = me;
  MPI_Comm_split(universe->uworld,color,0,&comm_replica);

  // comm mode for inter-replica exchange of coords

  if (nreplica == nprocs_universe && atom->sortfreq == 0)
    cmode = SINGLE_PROC_DIRECT;
  else if (nreplica == nprocs_universe) cmode = SINGLE_PROC_MAP;
  else cmode = MULTI_PROC;

  // workspace for inter-replica communication

  natoms = atom->natoms;

  tagall = nullptr;
  xall = nullptr;
  imageall = nullptr;

  if (cmode != SINGLE_PROC_DIRECT) {
    memory->create(tagall,natoms,"prd:tagall");
    memory->create(xall,natoms,3,"prd:xall");
    memory->create(imageall,natoms,"prd:imageall");
  }

  counts = nullptr;
  displacements = nullptr;

  if (cmode == MULTI_PROC) {
    memory->create(counts,nprocs,"prd:counts");
    memory->create(displacements,nprocs,"prd:displacements");
  }

  // random_select = same RNG for each replica, for multiple event selection
  // random_clock = same RNG for each replica, for clock updates
  // random_dephase = unique RNG for each replica, for dephasing

  random_select = new RanPark(lmp,seed);
  random_clock = new RanPark(lmp,seed+1000);
  random_dephase = new RanMars(lmp,seed+iworld);

  // create ComputeTemp class to monitor temperature

  temperature = modify->add_compute("prd_temp all temp");

  // create Velocity class for velocity creation in dephasing
  // pass it temperature compute, loop_setting, dist_setting settings

  atom->check_mass(FLERR);
  velocity = new Velocity(lmp);
  velocity->init_external("all");

  char *args[2];
  args[0] = (char *) "temp";
  args[1] = (char *) "prd_temp";
  velocity->options(2,args);
  args[0] = (char *) "loop";
  args[1] = (char *) loop_setting;
  if (loop_setting) velocity->options(2,args);
  args[0] = (char *) "dist";
  args[1] = (char *) dist_setting;
  if (dist_setting) velocity->options(2,args);

  // create FixEventPRD class to store event and pre-quench states

  fix_event = (FixEventPRD *) modify->add_fix("prd_event all EVENT/PRD");

  // create Finish for timing output

  finish = new Finish(lmp);

  // string clean-up

  delete [] loop_setting;
  delete [] dist_setting;

  // assign FixEventPRD to event-detection compute
  // necessary so it will know atom coords at last event

  int icompute = modify->find_compute(id_compute);
  if (icompute < 0) error->all(FLERR,"Could not find compute ID for PRD");
  compute_event = modify->compute[icompute];
  compute_event->reset_extra_compute_fix("prd_event");

  // reset reneighboring criteria since will perform minimizations

  neigh_every = neighbor->every;
  neigh_delay = neighbor->delay;
  neigh_dist_check = neighbor->dist_check;

  if (neigh_every != 1 || neigh_delay != 0 || neigh_dist_check != 1) {
    if (me == 0)
      error->warning(FLERR,"Resetting reneighboring criteria during PRD");
  }

  neighbor->every = 1;
  neighbor->delay = 0;
  neighbor->dist_check = 1;

  // initialize PRD as if one long dynamics run

  update->whichflag = 1;
  update->nsteps = nsteps;
  update->beginstep = update->firststep = update->ntimestep;
  update->endstep = update->laststep = update->firststep + nsteps;
  update->restrict_output = 1;
  if (update->laststep < 0)
    error->all(FLERR,"Too many timesteps");

  lmp->init();

  // init minimizer settings and minimizer itself

  update->etol = etol;
  update->ftol = ftol;
  update->max_eval = maxeval;

  update->minimize->init();

  // cannot use PRD with a changing box
  // removing this restriction would require saving/restoring box params

  if (domain->box_change)
    error->all(FLERR,"Cannot use PRD with a changing box");

  // cannot use PRD with time-dependent fixes or regions

  for (int i = 0; i < modify->nfix; i++)
    if (modify->fix[i]->time_depend)
      error->all(FLERR,"Cannot use PRD with a time-dependent fix defined");

  for (int i = 0; i < domain->nregion; i++)
    if (domain->regions[i]->dynamic_check())
      error->all(FLERR,"Cannot use PRD with a time-dependent region defined");

  // perform PRD simulation

  if (me_universe == 0 && universe->uscreen)
    fprintf(universe->uscreen,"Setting up PRD ...\n");

  if (me_universe == 0) {
    if (universe->uscreen)
      fprintf(universe->uscreen,"Step CPU Clock Event "
              "Correlated Coincident Replica\n");
    if (universe->ulogfile)
      fprintf(universe->ulogfile,"Step CPU Clock Event "
              "Correlated Coincident Replica\n");
  }

  // store hot state and quenched event for replica 0
  // use share_event() to copy that info to all replicas
  // this insures all start from same place

  // need this line if quench() does only setup_minimal()
  // update->minimize->setup();

  fix_event->store_state_quench();
  quench();
  ncoincident = 0;
  share_event(0,0,0);

  timer->init();
  timer->barrier_start();
  time_start = timer->get_wall(Timer::TOTAL);

  log_event();

  // do full init/setup since are starting all replicas after event
  // replica 0 bcasts temp to all replicas if temp_dephase is not set

  update->whichflag = 1;
  lmp->init();
  update->integrate->setup(1);

  if (temp_flag == 0) {
    if (universe->iworld == 0) temp_dephase = temperature->compute_scalar();
    MPI_Bcast(&temp_dephase,1,MPI_DOUBLE,universe->root_proc[0],
              universe->uworld);
  }

  // main loop: look for events until out of time
  // (1) dephase independently on each proc after event
  // (2) loop: dynamics, store state, quench, check event, restore state
  // (3) share and record event

  nbuild = ndanger = 0;
  time_dephase = time_dynamics = time_quench = time_comm = time_output = 0.0;
  bigint clock = 0;

  timer->init();
  timer->barrier_start();
  time_start = timer->get_wall(Timer::TOTAL);

  int istep = 0;

  while (istep < nsteps) {
    dephase();

    if (stepmode == 0) istep = update->ntimestep - update->beginstep;
    else istep = clock;

    ireplica = -1;
    while (istep < nsteps) {
      dynamics(t_event,time_dynamics);
      fix_event->store_state_quench();
      quench();
      clock += (bigint)t_event*universe->nworlds;
      ireplica = check_event();
      if (ireplica >= 0) break;
      fix_event->restore_state_quench();
      if (stepmode == 0) istep = update->ntimestep - update->beginstep;
      else istep = clock;
    }
    if (ireplica < 0) break;

    // decrement clock by random time at which 1 or more events occurred

    int frac_t_event = t_event;
    for (int i = 0; i < fix_event->ncoincident; i++) {
      int frac_rand = static_cast<int> (random_clock->uniform() * t_event);
      frac_t_event = MIN(frac_t_event,frac_rand);
    }
    int decrement = (t_event - frac_t_event)*universe->nworlds;
    clock -= decrement;

    // share event across replicas
    // NOTE: would be potentially more efficient for correlated events
    //   if don't share until correlated check below has completed
    // this will complicate the dump (always on replica 0)

    share_event(ireplica,1,decrement);
    log_event();

    int restart_flag = 0;
    if (output->restart_flag && universe->iworld == 0) {
      if (output->restart_every_single &&
          fix_event->event_number % output->restart_every_single == 0)
        restart_flag = 1;
      if (output->restart_every_double &&
          fix_event->event_number % output->restart_every_double == 0)
        restart_flag = 1;
    }

    // correlated event loop
    // other procs could be dephasing during this time

    int corr_endstep = update->ntimestep + t_corr;
    while (update->ntimestep < corr_endstep) {
      if (update->ntimestep == update->endstep) {
        restart_flag = 0;
        break;
      }
      dynamics(t_event,time_dynamics);
      fix_event->store_state_quench();
      quench();
      clock += t_event;
      int corr_event_check = check_event(ireplica);
      if (corr_event_check >= 0) {
        share_event(ireplica,2,0);
        log_event();
        corr_endstep = update->ntimestep + t_corr;
      } else fix_event->restore_state_quench();
    }

    // full init/setup since are starting all replicas after event
    // event replica bcasts temp to all replicas if temp_dephase is not set

    update->whichflag = 1;
    lmp->init();
    update->integrate->setup(1);

    timer->barrier_start();

    if (t_corr > 0) replicate(ireplica);
    if (temp_flag == 0) {
      if (ireplica == universe->iworld)
        temp_dephase = temperature->compute_scalar();
      MPI_Bcast(&temp_dephase,1,MPI_DOUBLE,universe->root_proc[ireplica],
                      universe->uworld);
    }

    timer->barrier_stop();
    time_comm += timer->get_wall(Timer::TOTAL);

    // write restart file of hot coords

    if (restart_flag) {
      timer->barrier_start();
      output->write_restart(update->ntimestep);
      timer->barrier_stop();
      time_output += timer->get_wall(Timer::TOTAL);
    }

    if (stepmode == 0) istep = update->ntimestep - update->beginstep;
    else istep = clock;
  }

  if (stepmode) nsteps = update->ntimestep - update->beginstep;

  // set total timers and counters so Finish() will process them

  timer->set_wall(Timer::TOTAL, time_start);
  timer->barrier_stop();

  timer->set_wall(Timer::DEPHASE, time_dephase);
  timer->set_wall(Timer::DYNAMICS, time_dynamics);
  timer->set_wall(Timer::QUENCH, time_quench);
  timer->set_wall(Timer::REPCOMM, time_comm);
  timer->set_wall(Timer::REPOUT, time_output);

  neighbor->ncalls = nbuild;
  neighbor->ndanger = ndanger;

  if (me_universe == 0) {
    if (universe->uscreen)
      fprintf(universe->uscreen,
              "Loop time of %g on %d procs for %d steps with " BIGINT_FORMAT
              " atoms\n",
              timer->get_wall(Timer::TOTAL),nprocs_universe,
              nsteps,atom->natoms);
    if (universe->ulogfile)
      fprintf(universe->ulogfile,
              "Loop time of %g on %d procs for %d steps with " BIGINT_FORMAT
              " atoms\n",
              timer->get_wall(Timer::TOTAL),nprocs_universe,
              nsteps,atom->natoms);
  }

  if (me == 0) utils::logmesg(lmp,"\nPRD done\n");

  finish->end(2);

  update->whichflag = 0;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;
  update->restrict_output = 0;

  // reset reneighboring criteria

  neighbor->every = neigh_every;
  neighbor->delay = neigh_delay;
  neighbor->dist_check = neigh_dist_check;

  // clean up

  memory->destroy(tagall);
  memory->destroy(xall);
  memory->destroy(imageall);
  memory->destroy(counts);
  memory->destroy(displacements);

  delete [] id_compute;
  MPI_Comm_free(&comm_replica);
  delete random_select;
  delete random_clock;
  delete random_dephase;
  delete velocity;
  delete finish;
  modify->delete_compute("prd_temp");
  modify->delete_fix("prd_event");

  compute_event->reset_extra_compute_fix(nullptr);
}

/* ----------------------------------------------------------------------
   dephasing = one or more short runs with new random velocities
------------------------------------------------------------------------- */

void PRD::dephase()
{
  bigint ntimestep_hold = update->ntimestep;

  // n_dephase iterations of dephasing, each of t_dephase steps

  for (int i = 0; i < n_dephase; i++) {

    fix_event->store_state_dephase();

    // do not proceed to next iteration until an event-free run occurs

    int done = 0;
    while (!done) {
      int seed = static_cast<int> (random_dephase->uniform() * MAXSMALLINT);
      if (seed == 0) seed = 1;
      velocity->create(temp_dephase,seed);

      dynamics(t_dephase,time_dephase);
      fix_event->store_state_quench();
      quench();

      if (compute_event->compute_scalar() > 0.0) {
        fix_event->restore_state_dephase();
        update->ntimestep -= t_dephase;
        log_event();
      } else {
        fix_event->restore_state_quench();
        done = 1;
      }

      if (temp_flag == 0) temp_dephase = temperature->compute_scalar();
    }
  }

  // reset timestep as if dephase did not occur
  // clear timestep storage from computes, since now invalid

  update->ntimestep = ntimestep_hold;
  for (int i = 0; i < modify->ncompute; i++)
    if (modify->compute[i]->timeflag) modify->compute[i]->clearstep();
}

/* ----------------------------------------------------------------------
   short dynamics run: for event search, decorrelation, or dephasing
------------------------------------------------------------------------- */

void PRD::dynamics(int nsteps, double &time_category)
{
  update->whichflag = 1;
  update->nsteps = nsteps;

  lmp->init();
  update->integrate->setup(1);
  // this may be needed if don't do full init
  //modify->addstep_compute_all(update->ntimestep);
  bigint ncalls = neighbor->ncalls;

  timer->barrier_start();
  update->integrate->run(nsteps);
  timer->barrier_stop();
  time_category += timer->get_wall(Timer::TOTAL);

  nbuild += neighbor->ncalls - ncalls;
  ndanger += neighbor->ndanger;

  update->integrate->cleanup();
  finish->end(0);
}

/* ----------------------------------------------------------------------
   quench minimization
------------------------------------------------------------------------- */

void PRD::quench()
{
  bigint ntimestep_hold = update->ntimestep;
  bigint endstep_hold = update->endstep;

  // need to change whichflag so that minimize->setup() calling
  // modify->setup() will call fix->min_setup()

  update->whichflag = 2;
  update->nsteps = maxiter;
  update->endstep = update->laststep = update->firststep + maxiter;
  if (update->laststep < 0)
    error->all(FLERR,"Too many iterations");

  // full init works

  lmp->init();
  update->minimize->setup();

  // partial init does not work

  //modify->addstep_compute_all(update->ntimestep);
  //update->minimize->setup_minimal(1);

  int ncalls = neighbor->ncalls;

  timer->barrier_start();
  update->minimize->run(maxiter);
  timer->barrier_stop();
  time_quench += timer->get_wall(Timer::TOTAL);

  if (neighbor->ncalls == ncalls) quench_reneighbor = 0;
  else quench_reneighbor = 1;

  update->minimize->cleanup();
  finish->end(0);

  // reset timestep as if quench did not occur
  // clear timestep storage from computes, since now invalid

  update->ntimestep = ntimestep_hold;
  update->endstep = update->laststep = endstep_hold;
  for (int i = 0; i < modify->ncompute; i++)
    if (modify->compute[i]->timeflag) modify->compute[i]->clearstep();
}

/* ----------------------------------------------------------------------
   check for an event in any replica
   if replica_num is non-negative only check for event on replica_num
   if multiple events, choose one at random
   return -1 if no event
   else return ireplica = world in which event occurred
------------------------------------------------------------------------- */

int PRD::check_event(int replica_num)
{
  int worldflag,universeflag,scanflag,replicaflag,ireplica;

  worldflag = 0;
  if (compute_event->compute_scalar() > 0.0) worldflag = 1;
  if (replica_num >= 0 && replica_num != universe->iworld) worldflag = 0;

  timer->barrier_start();

  if (me == 0) MPI_Allreduce(&worldflag,&universeflag,1,
                             MPI_INT,MPI_SUM,comm_replica);
  MPI_Bcast(&universeflag,1,MPI_INT,0,world);

  ncoincident = universeflag;

  if (!universeflag) ireplica = -1;
  else {

    // multiple events, choose one at random
    // iwhich = random # from 1 to N, N = # of events to choose from
    // scanflag = 1 to N on replicas with an event, 0 on non-event replicas
    // exit with worldflag = 1 on chosen replica, 0 on all others
    // note worldflag is already 0 on replicas that didn't perform event

    if (universeflag > 1) {
      int iwhich = static_cast<int>
        (universeflag*random_select->uniform()) + 1;

      if (me == 0)
        MPI_Scan(&worldflag,&scanflag,1,MPI_INT,MPI_SUM,comm_replica);
      MPI_Bcast(&scanflag,1,MPI_INT,0,world);

      if (scanflag != iwhich) worldflag = 0;
    }

    if (worldflag) replicaflag = universe->iworld;
    else replicaflag = 0;

    if (me == 0) MPI_Allreduce(&replicaflag,&ireplica,1,
                               MPI_INT,MPI_SUM,comm_replica);
    MPI_Bcast(&ireplica,1,MPI_INT,0,world);
  }

  timer->barrier_stop();
  time_comm += timer->get_wall(Timer::TOTAL);

  return ireplica;
}

/* ----------------------------------------------------------------------
   share quenched and hot coords owned by ireplica with all replicas
   all replicas store event in fix_event
   replica 0 dumps event snapshot
   flag = 0 = called before PRD run
   flag = 1 = called during PRD run = not correlated event
   flag = 2 = called during PRD run = correlated event
------------------------------------------------------------------------- */

void PRD::share_event(int ireplica, int flag, int decrement)
{
  timer->barrier_start();

  // communicate quenched coords to all replicas and store as event
  // decrement event counter if flag = 0 since not really an event

  replicate(ireplica);
  timer->barrier_stop();
  time_comm += timer->get_wall(Timer::TOTAL);

  // adjust time for last correlated event check (not on first event)

  int corr_adjust = t_corr;
  if (fix_event->event_number < 1 || flag == 2) corr_adjust = 0;

  // delta = time since last correlated event check

  int delta = update->ntimestep - fix_event->event_timestep - corr_adjust;

  // if this is a correlated event, time elapsed only on one partition

  if (flag != 2) delta *= universe->nworlds;
  if (delta > 0 && flag != 2) delta -= decrement;
  delta += corr_adjust;

  // delta passed to store_event_prd() should make its clock update
  //   be consistent with clock in main PRD loop
  // don't change the clock or timestep if this is a restart

  if (flag == 0 && fix_event->event_number != 0)
    fix_event->store_event_prd(fix_event->event_timestep,0);
  else {
    fix_event->store_event_prd(update->ntimestep,delta);
    fix_event->replica_number = ireplica;
    fix_event->correlated_event = 0;
    if (flag == 2) fix_event->correlated_event = 1;
    fix_event->ncoincident = ncoincident;
  }
  if (flag == 0) fix_event->event_number--;

  // dump snapshot of quenched coords, only on replica 0
  // must reneighbor and compute forces before dumping
  // since replica 0 possibly has new state from another replica
  // addstep_compute_all insures eng/virial are calculated if needed

  if (output->ndump && universe->iworld == 0) {
    timer->barrier_start();
    modify->addstep_compute_all(update->ntimestep);
    update->integrate->setup_minimal(1);
    output->write_dump(update->ntimestep);
    timer->barrier_stop();
    time_output += timer->get_wall(Timer::TOTAL);
  }

  // restore and communicate hot coords to all replicas

  fix_event->restore_state_quench();
  timer->barrier_start();
  replicate(ireplica);
  timer->barrier_stop();
  time_comm += timer->get_wall(Timer::TOTAL);
}

/* ----------------------------------------------------------------------
   universe proc 0 prints event info
------------------------------------------------------------------------- */

void PRD::log_event()
{
  timer->set_wall(Timer::TOTAL, time_start);
  if (universe->me == 0) {
    if (universe->uscreen)
      fprintf(universe->uscreen,
              BIGINT_FORMAT " %.3f " BIGINT_FORMAT " %d %d %d %d\n",
              fix_event->event_timestep,
              timer->elapsed(Timer::TOTAL),
              fix_event->clock,
              fix_event->event_number,fix_event->correlated_event,
              fix_event->ncoincident,
              fix_event->replica_number);
    if (universe->ulogfile)
      fprintf(universe->ulogfile,
              BIGINT_FORMAT " %.3f " BIGINT_FORMAT " %d %d %d %d\n",
              fix_event->event_timestep,
              timer->elapsed(Timer::TOTAL),
              fix_event->clock,
              fix_event->event_number,fix_event->correlated_event,
              fix_event->ncoincident,
              fix_event->replica_number);
  }
}

/* ----------------------------------------------------------------------
  communicate atom coords and image flags in ireplica to all other replicas
  if one proc per replica:
    direct overwrite via bcast
  else atoms could be stored in different order on a proc or on different procs:
    gather to root proc of event replica
    bcast to roots of other replicas
    bcast within each replica
    each proc extracts info for atoms it owns using atom IDs
------------------------------------------------------------------------- */

void PRD::replicate(int ireplica)
{
  int i,m;

  // -----------------------------------------------------
  // 3 cases: two for single proc per replica
  //          one for multiple procs per replica
  // -----------------------------------------------------

  // single proc per replica, no atom sorting
  // direct bcast of image and x

  if (cmode == SINGLE_PROC_DIRECT) {
    MPI_Bcast(atom->x[0],3*atom->nlocal,MPI_DOUBLE,ireplica,comm_replica);
    MPI_Bcast(atom->image,atom->nlocal,MPI_LMP_IMAGEINT,ireplica,comm_replica);
    return;
  }

  // single proc per replica, atom sorting is enabled
  // bcast atom IDs, x, image via tagall, xall, imageall
  // recv procs use atom->map() to match received info to owned atoms

  if (cmode == SINGLE_PROC_MAP) {
    double **x = atom->x;
    tagint *tag = atom->tag;
    imageint *image = atom->image;
    int nlocal = atom->nlocal;

    if (universe->iworld == ireplica) {
      memcpy(tagall,tag,nlocal*sizeof(tagint));
      memcpy(xall[0],x[0],3*nlocal*sizeof(double));
      memcpy(imageall,image,nlocal*sizeof(imageint));
    }

    MPI_Bcast(tagall,natoms,MPI_LMP_TAGINT,ireplica,comm_replica);
    MPI_Bcast(xall[0],3*natoms,MPI_DOUBLE,ireplica,comm_replica);
    MPI_Bcast(imageall,natoms,MPI_LMP_IMAGEINT,ireplica,comm_replica);

    for (i = 0; i < nlocal; i++) {
      m = atom->map(tagall[i]);
      x[m][0] = xall[i][0];
      x[m][1] = xall[i][1];
      x[m][2] = xall[i][2];
      atom->image[m] = imageall[i];
    }

    return;
  }

  // multiple procs per replica
  // MPI_Gather all atom IDs, x, image to root proc of ireplica
  // bcast to root of other replicas
  // bcast within each replica
  // each proc extracts info for atoms it owns via atom->map()
  // NOTE: assumes imagint and tagint are always the same size

  if (universe->iworld == ireplica) {
    MPI_Gather(&atom->nlocal,1,MPI_INT,counts,1,MPI_INT,0,world);
    displacements[0] = 0;
    for (i = 0; i < nprocs-1; i++)
      displacements[i+1] = displacements[i] + counts[i];
    MPI_Gatherv(atom->tag,atom->nlocal,MPI_LMP_TAGINT,
                tagall,counts,displacements,MPI_LMP_TAGINT,0,world);
    MPI_Gatherv(atom->image,atom->nlocal,MPI_LMP_IMAGEINT,
                imageall,counts,displacements,MPI_LMP_IMAGEINT,0,world);
    for (i = 0; i < nprocs; i++) counts[i] *= 3;
    for (i = 0; i < nprocs-1; i++)
      displacements[i+1] = displacements[i] + counts[i];
    MPI_Gatherv(atom->x[0],3*atom->nlocal,MPI_DOUBLE,
                xall[0],counts,displacements,MPI_DOUBLE,0,world);
  }

  if (me == 0) {
    MPI_Bcast(tagall,natoms,MPI_LMP_TAGINT,ireplica,comm_replica);
    MPI_Bcast(imageall,natoms,MPI_LMP_IMAGEINT,ireplica,comm_replica);
    MPI_Bcast(xall[0],3*natoms,MPI_DOUBLE,ireplica,comm_replica);
  }

  MPI_Bcast(tagall,natoms,MPI_LMP_TAGINT,0,world);
  MPI_Bcast(imageall,natoms,MPI_LMP_IMAGEINT,0,world);
  MPI_Bcast(xall[0],3*natoms,MPI_DOUBLE,0,world);

  double **x = atom->x;
  int nlocal = atom->nlocal;

  for (i = 0; i < natoms; i++) {
    m = atom->map(tagall[i]);
    if (m < 0 || m >= nlocal) continue;
    x[m][0] = xall[i][0];
    x[m][1] = xall[i][1];
    x[m][2] = xall[i][2];
    atom->image[m] = imageall[i];
  }
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of PRD input line
------------------------------------------------------------------------- */

void PRD::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR,"Illegal prd command");

  // set defaults

  etol = 0.1;
  ftol = 0.1;
  maxiter = 40;
  maxeval = 50;
  temp_flag = 0;
  stepmode = 0;

  loop_setting = utils::strdup("geom");
  dist_setting = utils::strdup("gaussian");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"min") == 0) {
      if (iarg+5 > narg) error->all(FLERR,"Illegal prd command");
      etol = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      ftol = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      maxiter = utils::inumeric(FLERR,arg[iarg+3],false,lmp);
      maxeval = utils::inumeric(FLERR,arg[iarg+4],false,lmp);
      if (maxiter < 0) error->all(FLERR,"Illegal prd command");
      iarg += 5;

    } else if (strcmp(arg[iarg],"temp") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal prd command");
      temp_flag = 1;
      temp_dephase = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (temp_dephase <= 0.0) error->all(FLERR,"Illegal prd command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"vel") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal prd command");
      delete [] loop_setting;
      delete [] dist_setting;

      if (strcmp(arg[iarg+1],"all") == 0) loop_setting = nullptr;
      else if (strcmp(arg[iarg+1],"local") == 0) loop_setting = nullptr;
      else if (strcmp(arg[iarg+1],"geom") == 0) loop_setting = nullptr;
      else error->all(FLERR,"Illegal prd command");
      loop_setting = utils::strdup(arg[iarg+1]);

      if (strcmp(arg[iarg+2],"uniform") == 0) dist_setting = nullptr;
      else if (strcmp(arg[iarg+2],"gaussian") == 0) dist_setting = nullptr;
      else error->all(FLERR,"Illegal prd command");
      dist_setting = utils::strdup(arg[iarg+2]);

      iarg += 3;

    } else if (strcmp(arg[iarg],"time") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal prd command");
      if (strcmp(arg[iarg+1],"steps") == 0) stepmode = 0;
      else if (strcmp(arg[iarg+1],"clock") == 0) stepmode = 1;
      else error->all(FLERR,"Illegal prd command");
      iarg += 2;

    } else error->all(FLERR,"Illegal prd command");
  }
}
