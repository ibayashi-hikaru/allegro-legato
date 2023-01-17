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

#include "fix_gpu.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "gpu_extra.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "respa.h"
#include "timer.h"
#include "universe.h"
#include "update.h"

#include <cstring>

#if (LAL_USE_OMP == 1)
#include <omp.h>
#endif

using namespace LAMMPS_NS;
using namespace FixConst;

enum{GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH};

extern int lmp_init_device(MPI_Comm world, MPI_Comm replica, const int ngpu,
                           const int first_gpu_id, const int gpu_mode,
                           const double particle_split, const int t_per_atom,
                           const double cell_size, char *opencl_args,
                           const int ocl_platform, char *device_type_flags,
                           const int block_pair);
extern void lmp_clear_device();
extern double lmp_gpu_forces(double **f, double **tor, double *eatom, double **vatom,
                             double *virial, double &ecoul, int &err_flag);
extern double lmp_gpu_update_bin_size(const double subx, const double suby, const double subz,
                                      const int nlocal, const double cut);

static const char cite_gpu_package[] =
  "GPU package (short-range, long-range and three-body potentials):\n\n"
  "@Article{Brown11,\n"
  " author = {W. M. Brown, P. Wang, S. J. Plimpton, A. N. Tharrington},\n"
  " title = {Implementing Molecular Dynamics on Hybrid High Performance Computers - Short Range Forces},\n"
  " journal = {Comp.~Phys.~Comm.},\n"
  " year =    2011,\n"
  " volume =  182,\n"
  " pages =   {898--911}\n"
  "}\n\n"
  "@Article{Brown12,\n"
  " author = {W. M. Brown, A. Kohlmeyer, S. J. Plimpton, A. N. Tharrington},\n"
  " title = {Implementing Molecular Dynamics on Hybrid High Performance Computers - Particle-Particle Particle-Mesh},\n"
  " journal = {Comp.~Phys.~Comm.},\n"
  " year =    2012,\n"
  " volume =  183,\n"
  " pages =   {449--459}\n"
  "}\n\n"
  "@Article{Brown13,\n"
  " author = {W. M. Brown, Y. Masako},\n"
  " title = {Implementing Molecular Dynamics on Hybrid High Performance Computers – Three-Body Potentials},\n"
  " journal = {Comp.~Phys.~Comm.},\n"
  " year =    2013,\n"
  " volume =  184,\n"
  " pages =   {2785--2793}\n"
  "}\n\n"
  "@Article{Trung15,\n"
  " author = {T. D. Nguyen, S. J. Plimpton},\n"
  " title = {Accelerating dissipative particle dynamics simulations for soft matter systems},\n"
  " journal = {Comput.~Mater.~Sci.},\n"
  " year =    2015,\n"
  " volume =  100,\n"
  " pages =   {173--180}\n"
  "}\n\n"
  "@Article{Trung17,\n"
  " author = {T. D. Nguyen},\n"
  " title = {GPU-accelerated Tersoff potentials for massively parallel Molecular Dynamics simulations},\n"
  " journal = {Comp.~Phys.~Comm.},\n"
  " year =    2017,\n"
  " volume =  212,\n"
  " pages =   {113--122}\n"
  "}\n\n"
  "@Article{Nikolskiy19,\n"
  " author = {V. Nikolskiy, V. Stegailov},\n"
  " title = {GPU acceleration of four-site water models in LAMMPS},\n"
  " journal = {Proceeding of the International Conference on Parallel Computing (ParCo 2019), Prague, Czech Republic},\n"
  " year =    2019\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

FixGPU::FixGPU(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (lmp->citeme) lmp->citeme->add(cite_gpu_package);

  if (narg < 4) error->all(FLERR,"Illegal package gpu command");

  // If ngpu is 0, autoset ngpu to the number of devices per node matching
  // best device
  int ngpu = atoi(arg[3]);
  if (ngpu < 0) error->all(FLERR,"Illegal package gpu command");

  // Negative value indicate GPU package should find the best device ID
  int first_gpu_id = -1;

  // options

  _gpu_mode = GPU_NEIGH;
  _particle_split = 1.0;
  int nthreads = 0;
  int newtonflag = 0;
  int threads_per_atom = -1;
  double binsize = 0.0;
  char *opencl_args = nullptr;
  int block_pair = -1;
  int pair_only_flag = 0;
  int ocl_platform = -1;
  char *device_type_flags = nullptr;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"neigh") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      if (strcmp(arg[iarg+1],"yes") == 0) _gpu_mode = GPU_NEIGH;
      else if (strcmp(arg[iarg+1],"no") == 0) _gpu_mode = GPU_FORCE;
      else if (strcmp(arg[iarg+1],"hybrid") == 0) _gpu_mode = GPU_HYB_NEIGH;
      else error->all(FLERR,"Illegal package gpu command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"newton") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      if (strcmp(arg[iarg+1],"off") == 0) newtonflag = 0;
      else if (strcmp(arg[iarg+1],"on") == 0) newtonflag = 1;
      else error->all(FLERR,"Illegal package gpu command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"binsize") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      binsize = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (binsize <= 0.0) error->all(FLERR,"Illegal fix GPU command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"split") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      _particle_split = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (_particle_split == 0.0 || _particle_split > 1.0)
        error->all(FLERR,"Illegal package GPU command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"gpuID") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      first_gpu_id = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"tpa") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      threads_per_atom = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"omp") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      nthreads = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (nthreads < 0) error->all(FLERR,"Illegal fix GPU command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"platform") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      ocl_platform = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"device_type") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      device_type_flags = arg[iarg+1];
      iarg += 2;
    } else if (strcmp(arg[iarg],"blocksize") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      block_pair = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"pair/only") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      if (strcmp(arg[iarg+1],"off") == 0) pair_only_flag = 0;
      else if (strcmp(arg[iarg+1],"on") == 0) pair_only_flag = 1;
      else error->all(FLERR,"Illegal package gpu command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"ocl_args") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package gpu command");
      opencl_args = arg[iarg+1];
      iarg += 2;
    } else error->all(FLERR,"Illegal package gpu command");
  }

  #if (LAL_USE_OMP == 0)
  if (nthreads > 1)
    error->all(FLERR,"No OpenMP support compiled into the GPU package");
  #else
  if (nthreads > 0) {
    omp_set_num_threads(nthreads);
    comm->nthreads = nthreads;
  }
  #endif

  // set newton pair flag
  // require newtonflag = 0 since currently required by all GPU pair styles

  if (newtonflag == 1) error->all(FLERR,"Illegal package gpu command");

  force->newton_pair = newtonflag;
  if (force->newton_pair || force->newton_bond) force->newton = 1;
  else force->newton = 0;

  if (pair_only_flag) {
    lmp->suffixp = lmp->suffix;
    lmp->suffix = nullptr;
  } else {
    if (lmp->suffixp) {
      lmp->suffix = lmp->suffixp;
      lmp->suffixp = nullptr;
    }
  }

  // pass params to GPU library
  // change binsize default (0.0) to -1.0 used by GPU lib

  if (binsize == 0.0) binsize = -1.0;
  _binsize = binsize;
  int gpu_flag = lmp_init_device(universe->uworld, world, ngpu, first_gpu_id,
                                 _gpu_mode, _particle_split, threads_per_atom,
                                 binsize, opencl_args, ocl_platform,
                                 device_type_flags, block_pair);
  GPU_EXTRA::check_flag(gpu_flag,error,world);
}

/* ---------------------------------------------------------------------- */

FixGPU::~FixGPU()
{
  lmp_clear_device();
}

/* ---------------------------------------------------------------------- */

int FixGPU::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGPU::init()
{
  // GPU package cannot be used with atom_style template

  if (atom->molecular == Atom::TEMPLATE)
    error->all(FLERR,"GPU package does not (yet) work with "
               "atom_style template");

  // give a warning if no pair style is defined

  if (!force->pair && (comm->me == 0))
    error->warning(FLERR,"Using package gpu without any pair style defined");

  // make sure fdotr virial is not accumulated multiple times
  // also disallow GPU neighbor lists for hybrid styles

  if (force->pair_match("^hybrid",0) != nullptr) {
    PairHybrid *hybrid = (PairHybrid *) force->pair;
    for (int i = 0; i < hybrid->nstyles; i++)
      if (!utils::strmatch(hybrid->keywords[i],"/gpu$"))
        force->pair->no_virial_fdotr_compute = 1;
    if (_gpu_mode != GPU_FORCE)
      error->all(FLERR, "Must not use GPU neighbor lists with hybrid pair style");
  }

  // rRESPA support

  if (utils::strmatch(update->integrate_style,"^respa"))
    _nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixGPU::setup(int vflag)
{
  if (_gpu_mode == GPU_NEIGH || _gpu_mode == GPU_HYB_NEIGH)
    if (neighbor->exclude_setting() != 0)
      error->all(FLERR, "Cannot use neigh_modify exclude with GPU neighbor builds");

  if (utils::strmatch(update->integrate_style,"^verlet")) post_force(vflag);
  else {
    // In setup only, all forces calculated on GPU are put in the outer level
    ((Respa *) update->integrate)->copy_flevel_f(_nlevels_respa-1);
    post_force(vflag);
    ((Respa *) update->integrate)->copy_f_flevel(_nlevels_respa-1);
  }
}

/* ---------------------------------------------------------------------- */

void FixGPU::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixGPU::post_force(int /* vflag */)
{
  if (!force->pair) return;

  timer->stamp();
  double lvirial[6];
  for (int i = 0; i < 6; i++) lvirial[i] = 0.0;
  int err_flag = 0;
  double my_eng = lmp_gpu_forces(atom->f, atom->torque, force->pair->eatom, force->pair->vatom,
                                 lvirial, force->pair->eng_coul, err_flag);
  if (err_flag==1)
    error->one(FLERR,"Neighbor list problem on the GPU. Try increasing the value of 'neigh_modify one' "
               "or the GPU neighbor list 'binsize'.");

  force->pair->eng_vdwl += my_eng;
  force->pair->virial[0] += lvirial[0];
  force->pair->virial[1] += lvirial[1];
  force->pair->virial[2] += lvirial[2];
  force->pair->virial[3] += lvirial[3];
  force->pair->virial[4] += lvirial[4];
  force->pair->virial[5] += lvirial[5];

  if (force->pair->vflag_fdotr) force->pair->virial_fdotr_compute();
  timer->stamp(Timer::PAIR);
}

/* ---------------------------------------------------------------------- */

void FixGPU::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixGPU::post_force_respa(int vflag, int /* ilevel */, int /* iloop */)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

double FixGPU::memory_usage()
{
  double bytes = 0.0;
  // memory usage currently returned by pair routine
  return bytes;
}

double FixGPU::binsize(const double subx, const double suby,
                       const double subz, const int nlocal,
                       const double cut) {
  if (_binsize > 0.0) return _binsize;
  else if (_gpu_mode == GPU_FORCE || comm->cutghostuser)
    return cut * 0.5;
  else
    return lmp_gpu_update_bin_size(subx, suby, subz, nlocal, cut);
}
