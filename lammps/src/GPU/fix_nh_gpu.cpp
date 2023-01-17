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
   Contributing author: W. Michael Brown (Intel)
------------------------------------------------------------------------- */

#include "fix_nh_gpu.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "gpu_extra.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

#define TILTMAX 1.5

enum{NOBIAS,BIAS};
enum{ISO,ANISO,TRICLINIC};

typedef struct { double x,y,z; } dbl3_t;

/* ----------------------------------------------------------------------
   NVT,NPH,NPT integrators for improved Nose-Hoover equations of motion
 ---------------------------------------------------------------------- */

FixNHGPU::FixNHGPU(LAMMPS *lmp, int narg, char **arg) :
  FixNH(lmp, narg, arg)
{
  _dtfm = 0;
  _nlocal3 = 0;
  _nlocal_max = 0;
}

/* ---------------------------------------------------------------------- */

FixNHGPU::~FixNHGPU()
{
}

/* ---------------------------------------------------------------------- */

void FixNHGPU::setup(int vflag)
{
  FixNH::setup(vflag);
  if (utils::strmatch(update->integrate_style,"^respa"))
    _respa_on = 1;
  else
    _respa_on = 0;
  reset_dt();
}

/* ----------------------------------------------------------------------
   change box size
   remap all atoms or dilate group atoms depending on allremap flag
   if rigid bodies exist, scale rigid body centers-of-mass
------------------------------------------------------------------------- */

void FixNHGPU::remap()
{
  if (_respa_on) { FixNH::remap(); return; }

  double oldlo,oldhi;
  double expfac;

  dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *h = domain->h;

  // omega is not used, except for book-keeping

  for (int i = 0; i < 6; i++) omega[i] += dto*omega_dot[i];

  // convert pertinent atoms and rigid bodies to lamda coords
  const double hi0 = domain->h_inv[0];
  const double hi1 = domain->h_inv[1];
  const double hi2 = domain->h_inv[2];
  const double hi3 = domain->h_inv[3];
  const double hi4 = domain->h_inv[4];
  const double hi5 = domain->h_inv[5];
  const double b0 = domain->boxlo[0];
  const double b1 = domain->boxlo[1];
  const double b2 = domain->boxlo[2];

  if (allremap) {
    #if (LAL_USE_OMP == 1) && (LAL_USE_OMP_SIMD == 1)
    #pragma omp parallel for simd schedule(static)
    #elif (LAL_USE_OMP_SIMD == 1)
    #pragma omp simd
    #endif
    for (int i = 0; i < nlocal; i++) {
      const double d0 = x[i].x - b0;
      const double d1 = x[i].y - b1;
      const double d2 = x[i].z - b2;
      x[i].x = hi0*d0 + hi5*d1 + hi4*d2;
      x[i].y = hi1*d1 + hi3*d2;
      x[i].z = hi2*d2;
    }
  } else {
    #if (LAL_USE_OMP == 1) && (LAL_USE_OMP_SIMD == 1)
    #pragma omp parallel for simd schedule(static)
    #elif (LAL_USE_OMP_SIMD == 1)
    #pragma omp simd
    #endif
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & dilate_group_bit) {
        const double d0 = x[i].x - b0;
        const double d1 = x[i].y - b1;
        const double d2 = x[i].z - b2;
        x[i].x = hi0*d0 + hi5*d1 + hi4*d2;
        x[i].y = hi1*d1 + hi3*d2;
        x[i].z = hi2*d2;
      }
    }
  }

  if (nrigid)
    for (int i = 0; i < nrigid; i++)
      modify->fix[rfix[i]]->deform(0);

  // reset global and local box to new size/shape

  // this operation corresponds to applying the
  // translate and scale operations
  // corresponding to the solution of the following ODE:
  //
  // h_dot = omega_dot * h
  //
  // where h_dot, omega_dot and h are all upper-triangular
  // 3x3 tensors. In Voigt notation, the elements of the
  // RHS product tensor are:
  // h_dot = [0*0, 1*1, 2*2, 1*3+3*2, 0*4+5*3+4*2, 0*5+5*1]
  //
  // Ordering of operations preserves time symmetry.

  double dto2 = dto/2.0;
  double dto4 = dto/4.0;
  double dto8 = dto/8.0;

  // off-diagonal components, first half

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dto2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dto2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }
  }

  // scale diagonal components
  // scale tilt factors with cell, if set

  if (p_flag[0]) {
    oldlo = domain->boxlo[0];
    oldhi = domain->boxhi[0];
    expfac = exp(dto*omega_dot[0]);
    domain->boxlo[0] = (oldlo-fixedpoint[0])*expfac + fixedpoint[0];
    domain->boxhi[0] = (oldhi-fixedpoint[0])*expfac + fixedpoint[0];
  }

  if (p_flag[1]) {
    oldlo = domain->boxlo[1];
    oldhi = domain->boxhi[1];
    expfac = exp(dto*omega_dot[1]);
    domain->boxlo[1] = (oldlo-fixedpoint[1])*expfac + fixedpoint[1];
    domain->boxhi[1] = (oldhi-fixedpoint[1])*expfac + fixedpoint[1];
    if (scalexy) h[5] *= expfac;
  }

  if (p_flag[2]) {
    oldlo = domain->boxlo[2];
    oldhi = domain->boxhi[2];
    expfac = exp(dto*omega_dot[2]);
    domain->boxlo[2] = (oldlo-fixedpoint[2])*expfac + fixedpoint[2];
    domain->boxhi[2] = (oldhi-fixedpoint[2])*expfac + fixedpoint[2];
    if (scalexz) h[4] *= expfac;
    if (scaleyz) h[3] *= expfac;
  }

  // off-diagonal components, second half

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dto2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dto2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

  }

  domain->yz = h[3];
  domain->xz = h[4];
  domain->xy = h[5];

  // tilt factor to cell length ratio can not exceed TILTMAX in one step

  if (domain->yz < -TILTMAX*domain->yprd ||
      domain->yz > TILTMAX*domain->yprd ||
      domain->xz < -TILTMAX*domain->xprd ||
      domain->xz > TILTMAX*domain->xprd ||
      domain->xy < -TILTMAX*domain->xprd ||
      domain->xy > TILTMAX*domain->xprd)
    error->all(FLERR,"Fix npt/nph has tilted box too far in one step - "
               "periodic cell is too far from equilibrium state");

  domain->set_global_box();
  domain->set_local_box();

  // convert pertinent atoms and rigid bodies back to box coords
  const double h0 = domain->h[0];
  const double h1 = domain->h[1];
  const double h2 = domain->h[2];
  const double h3 = domain->h[3];
  const double h4 = domain->h[4];
  const double h5 = domain->h[5];
  const double nb0 = domain->boxlo[0];
  const double nb1 = domain->boxlo[1];
  const double nb2 = domain->boxlo[2];

  if (allremap) {
    #if (LAL_USE_OMP == 1) && (LAL_USE_OMP_SIMD == 1)
    #pragma omp parallel for simd schedule(static)
    #elif (LAL_USE_OMP_SIMD == 1)
    #pragma omp simd
    #endif
    for (int i = 0; i < nlocal; i++) {
      x[i].x = h0*x[i].x + h5*x[i].y + h4*x[i].z + nb0;
      x[i].y = h1*x[i].y + h3*x[i].z + nb1;
      x[i].z = h2*x[i].z + nb2;
    }
  } else {
    #if (LAL_USE_OMP == 1) && (LAL_USE_OMP_SIMD == 1)
    #pragma omp parallel for simd schedule(static)
    #elif (LAL_USE_OMP_SIMD == 1)
    #pragma omp simd
    #endif
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & dilate_group_bit) {
        x[i].x = h0*x[i].x + h5*x[i].y + h4*x[i].z + nb0;
        x[i].y = h1*x[i].y + h3*x[i].z + nb1;
        x[i].z = h2*x[i].z + nb2;
      }
    }
  }

  if (nrigid)
    for (int i = 0; i < nrigid; i++)
      modify->fix[rfix[i]]->deform(1);
}

/* ----------------------------------------------------------------------
   2nd half of Verlet update
------------------------------------------------------------------------- */

void FixNHGPU::final_integrate() {
  if (neighbor->ago == 0 && _respa_on == 0) reset_dt();
  FixNH::final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNHGPU::reset_dt()
{
  if (_respa_on) { FixNH::reset_dt(); return; }
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;
  dt4 = 0.25 * update->dt;
  dt8 = 0.125 * update->dt;
  dto = dthalf;

  if (pstat_flag)
    pdrag_factor = 1.0 - (update->dt * p_freq_max * drag / nc_pchain);

  if (tstat_flag)
    tdrag_factor = 1.0 - (update->dt * t_freq * drag / nc_tchain);

  const int * const mask = atom->mask;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst :
    atom->nlocal;

  if (nlocal > _nlocal_max) {
    if (_nlocal_max) memory->destroy(_dtfm);
    _nlocal_max = static_cast<int>(1.20 * nlocal);
    memory->create(_dtfm, _nlocal_max * 3, "fix_nh_gpu:dtfm");
  }

  _nlocal3 = nlocal * 3;

  if (igroup == 0) {
    if (atom->rmass) {
      const double * const rmass = atom->rmass;
      int n = 0;
      for (int i = 0; i < nlocal; i++) {
        const double dtfir = dtf / rmass[i];
        _dtfm[n++] = dtfir;
        _dtfm[n++] = dtfir;
        _dtfm[n++] = dtfir;
      }
    } else {
      const double * const mass = atom->mass;
      const int * const type = atom->type;
      int n = 0;
      for (int i = 0; i < nlocal; i++) {
        const double dtfim = dtf / mass[type[i]];
        _dtfm[n++] = dtfim;
        _dtfm[n++] = dtfim;
        _dtfm[n++] = dtfim;
      }
    }
  } else {
    if (atom->rmass) {
      const double * const rmass = atom->rmass;
      int n = 0;
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          const double dtfir = dtf / rmass[i];
          _dtfm[n++] = dtfir;
          _dtfm[n++] = dtfir;
          _dtfm[n++] = dtfir;
        } else {
          _dtfm[n++] = 0.0;
          _dtfm[n++] = 0.0;
          _dtfm[n++] = 0.0;
        }
    } else {
      const double * const mass = atom->mass;
      const int * const type = atom->type;
      int n = 0;
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          const double dtfim = dtf / mass[type[i]];
          _dtfm[n++] = dtfim;
          _dtfm[n++] = dtfim;
          _dtfm[n++] = dtfim;
        } else {
          _dtfm[n++] = 0.0;
          _dtfm[n++] = 0.0;
          _dtfm[n++] = 0.0;
        }
    }
  }
}

/* ----------------------------------------------------------------------
   perform half-step barostat scaling of velocities
-----------------------------------------------------------------------*/

void FixNHGPU::nh_v_press()
{
  if (pstyle == TRICLINIC || which == BIAS || _respa_on) {
    FixNH::nh_v_press();
    return;
  }

  dbl3_t * _noalias const v = (dbl3_t *)atom->v[0];
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double f0 = exp(-dt4*(omega_dot[0]+mtk_term2));
  double f1 = exp(-dt4*(omega_dot[1]+mtk_term2));
  double f2 = exp(-dt4*(omega_dot[2]+mtk_term2));
  f0 *= f0;
  f1 *= f1;
  f2 *= f2;

  if (igroup == 0) {
    #if (LAL_USE_OMP == 1) && (LAL_USE_OMP_SIMD == 1)
    #pragma omp parallel for simd schedule(static)
    #elif (LAL_USE_OMP_SIMD == 1)
    #pragma omp simd
    #endif
    for (int i = 0; i < nlocal; i++) {
      v[i].x *= f0;
      v[i].y *= f1;
      v[i].z *= f2;
    }
  } else {
    #if (LAL_USE_OMP == 1) && (LAL_USE_OMP_SIMD == 1)
    #pragma omp parallel for simd schedule(static)
    #elif (LAL_USE_OMP_SIMD == 1)
    #pragma omp simd
    #endif
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        v[i].x *= f0;
        v[i].y *= f1;
        v[i].z *= f2;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   perform half-step update of velocities
-----------------------------------------------------------------------*/

void FixNHGPU::nve_v()
{
  if (_respa_on) { FixNH::nve_v(); return; }

  double * _noalias const v = atom->v[0];
  const double * _noalias const f = atom->f[0];
  #if (LAL_USE_OMP == 1) && (LAL_USE_OMP_SIMD == 1)
  #pragma omp parallel for simd schedule(static)
  #elif (LAL_USE_OMP_SIMD == 1)
  #pragma omp simd
  #endif
  for (int i = 0; i < _nlocal3; i++)
    v[i] += _dtfm[i] * f[i];
}

/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

void FixNHGPU::nve_x()
{
  if (_respa_on) { FixNH::nve_x(); return; }

  double * _noalias const x = atom->x[0];
  double * _noalias const v = atom->v[0];

  // x update by full step only for atoms in group

  if (igroup == 0) {
    #if (LAL_USE_OMP == 1) && (LAL_USE_OMP_SIMD == 1)
    #pragma omp parallel for simd schedule(static)
    #elif (LAL_USE_OMP_SIMD == 1)
    #pragma omp simd
    #endif
    for (int i = 0; i < _nlocal3; i++)
      x[i] += dtv * v[i];
  } else {
    #if (LAL_USE_OMP == 1) && (LAL_USE_OMP_SIMD == 1)
    #pragma omp parallel for simd schedule(static)
    #elif (LAL_USE_OMP_SIMD == 1)
    #pragma omp simd
    #endif
    for (int i = 0; i < _nlocal3; i++) {
      if (_dtfm[i] != 0.0)
        x[i] += dtv * v[i];
    }
  }
}

/* ----------------------------------------------------------------------
   perform half-step thermostat scaling of velocities
-----------------------------------------------------------------------*/

void FixNHGPU::nh_v_temp()
{
  if (which == BIAS || _respa_on) {
    FixNH::nh_v_temp();
    return;
  }

  double * _noalias const v = atom->v[0];

  if (igroup == 0) {
    #if (LAL_USE_OMP == 1) && (LAL_USE_OMP_SIMD == 1)
    #pragma omp parallel for simd schedule(static)
    #elif (LAL_USE_OMP_SIMD == 1)
    #pragma omp simd
    #endif
    for (int i = 0; i < _nlocal3; i++)
        v[i] *= factor_eta;
  } else {
    #if (LAL_USE_OMP == 1) && (LAL_USE_OMP_SIMD == 1)
    #pragma omp parallel for simd schedule(static)
    #elif (LAL_USE_OMP_SIMD == 1)
    #pragma omp simd
    #endif
    for (int i = 0; i < _nlocal3; i++) {
      if (_dtfm[i] != 0.0)
        v[i] *= factor_eta;
    }
  }
}

double FixNHGPU::memory_usage()
{
  return FixNH::memory_usage() + _nlocal_max * 3 * sizeof(double);
}
