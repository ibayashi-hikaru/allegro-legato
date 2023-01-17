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
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "pppm_disp_tip4p_omp.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "suffix.h"

#include <cstring>
#include <cmath>

#include "omp_compat.h"
#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace LAMMPS_NS;
using namespace MathConst;

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#else
#define ZEROF 0.0
#endif

#define OFFSET 16384

/* ---------------------------------------------------------------------- */

PPPMDispTIP4POMP::PPPMDispTIP4POMP(LAMMPS *lmp) :
  PPPMDispTIP4P(lmp), ThrOMP(lmp, THR_KSPACE)
{
  triclinic_support = 0;
  tip4pflag = 1;
  suffix_flag |= Suffix::OMP;
}

/* ---------------------------------------------------------------------- */

PPPMDispTIP4POMP::~PPPMDispTIP4POMP()
{

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    if (function[0]) {
      ThrData * thr = fix->get_thr(tid);
      thr->init_pppm(-order,memory);
    }
    if (function[1] + function[2]) {
      ThrData * thr = fix->get_thr(tid);
      thr->init_pppm_disp(-order_6,memory);
    }
  }
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors and order
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::allocate()
{
  PPPMDispTIP4P::allocate();

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif

    if (function[0]) {
      ThrData *thr = fix->get_thr(tid);
      thr->init_pppm(order,memory);
    }
    if (function[1] + function[2]) {
      ThrData * thr = fix->get_thr(tid);
      thr->init_pppm_disp(order_6,memory);
    }
  }
}

/* ----------------------------------------------------------------------
   Compute the modified (hockney-eastwood) coulomb green function
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::compute_gf()
{
#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {

    double *prd;
    if (triclinic == 0) prd = domain->prd;
    else prd = domain->prd_lamda;

    double xprd = prd[0];
    double yprd = prd[1];
    double zprd = prd[2];
    double zprd_slab = zprd*slab_volfactor;

    double unitkx = (2.0*MY_PI/xprd);
    double unitky = (2.0*MY_PI/yprd);
    double unitkz = (2.0*MY_PI/zprd_slab);

    int tid,nn,nnfrom,nnto,k,l,m;
    int kper,lper,mper;
    double snx,sny,snz,snx2,sny2,snz2;
    double sqk;
    double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
    double numerator,denominator;

    const int nnx = nxhi_fft-nxlo_fft+1;
    const int nny = nyhi_fft-nylo_fft+1;

    loop_setup_thr(nnfrom, nnto, tid, nfft, comm->nthreads);

    for (m = nzlo_fft; m <= nzhi_fft; m++) {
      mper = m - nz_pppm*(2*m/nz_pppm);
      qz = unitkz*mper;
      snz = sin(0.5*qz*zprd_slab/nz_pppm);
      snz2 = snz*snz;
      sz = exp(-0.25*pow(qz/g_ewald,2.0));
      wz = 1.0;
      argz = 0.5*qz*zprd_slab/nz_pppm;
      if (argz != 0.0) wz = pow(sin(argz)/argz,order);
      wz *= wz;

      for (l = nylo_fft; l <= nyhi_fft; l++) {
        lper = l - ny_pppm*(2*l/ny_pppm);
        qy = unitky*lper;
        sny = sin(0.5*qy*yprd/ny_pppm);
        sny2 = sny*sny;
        sy = exp(-0.25*pow(qy/g_ewald,2.0));
        wy = 1.0;
        argy = 0.5*qy*yprd/ny_pppm;
        if (argy != 0.0) wy = pow(sin(argy)/argy,order);
        wy *= wy;

        for (k = nxlo_fft; k <= nxhi_fft; k++) {

          /* only compute the part designated to this thread */
          nn = k-nxlo_fft + nnx*(l-nylo_fft + nny*(m-nzlo_fft));
          if ((nn < nnfrom) || (nn >=nnto)) continue;

          kper = k - nx_pppm*(2*k/nx_pppm);
          qx = unitkx*kper;
          snx = sin(0.5*qx*xprd/nx_pppm);
          snx2 = snx*snx;
          sx = exp(-0.25*pow(qx/g_ewald,2.0));
          wx = 1.0;
          argx = 0.5*qx*xprd/nx_pppm;
          if (argx != 0.0) wx = pow(sin(argx)/argx,order);
          wx *= wx;

          sqk = pow(qx,2.0) + pow(qy,2.0) + pow(qz,2.0);

          if (sqk != 0.0) {
            numerator = 4.0*MY_PI/sqk;
            denominator = gf_denom(snx2,sny2,snz2, gf_b, order);
            greensfn[nn] = numerator*sx*sy*sz*wx*wy*wz/denominator;
          } else greensfn[nn] = 0.0;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Compyute the modified (hockney-eastwood) dispersion green function
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::compute_gf_6()
{
#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
    double *prd;
    int k,l,m,nn;

    // volume-dependent factors
    // adjust z dimension for 2d slab PPPM
    // z dimension for 3d PPPM is zprd since slab_volfactor = 1.0

    if (triclinic == 0) prd = domain->prd;
    else prd = domain->prd_lamda;

    double xprd = prd[0];
    double yprd = prd[1];
    double zprd = prd[2];
    double zprd_slab = zprd*slab_volfactor;

    double unitkx = (2.0*MY_PI/xprd);
    double unitky = (2.0*MY_PI/yprd);
    double unitkz = (2.0*MY_PI/zprd_slab);

    int kper,lper,mper;
    double sqk;
    double snx,sny,snz,snx2,sny2,snz2;
    double argx,argy,argz,wx,wy,wz,sx,sy,sz;
    double qx,qy,qz;
    double rtsqk, term;
    double numerator,denominator;
    double inv2ew = 2*g_ewald_6;
    inv2ew = 1/inv2ew;
    double rtpi = sqrt(MY_PI);
    int nnfrom, nnto, tid;

    numerator = -MY_PI*rtpi*g_ewald_6*g_ewald_6*g_ewald_6/(3.0);

    const int nnx = nxhi_fft_6-nxlo_fft_6+1;
    const int nny = nyhi_fft_6-nylo_fft_6+1;

    loop_setup_thr(nnfrom, nnto, tid, nfft_6, comm->nthreads);

    for (m = nzlo_fft_6; m <= nzhi_fft_6; m++) {
      mper = m - nz_pppm_6*(2*m/nz_pppm_6);
      qz = unitkz*mper;
      snz = sin(0.5*unitkz*mper*zprd_slab/nz_pppm_6);
      snz2 = snz*snz;
      sz = exp(-qz*qz*inv2ew*inv2ew);
      wz = 1.0;
      argz = 0.5*qz*zprd_slab/nz_pppm_6;
      if (argz != 0.0) wz = pow(sin(argz)/argz,order_6);
      wz *= wz;

      for (l = nylo_fft_6; l <= nyhi_fft_6; l++) {
        lper = l - ny_pppm_6*(2*l/ny_pppm_6);
        qy = unitky*lper;
        sny = sin(0.5*unitky*lper*yprd/ny_pppm_6);
        sny2 = sny*sny;
        sy = exp(-qy*qy*inv2ew*inv2ew);
        wy = 1.0;
        argy = 0.5*qy*yprd/ny_pppm_6;
        if (argy != 0.0) wy = pow(sin(argy)/argy,order_6);
        wy *= wy;

        for (k = nxlo_fft_6; k <= nxhi_fft_6; k++) {

          /* only compute the part designated to this thread */
          nn = k-nxlo_fft_6 + nnx*(l-nylo_fft_6 + nny*(m-nzlo_fft_6));
          if ((nn < nnfrom) || (nn >=nnto)) continue;

          kper = k - nx_pppm_6*(2*k/nx_pppm_6);
          qx = unitkx*kper;
          snx = sin(0.5*unitkx*kper*xprd/nx_pppm_6);
          snx2 = snx*snx;
          sx = exp(-qx*qx*inv2ew*inv2ew);
          wx = 1.0;
          argx = 0.5*qx*xprd/nx_pppm_6;
          if (argx != 0.0) wx = pow(sin(argx)/argx,order_6);
          wx *= wx;

          sqk = pow(qx,2.0) + pow(qy,2.0) + pow(qz,2.0);

          denominator = gf_denom(snx2,sny2,snz2, gf_b_6, order_6);
          rtsqk = sqrt(sqk);
          term = (1-2*sqk*inv2ew*inv2ew)*sx*sy*sz +
                  2*sqk*rtsqk*inv2ew*inv2ew*inv2ew*rtpi*erfc(rtsqk*inv2ew);
          greensfn_6[nn] = numerator*term*wx*wy*wz/denominator;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   run the regular toplevel compute method from plain PPPM
   which will have individual methods replaced by our threaded
   versions and then call the obligatory force reduction.
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::compute(int eflag, int vflag)
{

  PPPMDispTIP4P::compute(eflag,vflag);

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    ThrData *thr = fix->get_thr(tid);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

/* ----------------------------------------------------------------------
   find center grid pt for each of my particles
   check that full stencil for the particle will fit in my 3d brick
   store central grid pt indices in part2grid array
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::particle_map_c(double dxinv, double dyinv,
                                      double dzinv, double sft,
                                      int ** part2grid, int nup,
                                      int nlw, int nxlo_o,
                                      int nylo_o, int nzlo_o,
                                      int nxhi_o, int nyhi_o,
                                      int nzhi_o)
{
  // no local atoms => nothing to do
  if (atom->nlocal == 0) return;

  const int * _noalias const type = atom->type;
  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  int3_t * _noalias const p2g = (int3_t *) part2grid[0];
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];
  const int nlocal = atom->nlocal;

  const double delxinv = dxinv;
  const double delyinv = dyinv;
  const double delzinv = dzinv;
  const double shift = sft;
  const int nupper = nup;
  const int nlower = nlw;
  const int nxlo_out = nxlo_o;
  const int nylo_out = nylo_o;
  const int nzlo_out = nzlo_o;
  const int nxhi_out = nxhi_o;
  const int nyhi_out = nyhi_o;
  const int nzhi_out = nzhi_o;

  if (!std::isfinite(boxlo[0]) || !std::isfinite(boxlo[1]) || !std::isfinite(boxlo[2]))
    error->one(FLERR,"Non-numeric box dimensions - simulation unstable");

  int flag = 0;
#if defined(_OPENMP)
#pragma omp parallel for LMP_DEFAULT_NONE reduction(+:flag) schedule(static)
#endif
  for (int i = 0; i < nlocal; i++) {
    dbl3_t xM;
    int iH1,iH2;

    if (type[i] == typeO) {
      find_M_thr(i,iH1,iH2,xM);
    } else {
      xM = x[i];
    }

    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // current particle coord can be outside global and local box
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

    const int nx = static_cast<int> ((xM.x-boxlox)*delxinv+shift) - OFFSET;
    const int ny = static_cast<int> ((xM.y-boxloy)*delyinv+shift) - OFFSET;
    const int nz = static_cast<int> ((xM.z-boxloz)*delzinv+shift) - OFFSET;

    p2g[i].a = nx;
    p2g[i].b = ny;
    p2g[i].t = nz;

    // check that entire stencil around nx,ny,nz will fit in my 3d brick

    if (nx+nlower < nxlo_out || nx+nupper > nxhi_out ||
        ny+nlower < nylo_out || ny+nupper > nyhi_out ||
        nz+nlower < nzlo_out || nz+nupper > nzhi_out)
      flag++;
  }

  int flag_all;
  MPI_Allreduce(&flag,&flag_all,1,MPI_INT,MPI_SUM,world);
  if (flag_all) error->all(FLERR,"Out of range atoms - cannot compute PPPM");
}

/* ----------------------------------------------------------------------
   find center grid pt for each of my particles
   check that full stencil for the particle will fit in my 3d brick
   store central grid pt indices in part2grid array
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::particle_map(double dxinv, double dyinv,
                               double dzinv, double sft,
                               int ** part2grid, int nup,
                               int nlw, int nxlo_o,
                               int nylo_o, int nzlo_o,
                               int nxhi_o, int nyhi_o,
                               int nzhi_o)
{
  // no local atoms => nothing to do
  if (atom->nlocal == 0) return;

  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  int3_t * _noalias const p2g = (int3_t *) part2grid[0];
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];
  const int nlocal = atom->nlocal;

  const double delxinv = dxinv;
  const double delyinv = dyinv;
  const double delzinv = dzinv;
  const double shift = sft;
  const int nupper = nup;
  const int nlower = nlw;
  const int nxlo_out = nxlo_o;
  const int nylo_out = nylo_o;
  const int nzlo_out = nzlo_o;
  const int nxhi_out = nxhi_o;
  const int nyhi_out = nyhi_o;
  const int nzhi_out = nzhi_o;

  int flag = 0;
#if defined(_OPENMP)
#pragma omp parallel for LMP_DEFAULT_NONE reduction(+:flag) schedule(static)
#endif
  for (int i = 0; i < nlocal; i++) {

    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // current particle coord can be outside global and local box
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

    const int nx = static_cast<int> ((x[i].x-boxlox)*delxinv+shift) - OFFSET;
    const int ny = static_cast<int> ((x[i].y-boxloy)*delyinv+shift) - OFFSET;
    const int nz = static_cast<int> ((x[i].z-boxloz)*delzinv+shift) - OFFSET;

    p2g[i].a = nx;
    p2g[i].b = ny;
    p2g[i].t = nz;

    // check that entire stencil around nx,ny,nz will fit in my 3d brick

    if (nx+nlower < nxlo_out || nx+nupper > nxhi_out ||
        ny+nlower < nylo_out || ny+nupper > nyhi_out ||
        nz+nlower < nzlo_out || nz+nupper > nzhi_out)
      flag++;
  }

  int flag_all;
  MPI_Allreduce(&flag,&flag_all,1,MPI_INT,MPI_SUM,world);
  if (flag_all) error->all(FLERR,"Out of range atoms - cannot compute PPPM");
}

/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::make_rho_c()
{

  // clear 3d density array

  FFT_SCALAR * _noalias const d = &(density_brick[nzlo_out][nylo_out][nxlo_out]);
  memset(d,0,ngrid*sizeof(FFT_SCALAR));

  // no local atoms => nothing else to do

  const int nlocal = atom->nlocal;
  if (nlocal == 0) return;

  const int ix = nxhi_out - nxlo_out + 1;
  const int iy = nyhi_out - nylo_out + 1;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
    const double * _noalias const q = atom->q;
    const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
    const int3_t * _noalias const p2g = (int3_t *) part2grid[0];
    const int * _noalias const type = atom->type;
    dbl3_t xM;

    const double boxlox = boxlo[0];
    const double boxloy = boxlo[1];
    const double boxloz = boxlo[2];

    // determine range of grid points handled by this thread
    int i,jfrom,jto,tid,iH1,iH2;
    loop_setup_thr(jfrom,jto,tid,ngrid,comm->nthreads);

    // get per thread data
    ThrData *thr = fix->get_thr(tid);
    FFT_SCALAR * const * const r1d = static_cast<FFT_SCALAR **>(thr->get_rho1d());

    // loop over my charges, add their contribution to nearby grid points
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // (dx,dy,dz) = distance to "lower left" grid pt

    // loop over all local atoms for all threads
    for (i = 0; i < nlocal; i++) {

      const int nx = p2g[i].a;
      const int ny = p2g[i].b;
      const int nz = p2g[i].t;

      // pre-screen whether this atom will ever come within
      // reach of the data segement this thread is updating.
      if ( ((nz+nlower-nzlo_out)*ix*iy >= jto)
           || ((nz+nupper-nzlo_out+1)*ix*iy < jfrom) ) continue;

      if (type[i] == typeO) {
        find_M_thr(i,iH1,iH2,xM);
      } else {
        xM = x[i];
      }
      const FFT_SCALAR dx = nx+shiftone - (xM.x-boxlox)*delxinv;
      const FFT_SCALAR dy = ny+shiftone - (xM.y-boxloy)*delyinv;
      const FFT_SCALAR dz = nz+shiftone - (xM.z-boxloz)*delzinv;

      compute_rho1d_thr(r1d,dx,dy,dz,order,rho_coeff);

      const FFT_SCALAR z0 = delvolinv * q[i];

      for (int n = nlower; n <= nupper; ++n) {
        const int jn = (nz+n-nzlo_out)*ix*iy;
        const FFT_SCALAR y0 = z0*r1d[2][n];

        for (int m = nlower; m <= nupper; ++m) {
          const int jm = jn+(ny+m-nylo_out)*ix;
          const FFT_SCALAR x0 = y0*r1d[1][m];

          for (int l = nlower; l <= nupper; ++l) {
            const int jl = jm+nx+l-nxlo_out;
            // make sure each thread only updates
            // "his" elements of the density grid
            if (jl >= jto) break;
            if (jl < jfrom) continue;

            d[jl] += x0*r1d[0][l];
          }
        }
      }
    }
  }
}


/* ----------------------------------------------------------------------
   same as above for dispersion interaction with geometric mixing rule
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::make_rho_g()
{

  // clear 3d density array

  FFT_SCALAR * _noalias const d = &(density_brick_g[nzlo_out_6][nylo_out_6][nxlo_out_6]);
  memset(d,0,ngrid_6*sizeof(FFT_SCALAR));

  // no local atoms => nothing else to do

  const int nlocal = atom->nlocal;
  if (nlocal == 0) return;

  const int ix = nxhi_out_6 - nxlo_out_6 + 1;
  const int iy = nyhi_out_6 - nylo_out_6 + 1;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
    const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
    const int3_t * _noalias const p2g = (int3_t *) part2grid_6[0];

    const double boxlox = boxlo[0];
    const double boxloy = boxlo[1];
    const double boxloz = boxlo[2];

    // determine range of grid points handled by this thread
    int i,jfrom,jto,tid;
    loop_setup_thr(jfrom,jto,tid,ngrid_6,comm->nthreads);

    // get per thread data
    ThrData *thr = fix->get_thr(tid);
    FFT_SCALAR * const * const r1d = static_cast<FFT_SCALAR **>(thr->get_rho1d_6());

    // loop over my charges, add their contribution to nearby grid points
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // (dx,dy,dz) = distance to "lower left" grid pt

    // loop over all local atoms for all threads
    for (i = 0; i < nlocal; i++) {

      const int nx = p2g[i].a;
      const int ny = p2g[i].b;
      const int nz = p2g[i].t;

      // pre-screen whether this atom will ever come within
      // reach of the data segement this thread is updating.
      if ( ((nz+nlower_6-nzlo_out_6)*ix*iy >= jto)
           || ((nz+nupper_6-nzlo_out_6+1)*ix*iy < jfrom) ) continue;

      const FFT_SCALAR dx = nx+shiftone_6 - (x[i].x-boxlox)*delxinv_6;
      const FFT_SCALAR dy = ny+shiftone_6 - (x[i].y-boxloy)*delyinv_6;
      const FFT_SCALAR dz = nz+shiftone_6 - (x[i].z-boxloz)*delzinv_6;

      compute_rho1d_thr(r1d,dx,dy,dz,order_6,rho_coeff_6);

      const int type = atom->type[i];
      const double lj = B[type];
      const FFT_SCALAR z0 = delvolinv_6 * lj;

      for (int n = nlower_6; n <= nupper_6; ++n) {
        const int jn = (nz+n-nzlo_out_6)*ix*iy;
        const FFT_SCALAR y0 = z0*r1d[2][n];

        for (int m = nlower_6; m <= nupper_6; ++m) {
          const int jm = jn+(ny+m-nylo_out_6)*ix;
          const FFT_SCALAR x0 = y0*r1d[1][m];

          for (int l = nlower_6; l <= nupper_6; ++l) {
            const int jl = jm+nx+l-nxlo_out_6;
            // make sure each thread only updates
            // "his" elements of the density grid
            if (jl >= jto) break;
            if (jl < jfrom) continue;

            d[jl] += x0*r1d[0][l];
          }
        }
      }
    }
  }
}


/* ----------------------------------------------------------------------
   same as above for dispersion interaction with arithmetic mixing rule
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::make_rho_a()
{

  // clear 3d density array

  FFT_SCALAR * _noalias const d0 = &(density_brick_a0[nzlo_out_6][nylo_out_6][nxlo_out_6]);
  FFT_SCALAR * _noalias const d1 = &(density_brick_a1[nzlo_out_6][nylo_out_6][nxlo_out_6]);
  FFT_SCALAR * _noalias const d2 = &(density_brick_a2[nzlo_out_6][nylo_out_6][nxlo_out_6]);
  FFT_SCALAR * _noalias const d3 = &(density_brick_a3[nzlo_out_6][nylo_out_6][nxlo_out_6]);
  FFT_SCALAR * _noalias const d4 = &(density_brick_a4[nzlo_out_6][nylo_out_6][nxlo_out_6]);
  FFT_SCALAR * _noalias const d5 = &(density_brick_a5[nzlo_out_6][nylo_out_6][nxlo_out_6]);
  FFT_SCALAR * _noalias const d6 = &(density_brick_a6[nzlo_out_6][nylo_out_6][nxlo_out_6]);

  memset(d0,0,ngrid_6*sizeof(FFT_SCALAR));
  memset(d1,0,ngrid_6*sizeof(FFT_SCALAR));
  memset(d2,0,ngrid_6*sizeof(FFT_SCALAR));
  memset(d3,0,ngrid_6*sizeof(FFT_SCALAR));
  memset(d4,0,ngrid_6*sizeof(FFT_SCALAR));
  memset(d5,0,ngrid_6*sizeof(FFT_SCALAR));
  memset(d6,0,ngrid_6*sizeof(FFT_SCALAR));

  // no local atoms => nothing else to do

  const int nlocal = atom->nlocal;
  if (nlocal == 0) return;

  const int ix = nxhi_out_6 - nxlo_out_6 + 1;
  const int iy = nyhi_out_6 - nylo_out_6 + 1;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
    const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
    const int3_t * _noalias const p2g = (int3_t *) part2grid_6[0];

    const double boxlox = boxlo[0];
    const double boxloy = boxlo[1];
    const double boxloz = boxlo[2];

    // determine range of grid points handled by this thread
    int i,jfrom,jto,tid;
    loop_setup_thr(jfrom,jto,tid,ngrid_6,comm->nthreads);

    // get per thread data
    ThrData *thr = fix->get_thr(tid);
    FFT_SCALAR * const * const r1d = static_cast<FFT_SCALAR **>(thr->get_rho1d_6());

    // loop over my charges, add their contribution to nearby grid points
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // (dx,dy,dz) = distance to "lower left" grid pt

    // loop over all local atoms for all threads
    for (i = 0; i < nlocal; i++) {

      const int nx = p2g[i].a;
      const int ny = p2g[i].b;
      const int nz = p2g[i].t;

      // pre-screen whether this atom will ever come within
      // reach of the data segement this thread is updating.
      if ( ((nz+nlower_6-nzlo_out_6)*ix*iy >= jto)
           || ((nz+nupper_6-nzlo_out_6+1)*ix*iy < jfrom) ) continue;

      const FFT_SCALAR dx = nx+shiftone_6 - (x[i].x-boxlox)*delxinv_6;
      const FFT_SCALAR dy = ny+shiftone_6 - (x[i].y-boxloy)*delyinv_6;
      const FFT_SCALAR dz = nz+shiftone_6 - (x[i].z-boxloz)*delzinv_6;

      compute_rho1d_thr(r1d,dx,dy,dz,order_6,rho_coeff_6);

      const int type = atom->type[i];
      const double lj0 = B[7*type];
      const double lj1 = B[7*type+1];
      const double lj2 = B[7*type+2];
      const double lj3 = B[7*type+3];
      const double lj4 = B[7*type+4];
      const double lj5 = B[7*type+5];
      const double lj6 = B[7*type+6];

      const FFT_SCALAR z0 = delvolinv_6;

      for (int n = nlower_6; n <= nupper_6; ++n) {
        const int jn = (nz+n-nzlo_out_6)*ix*iy;
        const FFT_SCALAR y0 = z0*r1d[2][n];

        for (int m = nlower_6; m <= nupper_6; ++m) {
          const int jm = jn+(ny+m-nylo_out_6)*ix;
          const FFT_SCALAR x0 = y0*r1d[1][m];

          for (int l = nlower_6; l <= nupper_6; ++l) {
            const int jl = jm+nx+l-nxlo_out_6;
            // make sure each thread only updates
            // "his" elements of the density grid
            if (jl >= jto) break;
            if (jl < jfrom) continue;

            const double w = x0*r1d[0][l];

            d0[jl] += w*lj0;
            d1[jl] += w*lj1;
            d2[jl] += w*lj2;
            d3[jl] += w*lj3;
            d4[jl] += w*lj4;
            d5[jl] += w*lj5;
            d6[jl] += w*lj6;
          }
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ik
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::fieldforce_c_ik()
{
  const int nlocal = atom->nlocal;

  // no local atoms => nothing to do

  if (nlocal == 0) return;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  const double * _noalias const q = atom->q;
  const int3_t * _noalias const p2g = (int3_t *) part2grid[0];
  const int * _noalias const type = atom->type;

  const double qqrd2e = force->qqrd2e;
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
    dbl3_t xM;
    FFT_SCALAR x0,y0,z0,ekx,eky,ekz;
    int i,ifrom,ito,tid,iH1,iH2,l,m,n,mx,my,mz;

    loop_setup_thr(ifrom,ito,tid,nlocal,comm->nthreads);

    // get per thread data
    ThrData *thr = fix->get_thr(tid);
    dbl3_t * _noalias const f = (dbl3_t *) thr->get_f()[0];
    FFT_SCALAR * const * const r1d = static_cast<FFT_SCALAR **>(thr->get_rho1d());

    for (i = ifrom; i < ito; ++i) {
      if (type[i] == typeO) {
        find_M_thr(i,iH1,iH2,xM);
      } else xM = x[i];

      const int nx = p2g[i].a;
      const int ny = p2g[i].b;
      const int nz = p2g[i].t;
      const FFT_SCALAR dx = nx+shiftone - (xM.x-boxlox)*delxinv;
      const FFT_SCALAR dy = ny+shiftone - (xM.y-boxloy)*delyinv;
      const FFT_SCALAR dz = nz+shiftone - (xM.z-boxloz)*delzinv;

      compute_rho1d_thr(r1d,dx,dy,dz, order, rho_coeff);

      ekx = eky = ekz = ZEROF;
      for (n = nlower; n <= nupper; n++) {
        mz = n+nz;
        z0 = r1d[2][n];
        for (m = nlower; m <= nupper; m++) {
          my = m+ny;
          y0 = z0*r1d[1][m];
          for (l = nlower; l <= nupper; l++) {
            mx = l+nx;
            x0 = y0*r1d[0][l];
            ekx -= x0*vdx_brick[mz][my][mx];
            eky -= x0*vdy_brick[mz][my][mx];
            ekz -= x0*vdz_brick[mz][my][mx];
          }
        }
      }

      // convert E-field to force

      const double qfactor = qqrd2e * scale * q[i];
      if (type[i] != typeO) {
        f[i].x += qfactor*ekx;
        f[i].y += qfactor*eky;
        if (slabflag != 2) f[i].z += qfactor*ekz;

      } else {
        const double fx = qfactor * ekx;
        const double fy = qfactor * eky;
        const double fz = qfactor * ekz;

        f[i].x += fx*(1 - alpha);
        f[i].y += fy*(1 - alpha);
        if (slabflag != 2) f[i].z += fz*(1 - alpha);

        f[iH1].x += 0.5*alpha*fx;
        f[iH1].y += 0.5*alpha*fy;
        if (slabflag != 2) f[iH1].z += 0.5*alpha*fz;

        f[iH2].x += 0.5*alpha*fx;
        f[iH2].y += 0.5*alpha*fy;
        if (slabflag != 2) f[iH2].z += 0.5*alpha*fz;
      }
    }
  } // end of parallel region
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ad
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::fieldforce_c_ad()
{
  const int nlocal = atom->nlocal;

  // no local atoms => nothing to do

  if (nlocal == 0) return;

  const double *prd = (triclinic == 0) ? domain->prd : domain->prd_lamda;
  const double hx_inv = nx_pppm/prd[0];
  const double hy_inv = ny_pppm/prd[1];
  const double hz_inv = nz_pppm/prd[2];

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  const double * _noalias const q = atom->q;
  const int3_t * _noalias const p2g = (int3_t *) part2grid[0];
  const int * _noalias const type = atom->type;

  const double qqrd2e = force->qqrd2e;
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
    double s1,s2,s3,sf;
    dbl3_t xM;
    FFT_SCALAR ekx,eky,ekz;
    int i,ifrom,ito,tid,iH1,iH2,l,m,n,mx,my,mz;

    loop_setup_thr(ifrom,ito,tid,nlocal,comm->nthreads);

    // get per thread data
    ThrData *thr = fix->get_thr(tid);
    dbl3_t * _noalias const f = (dbl3_t *) thr->get_f()[0];
    FFT_SCALAR * const * const r1d = static_cast<FFT_SCALAR **>(thr->get_rho1d());
    FFT_SCALAR * const * const d1d = static_cast<FFT_SCALAR **>(thr->get_drho1d());

    for (i = ifrom; i < ito; ++i) {
      if (type[i] == typeO) {
        find_M_thr(i,iH1,iH2,xM);
      } else xM = x[i];

      const int nx = p2g[i].a;
      const int ny = p2g[i].b;
      const int nz = p2g[i].t;
      const FFT_SCALAR dx = nx+shiftone - (xM.x-boxlox)*delxinv;
      const FFT_SCALAR dy = ny+shiftone - (xM.y-boxloy)*delyinv;
      const FFT_SCALAR dz = nz+shiftone - (xM.z-boxloz)*delzinv;

      compute_rho1d_thr(r1d,dx,dy,dz,order,rho_coeff);
      compute_drho1d_thr(d1d,dx,dy,dz,order,drho_coeff);

      ekx = eky = ekz = ZEROF;
      for (n = nlower; n <= nupper; n++) {
        mz = n+nz;
        for (m = nlower; m <= nupper; m++) {
          my = m+ny;
          for (l = nlower; l <= nupper; l++) {
            mx = l+nx;
            ekx += d1d[0][l]*r1d[1][m]*r1d[2][n]*u_brick[mz][my][mx];
            eky += r1d[0][l]*d1d[1][m]*r1d[2][n]*u_brick[mz][my][mx];
            ekz += r1d[0][l]*r1d[1][m]*d1d[2][n]*u_brick[mz][my][mx];
          }
        }
      }
      ekx *= hx_inv;
      eky *= hy_inv;
      ekz *= hz_inv;

      // convert E-field to force and subtract self forces

      const double qi = q[i];
      const double qfactor = qqrd2e * scale * qi;

      s1 = x[i].x*hx_inv;
      sf = sf_coeff[0]*sin(MY_2PI*s1);
      sf += sf_coeff[1]*sin(MY_4PI*s1);
      sf *= 2.0*qi;
      const double fx = qfactor*(ekx - sf);

      s2 = x[i].y*hy_inv;
      sf = sf_coeff[2]*sin(MY_2PI*s2);
      sf += sf_coeff[3]*sin(MY_4PI*s2);
      sf *= 2.0*qi;
      const double fy = qfactor*(eky - sf);

      s3 = x[i].z*hz_inv;
      sf = sf_coeff[4]*sin(MY_2PI*s3);
      sf += sf_coeff[5]*sin(MY_4PI*s3);
      sf *= 2.0*qi;
      const double fz = qfactor*(ekz - sf);

      if (type[i] != typeO) {
        f[i].x += fx;
        f[i].y += fy;
        if (slabflag != 2) f[i].z += fz;

      } else {
        f[i].x += fx*(1 - alpha);
        f[i].y += fy*(1 - alpha);
        if (slabflag != 2) f[i].z += fz*(1 - alpha);

        f[iH1].x += 0.5*alpha*fx;
        f[iH1].y += 0.5*alpha*fy;
        if (slabflag != 2) f[iH1].z += 0.5*alpha*fz;

        f[iH2].x += 0.5*alpha*fx;
        f[iH2].y += 0.5*alpha*fy;
        if (slabflag != 2) f[iH2].z += 0.5*alpha*fz;
      }
    }
  } // end of parallel region
}

/* ----------------------------------------------------------------------
   interpolate from grid to get dispersion field & force on my particles
   for ik scheme and geometric mixing rule
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::fieldforce_g_ik()
{
  const int nlocal = atom->nlocal;

  // no local atoms => nothing to do

  if (nlocal == 0) return;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  const double * const * const x = atom->x;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int tid = omp_get_thread_num();
    const int inum = nlocal;
    const int idelta = 1 + inum/comm->nthreads;
    const int ifrom = tid*idelta;
    const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
#else
    const int ifrom = 0;
    const int ito = nlocal;
    const int tid = 0;
#endif
    ThrData *thr = fix->get_thr(tid);
    double * const * const f = thr->get_f();
    FFT_SCALAR * const * const r1d =  static_cast<FFT_SCALAR **>(thr->get_rho1d_6());

    int l,m,n,nx,ny,nz,mx,my,mz;
    FFT_SCALAR dx,dy,dz,x0,y0,z0;
    FFT_SCALAR ekx,eky,ekz;
    int type;
    double lj;

    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) {
      for (int i = ifrom; i < ito; i++) {

        nx = part2grid_6[i][0];
        ny = part2grid_6[i][1];
        nz = part2grid_6[i][2];
        dx = nx+shiftone_6 - (x[i][0]-boxlo[0])*delxinv_6;
        dy = ny+shiftone_6 - (x[i][1]-boxlo[1])*delyinv_6;
        dz = nz+shiftone_6 - (x[i][2]-boxlo[2])*delzinv_6;

        compute_rho1d_thr(r1d,dx,dy,dz, order_6, rho_coeff_6);

        ekx = eky = ekz = ZEROF;
        for (n = nlower_6; n <= nupper_6; n++) {
          mz = n+nz;
          z0 = r1d[2][n];
          for (m = nlower_6; m <= nupper_6; m++) {
            my = m+ny;
            y0 = z0*r1d[1][m];
            for (l = nlower_6; l <= nupper_6; l++) {
              mx = l+nx;
              x0 = y0*r1d[0][l];
              ekx -= x0*vdx_brick_g[mz][my][mx];
              eky -= x0*vdy_brick_g[mz][my][mx];
              ekz -= x0*vdz_brick_g[mz][my][mx];
            }
          }
        }

        // convert E-field to force
        type = atom->type[i];
        lj = B[type];
        f[i][0] += lj*ekx;
        f[i][1] += lj*eky;
        f[i][2] += lj*ekz;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get dispersion field & force on my particles
   for ad scheme and geometric mixing rule
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::fieldforce_g_ad()
{
  const int nlocal = atom->nlocal;

  // no local atoms => nothing to do

  if (nlocal == 0) return;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  const double * const * const x = atom->x;
  double *prd;

  if (triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;

  const double hx_inv = nx_pppm_6/xprd;
  const double hy_inv = ny_pppm_6/yprd;
  const double hz_inv = nz_pppm_6/zprd_slab;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int tid = omp_get_thread_num();
    const int inum = nlocal;
    const int idelta = 1 + inum/comm->nthreads;
    const int ifrom = tid*idelta;
    const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
#else
    const int ifrom = 0;
    const int ito = nlocal;
    const int tid = 0;
#endif
    ThrData *thr = fix->get_thr(tid);
    double * const * const f = thr->get_f();
    FFT_SCALAR * const * const r1d =  static_cast<FFT_SCALAR **>(thr->get_rho1d_6());
    FFT_SCALAR * const * const dr1d = static_cast<FFT_SCALAR **>(thr->get_drho1d_6());

    int l,m,n,nx,ny,nz,mx,my,mz;
    FFT_SCALAR dx,dy,dz;
    FFT_SCALAR ekx,eky,ekz;
    int type;
    double lj;
    double sf = 0.0;
    double s1,s2,s3;

    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) {
      for (int i = ifrom; i < ito; i++) {

        nx = part2grid_6[i][0];
        ny = part2grid_6[i][1];
        nz = part2grid_6[i][2];
        dx = nx+shiftone_6 - (x[i][0]-boxlo[0])*delxinv_6;
        dy = ny+shiftone_6 - (x[i][1]-boxlo[1])*delyinv_6;
        dz = nz+shiftone_6 - (x[i][2]-boxlo[2])*delzinv_6;

        compute_rho1d_thr(r1d,dx,dy,dz, order_6, rho_coeff_6);
        compute_drho1d_thr(dr1d,dx,dy,dz, order_6, drho_coeff_6);

        ekx = eky = ekz = ZEROF;
        for (n = nlower_6; n <= nupper_6; n++) {
          mz = n+nz;
          for (m = nlower_6; m <= nupper_6; m++) {
            my = m+ny;
            for (l = nlower_6; l <= nupper_6; l++) {
              mx = l+nx;
              ekx += dr1d[0][l]*r1d[1][m]*r1d[2][n]*u_brick_g[mz][my][mx];
              eky += r1d[0][l]*dr1d[1][m]*r1d[2][n]*u_brick_g[mz][my][mx];
              ekz += r1d[0][l]*r1d[1][m]*dr1d[2][n]*u_brick_g[mz][my][mx];
            }
          }
        }
        ekx *= hx_inv;
        eky *= hy_inv;
        ekz *= hz_inv;

        // convert E-field to force
        type = atom->type[i];
        lj = B[type];

        s1 = x[i][0]*hx_inv;
        s2 = x[i][1]*hy_inv;
        s3 = x[i][2]*hz_inv;

        sf = sf_coeff_6[0]*sin(2*MY_PI*s1);
        sf += sf_coeff_6[1]*sin(4*MY_PI*s1);
        sf *= 2*lj*lj;
        f[i][0] += ekx*lj - sf;

        sf = sf_coeff_6[2]*sin(2*MY_PI*s2);
        sf += sf_coeff_6[3]*sin(4*MY_PI*s2);
        sf *= 2*lj*lj;
        f[i][1] += eky*lj - sf;

        sf = sf_coeff_6[4]*sin(2*MY_PI*s3);
        sf += sf_coeff_6[5]*sin(4*MY_PI*s3);
        sf *= 2*lj*lj;
        if (slabflag != 2) f[i][2] += ekz*lj - sf;
      }
    }
  }
}

/* ----------------------------------------------------------------------
 interpolate from grid to get per-atom energy/virial for dispersion
 interaction and geometric mixing rule
 ------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::fieldforce_g_peratom()
{
  const int nlocal = atom->nlocal;

  // no local atoms => nothing to do

  if (nlocal == 0) return;

  // loop over my charges, interpolate from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  const double * const * const x = atom->x;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int tid = omp_get_thread_num();
    const int inum = nlocal;
    const int idelta = 1 + inum/comm->nthreads;
    const int ifrom = tid*idelta;
    const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
#else
    const int ifrom = 0;
    const int ito = nlocal;
    const int tid = 0;
#endif
    ThrData *thr = fix->get_thr(tid);
    FFT_SCALAR * const * const r1d =  static_cast<FFT_SCALAR **>(thr->get_rho1d_6());

    int l,m,n,nx,ny,nz,mx,my,mz;
    FFT_SCALAR dx,dy,dz,x0,y0,z0;
    FFT_SCALAR u,v0,v1,v2,v3,v4,v5;
    int type;
    double lj;

    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) {
      for (int i = ifrom; i < ito; i++) {

        nx = part2grid_6[i][0];
        ny = part2grid_6[i][1];
        nz = part2grid_6[i][2];
        dx = nx+shiftone_6 - (x[i][0]-boxlo[0])*delxinv_6;
        dy = ny+shiftone_6 - (x[i][1]-boxlo[1])*delyinv_6;
        dz = nz+shiftone_6 - (x[i][2]-boxlo[2])*delzinv_6;

        compute_rho1d_thr(r1d,dx,dy,dz, order_6, rho_coeff_6);

        u = v0 = v1 = v2 = v3 = v4 = v5 = ZEROF;
        for (n = nlower_6; n <= nupper_6; n++) {
          mz = n+nz;
          z0 = r1d[2][n];
          for (m = nlower_6; m <= nupper_6; m++) {
            my = m+ny;
            y0 = z0*r1d[1][m];
            for (l = nlower_6; l <= nupper_6; l++) {
              mx = l+nx;
              x0 = y0*r1d[0][l];
              if (eflag_atom) u += x0*u_brick_g[mz][my][mx];
              if (vflag_atom) {
                v0 += x0*v0_brick_g[mz][my][mx];
                v1 += x0*v1_brick_g[mz][my][mx];
                v2 += x0*v2_brick_g[mz][my][mx];
                v3 += x0*v3_brick_g[mz][my][mx];
                v4 += x0*v4_brick_g[mz][my][mx];
                v5 += x0*v5_brick_g[mz][my][mx];
              }
            }
          }
        }

        type = atom->type[i];
        lj = B[type]*0.5;

        if (eflag_atom) eatom[i] += u*lj;
        if (vflag_atom) {
          vatom[i][0] += v0*lj;
          vatom[i][1] += v1*lj;
          vatom[i][2] += v2*lj;
          vatom[i][3] += v3*lj;
          vatom[i][4] += v4*lj;
          vatom[i][5] += v5*lj;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get dispersion field & force on my particles
   for ik scheme and arithmetic mixing rule
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::fieldforce_a_ik()
{
  const int nlocal = atom->nlocal;

  // no local atoms => nothing to do

  if (nlocal == 0) return;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  const double * const * const x = atom->x;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int tid = omp_get_thread_num();
    const int inum = nlocal;
    const int idelta = 1 + inum/comm->nthreads;
    const int ifrom = tid*idelta;
    const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
#else
    const int ifrom = 0;
    const int ito = nlocal;
    const int tid = 0;
#endif
    ThrData *thr = fix->get_thr(tid);
    double * const * const f = thr->get_f();
    FFT_SCALAR * const * const r1d =  static_cast<FFT_SCALAR **>(thr->get_rho1d_6());

    int l,m,n,nx,ny,nz,mx,my,mz;
    FFT_SCALAR dx,dy,dz,x0,y0,z0;
    FFT_SCALAR ekx0, eky0, ekz0, ekx1, eky1, ekz1, ekx2, eky2, ekz2;
    FFT_SCALAR ekx3, eky3, ekz3, ekx4, eky4, ekz4, ekx5, eky5, ekz5;
    FFT_SCALAR ekx6, eky6, ekz6;
    int type;
    double lj0,lj1,lj2,lj3,lj4,lj5,lj6;

    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) {
      for (int i = ifrom; i < ito; i++) {

        nx = part2grid_6[i][0];
        ny = part2grid_6[i][1];
        nz = part2grid_6[i][2];
        dx = nx+shiftone_6 - (x[i][0]-boxlo[0])*delxinv_6;
        dy = ny+shiftone_6 - (x[i][1]-boxlo[1])*delyinv_6;
        dz = nz+shiftone_6 - (x[i][2]-boxlo[2])*delzinv_6;

        compute_rho1d_thr(r1d,dx,dy,dz, order_6, rho_coeff_6);

        ekx0 = eky0 = ekz0 = ZEROF;
        ekx1 = eky1 = ekz1 = ZEROF;
        ekx2 = eky2 = ekz2 = ZEROF;
        ekx3 = eky3 = ekz3 = ZEROF;
        ekx4 = eky4 = ekz4 = ZEROF;
        ekx5 = eky5 = ekz5 = ZEROF;
        ekx6 = eky6 = ekz6 = ZEROF;
        for (n = nlower_6; n <= nupper_6; n++) {
          mz = n+nz;
          z0 = r1d[2][n];
          for (m = nlower_6; m <= nupper_6; m++) {
            my = m+ny;
            y0 = z0*r1d[1][m];
            for (l = nlower_6; l <= nupper_6; l++) {
              mx = l+nx;
              x0 = y0*r1d[0][l];
              ekx0 -= x0*vdx_brick_a0[mz][my][mx];
              eky0 -= x0*vdy_brick_a0[mz][my][mx];
              ekz0 -= x0*vdz_brick_a0[mz][my][mx];
              ekx1 -= x0*vdx_brick_a1[mz][my][mx];
              eky1 -= x0*vdy_brick_a1[mz][my][mx];
              ekz1 -= x0*vdz_brick_a1[mz][my][mx];
              ekx2 -= x0*vdx_brick_a2[mz][my][mx];
              eky2 -= x0*vdy_brick_a2[mz][my][mx];
              ekz2 -= x0*vdz_brick_a2[mz][my][mx];
              ekx3 -= x0*vdx_brick_a3[mz][my][mx];
              eky3 -= x0*vdy_brick_a3[mz][my][mx];
              ekz3 -= x0*vdz_brick_a3[mz][my][mx];
              ekx4 -= x0*vdx_brick_a4[mz][my][mx];
              eky4 -= x0*vdy_brick_a4[mz][my][mx];
              ekz4 -= x0*vdz_brick_a4[mz][my][mx];
              ekx5 -= x0*vdx_brick_a5[mz][my][mx];
              eky5 -= x0*vdy_brick_a5[mz][my][mx];
              ekz5 -= x0*vdz_brick_a5[mz][my][mx];
              ekx6 -= x0*vdx_brick_a6[mz][my][mx];
              eky6 -= x0*vdy_brick_a6[mz][my][mx];
              ekz6 -= x0*vdz_brick_a6[mz][my][mx];
            }
          }
        }

        // convert D-field to force
        type = atom->type[i];
        lj0 = B[7*type+6];
        lj1 = B[7*type+5];
        lj2 = B[7*type+4];
        lj3 = B[7*type+3];
        lj4 = B[7*type+2];
        lj5 = B[7*type+1];
        lj6 = B[7*type];
        f[i][0] += lj0*ekx0 + lj1*ekx1 + lj2*ekx2 + lj3*ekx3 + lj4*ekx4 + lj5*ekx5 + lj6*ekx6;
        f[i][1] += lj0*eky0 + lj1*eky1 + lj2*eky2 + lj3*eky3 + lj4*eky4 + lj5*eky5 + lj6*eky6;
        f[i][2] += lj0*ekz0 + lj1*ekz1 + lj2*ekz2 + lj3*ekz3 + lj4*ekz4 + lj5*ekz5 + lj6*ekz6;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get dispersion field & force on my particles
   for ad scheme and arithmetic mixing rule
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::fieldforce_a_ad()
{
  const int nlocal = atom->nlocal;

  // no local atoms => nothing to do

  if (nlocal == 0) return;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  const double * const * const x = atom->x;
  double *prd;

  if (triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;

  const double hx_inv = nx_pppm_6/xprd;
  const double hy_inv = ny_pppm_6/yprd;
  const double hz_inv = nz_pppm_6/zprd_slab;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int tid = omp_get_thread_num();
    const int inum = nlocal;
    const int idelta = 1 + inum/comm->nthreads;
    const int ifrom = tid*idelta;
    const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
#else
    const int ifrom = 0;
    const int ito = nlocal;
    const int tid = 0;
#endif
    ThrData *thr = fix->get_thr(tid);
    double * const * const f = thr->get_f();
    FFT_SCALAR * const * const r1d =  static_cast<FFT_SCALAR **>(thr->get_rho1d_6());
    FFT_SCALAR * const * const dr1d = static_cast<FFT_SCALAR **>(thr->get_drho1d_6());

    int l,m,n,nx,ny,nz,mx,my,mz;
    FFT_SCALAR dx,dy,dz,x0,y0,z0;
    FFT_SCALAR ekx0, eky0, ekz0, ekx1, eky1, ekz1, ekx2, eky2, ekz2;
    FFT_SCALAR ekx3, eky3, ekz3, ekx4, eky4, ekz4, ekx5, eky5, ekz5;
    FFT_SCALAR ekx6, eky6, ekz6;
    int type;
    double lj0,lj1,lj2,lj3,lj4,lj5,lj6;
    double sf = 0.0;
    double s1,s2,s3;

    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) {
      for (int i = ifrom; i < ito; i++) {

        nx = part2grid_6[i][0];
        ny = part2grid_6[i][1];
        nz = part2grid_6[i][2];
        dx = nx+shiftone_6 - (x[i][0]-boxlo[0])*delxinv_6;
        dy = ny+shiftone_6 - (x[i][1]-boxlo[1])*delyinv_6;
        dz = nz+shiftone_6 - (x[i][2]-boxlo[2])*delzinv_6;

        compute_rho1d_thr(r1d,dx,dy,dz, order_6, rho_coeff_6);
        compute_drho1d_thr(dr1d,dx,dy,dz, order_6, drho_coeff_6);

        ekx0 = eky0 = ekz0 = ZEROF;
        ekx1 = eky1 = ekz1 = ZEROF;
        ekx2 = eky2 = ekz2 = ZEROF;
        ekx3 = eky3 = ekz3 = ZEROF;
        ekx4 = eky4 = ekz4 = ZEROF;
        ekx5 = eky5 = ekz5 = ZEROF;
        ekx6 = eky6 = ekz6 = ZEROF;
        for (n = nlower_6; n <= nupper_6; n++) {
          mz = n+nz;
          for (m = nlower_6; m <= nupper_6; m++) {
            my = m+ny;
            for (l = nlower_6; l <= nupper_6; l++) {
              mx = l+nx;
              x0 = dr1d[0][l]*r1d[1][m]*r1d[2][n];
              y0 = r1d[0][l]*dr1d[1][m]*r1d[2][n];
              z0 = r1d[0][l]*r1d[1][m]*dr1d[2][n];

              ekx0 += x0*u_brick_a0[mz][my][mx];
              eky0 += y0*u_brick_a0[mz][my][mx];
              ekz0 += z0*u_brick_a0[mz][my][mx];

              ekx1 += x0*u_brick_a1[mz][my][mx];
              eky1 += y0*u_brick_a1[mz][my][mx];
              ekz1 += z0*u_brick_a1[mz][my][mx];

              ekx2 += x0*u_brick_a2[mz][my][mx];
              eky2 += y0*u_brick_a2[mz][my][mx];
              ekz2 += z0*u_brick_a2[mz][my][mx];

              ekx3 += x0*u_brick_a3[mz][my][mx];
              eky3 += y0*u_brick_a3[mz][my][mx];
              ekz3 += z0*u_brick_a3[mz][my][mx];

              ekx4 += x0*u_brick_a4[mz][my][mx];
              eky4 += y0*u_brick_a4[mz][my][mx];
              ekz4 += z0*u_brick_a4[mz][my][mx];

              ekx5 += x0*u_brick_a5[mz][my][mx];
              eky5 += y0*u_brick_a5[mz][my][mx];
              ekz5 += z0*u_brick_a5[mz][my][mx];

              ekx6 += x0*u_brick_a6[mz][my][mx];
              eky6 += y0*u_brick_a6[mz][my][mx];
              ekz6 += z0*u_brick_a6[mz][my][mx];
            }
          }
        }

        ekx0 *= hx_inv;
        eky0 *= hy_inv;
        ekz0 *= hz_inv;

        ekx1 *= hx_inv;
        eky1 *= hy_inv;
        ekz1 *= hz_inv;

        ekx2 *= hx_inv;
        eky2 *= hy_inv;
        ekz2 *= hz_inv;

        ekx3 *= hx_inv;
        eky3 *= hy_inv;
        ekz3 *= hz_inv;

        ekx4 *= hx_inv;
        eky4 *= hy_inv;
        ekz4 *= hz_inv;

        ekx5 *= hx_inv;
        eky5 *= hy_inv;
        ekz5 *= hz_inv;

        ekx6 *= hx_inv;
        eky6 *= hy_inv;
        ekz6 *= hz_inv;

        // convert D-field to force
        type = atom->type[i];
        lj0 = B[7*type+6];
        lj1 = B[7*type+5];
        lj2 = B[7*type+4];
        lj3 = B[7*type+3];
        lj4 = B[7*type+2];
        lj5 = B[7*type+1];
        lj6 = B[7*type];

        s1 = x[i][0]*hx_inv;
        s2 = x[i][1]*hy_inv;
        s3 = x[i][2]*hz_inv;

        sf = sf_coeff_6[0]*sin(2*MY_PI*s1);
        sf += sf_coeff_6[1]*sin(4*MY_PI*s1);
        sf *= 4*lj0*lj6 + 4*lj1*lj5 + 4*lj2*lj4 + 2*lj3*lj3;
        f[i][0] += lj0*ekx0 + lj1*ekx1 + lj2*ekx2 + lj3*ekx3 + lj4*ekx4 + lj5*ekx5 + lj6*ekx6 - sf;

        sf = sf_coeff_6[2]*sin(2*MY_PI*s2);
        sf += sf_coeff_6[3]*sin(4*MY_PI*s2);
        sf *= 4*lj0*lj6 + 4*lj1*lj5 + 4*lj2*lj4 + 2*lj3*lj3;
        f[i][1] += lj0*eky0 + lj1*eky1 + lj2*eky2 + lj3*eky3 + lj4*eky4 + lj5*eky5 + lj6*eky6 - sf;

        sf = sf_coeff_6[4]*sin(2*MY_PI*s3);
        sf += sf_coeff_6[5]*sin(4*MY_PI*s3);
        sf *= 4*lj0*lj6 + 4*lj1*lj5 + 4*lj2*lj4 + 2*lj3*lj3;
        if (slabflag != 2) f[i][2] += lj0*ekz0 + lj1*ekz1 + lj2*ekz2 + lj3*ekz3 + lj4*ekz4 + lj5*ekz5 + lj6*ekz6 - sf;
      }
    }
  }
}

/* ----------------------------------------------------------------------
 interpolate from grid to get per-atom energy/virial for dispersion
 interaction and arithmetic mixing rule
 ------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::fieldforce_a_peratom()
{
  const int nlocal = atom->nlocal;

  // no local atoms => nothing to do

  if (nlocal == 0) return;

  // loop over my charges, interpolate from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  const double * const * const x = atom->x;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int tid = omp_get_thread_num();
    const int inum = nlocal;
    const int idelta = 1 + inum/comm->nthreads;
    const int ifrom = tid*idelta;
    const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
#else
    const int ifrom = 0;
    const int ito = nlocal;
    const int tid = 0;
#endif
    ThrData *thr = fix->get_thr(tid);
    FFT_SCALAR * const * const r1d =  static_cast<FFT_SCALAR **>(thr->get_rho1d_6());

    int i,l,m,n,nx,ny,nz,mx,my,mz;
    FFT_SCALAR dx,dy,dz,x0,y0,z0;
    FFT_SCALAR u0,v00,v10,v20,v30,v40,v50;
    FFT_SCALAR u1,v01,v11,v21,v31,v41,v51;
    FFT_SCALAR u2,v02,v12,v22,v32,v42,v52;
    FFT_SCALAR u3,v03,v13,v23,v33,v43,v53;
    FFT_SCALAR u4,v04,v14,v24,v34,v44,v54;
    FFT_SCALAR u5,v05,v15,v25,v35,v45,v55;
    FFT_SCALAR u6,v06,v16,v26,v36,v46,v56;
    int type;
    double lj0,lj1,lj2,lj3,lj4,lj5,lj6;

    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) {
      for (i = ifrom; i < ito; i++) {

        nx = part2grid_6[i][0];
        ny = part2grid_6[i][1];
        nz = part2grid_6[i][2];
        dx = nx+shiftone_6 - (x[i][0]-boxlo[0])*delxinv_6;
        dy = ny+shiftone_6 - (x[i][1]-boxlo[1])*delyinv_6;
        dz = nz+shiftone_6 - (x[i][2]-boxlo[2])*delzinv_6;

        compute_rho1d_thr(r1d,dx,dy,dz, order_6, rho_coeff_6);

        u0 = v00 = v10 = v20 = v30 = v40 = v50 = ZEROF;
        u1 = v01 = v11 = v21 = v31 = v41 = v51 = ZEROF;
        u2 = v02 = v12 = v22 = v32 = v42 = v52 = ZEROF;
        u3 = v03 = v13 = v23 = v33 = v43 = v53 = ZEROF;
        u4 = v04 = v14 = v24 = v34 = v44 = v54 = ZEROF;
        u5 = v05 = v15 = v25 = v35 = v45 = v55 = ZEROF;
        u6 = v06 = v16 = v26 = v36 = v46 = v56 = ZEROF;
        for (n = nlower_6; n <= nupper_6; n++) {
          mz = n+nz;
          z0 = r1d[2][n];
          for (m = nlower_6; m <= nupper_6; m++) {
            my = m+ny;
            y0 = z0*r1d[1][m];
            for (l = nlower_6; l <= nupper_6; l++) {
              mx = l+nx;
              x0 = y0*r1d[0][l];
              if (eflag_atom) {
                u0 += x0*u_brick_a0[mz][my][mx];
                u1 += x0*u_brick_a1[mz][my][mx];
                u2 += x0*u_brick_a2[mz][my][mx];
                u3 += x0*u_brick_a3[mz][my][mx];
                u4 += x0*u_brick_a4[mz][my][mx];
                u5 += x0*u_brick_a5[mz][my][mx];
                u6 += x0*u_brick_a6[mz][my][mx];
              }
              if (vflag_atom) {
                v00 += x0*v0_brick_a0[mz][my][mx];
                v10 += x0*v1_brick_a0[mz][my][mx];
                v20 += x0*v2_brick_a0[mz][my][mx];
                v30 += x0*v3_brick_a0[mz][my][mx];
                v40 += x0*v4_brick_a0[mz][my][mx];
                v50 += x0*v5_brick_a0[mz][my][mx];
                v01 += x0*v0_brick_a1[mz][my][mx];
                v11 += x0*v1_brick_a1[mz][my][mx];
                v21 += x0*v2_brick_a1[mz][my][mx];
                v31 += x0*v3_brick_a1[mz][my][mx];
                v41 += x0*v4_brick_a1[mz][my][mx];
                v51 += x0*v5_brick_a1[mz][my][mx];
                v02 += x0*v0_brick_a2[mz][my][mx];
                v12 += x0*v1_brick_a2[mz][my][mx];
                v22 += x0*v2_brick_a2[mz][my][mx];
                v32 += x0*v3_brick_a2[mz][my][mx];
                v42 += x0*v4_brick_a2[mz][my][mx];
                v52 += x0*v5_brick_a2[mz][my][mx];
                v03 += x0*v0_brick_a3[mz][my][mx];
                v13 += x0*v1_brick_a3[mz][my][mx];
                v23 += x0*v2_brick_a3[mz][my][mx];
                v33 += x0*v3_brick_a3[mz][my][mx];
                v43 += x0*v4_brick_a3[mz][my][mx];
                v53 += x0*v5_brick_a3[mz][my][mx];
                v04 += x0*v0_brick_a4[mz][my][mx];
                v14 += x0*v1_brick_a4[mz][my][mx];
                v24 += x0*v2_brick_a4[mz][my][mx];
                v34 += x0*v3_brick_a4[mz][my][mx];
                v44 += x0*v4_brick_a4[mz][my][mx];
                v54 += x0*v5_brick_a4[mz][my][mx];
                v05 += x0*v0_brick_a5[mz][my][mx];
                v15 += x0*v1_brick_a5[mz][my][mx];
                v25 += x0*v2_brick_a5[mz][my][mx];
                v35 += x0*v3_brick_a5[mz][my][mx];
                v45 += x0*v4_brick_a5[mz][my][mx];
                v55 += x0*v5_brick_a5[mz][my][mx];
                v06 += x0*v0_brick_a6[mz][my][mx];
                v16 += x0*v1_brick_a6[mz][my][mx];
                v26 += x0*v2_brick_a6[mz][my][mx];
                v36 += x0*v3_brick_a6[mz][my][mx];
                v46 += x0*v4_brick_a6[mz][my][mx];
                v56 += x0*v5_brick_a6[mz][my][mx];
              }
            }
          }
        }

        // convert D-field to force
        type = atom->type[i];
        lj0 = B[7*type+6]*0.5;
        lj1 = B[7*type+5]*0.5;
        lj2 = B[7*type+4]*0.5;
        lj3 = B[7*type+3]*0.5;
        lj4 = B[7*type+2]*0.5;
        lj5 = B[7*type+1]*0.5;
        lj6 = B[7*type]*0.5;

        if (eflag_atom)
          eatom[i] += u0*lj0 + u1*lj1 + u2*lj2 +
            u3*lj3 + u4*lj4 + u5*lj5 + u6*lj6;
        if (vflag_atom) {
          vatom[i][0] += v00*lj0 + v01*lj1 + v02*lj2 + v03*lj3 +
            v04*lj4 + v05*lj5 + v06*lj6;
          vatom[i][1] += v10*lj0 + v11*lj1 + v12*lj2 + v13*lj3 +
            v14*lj4 + v15*lj5 + v16*lj6;
          vatom[i][2] += v20*lj0 + v21*lj1 + v22*lj2 + v23*lj3 +
            v24*lj4 + v25*lj5 + v26*lj6;
          vatom[i][3] += v30*lj0 + v31*lj1 + v32*lj2 + v33*lj3 +
            v34*lj4 + v35*lj5 + v36*lj6;
          vatom[i][4] += v40*lj0 + v41*lj1 + v42*lj2 + v43*lj3 +
            v44*lj4 + v45*lj5 + v46*lj6;
          vatom[i][5] += v50*lj0 + v51*lj1 + v52*lj2 + v53*lj3 +
            v54*lj4 + v55*lj5 + v56*lj6;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   charge assignment into rho1d
   dx,dy,dz = distance of particle from "lower left" grid point
------------------------------------------------------------------------- */
void PPPMDispTIP4POMP::compute_rho1d_thr(FFT_SCALAR * const * const r1d, const FFT_SCALAR &dx,
                                    const FFT_SCALAR &dy, const FFT_SCALAR &dz,
                                    const int ord, FFT_SCALAR * const * const rho_c)
{
  int k,l;
  FFT_SCALAR r1,r2,r3;

  for (k = (1-ord)/2; k <= ord/2; k++) {
    r1 = r2 = r3 = ZEROF;

    for (l = ord-1; l >= 0; l--) {
      r1 = rho_c[l][k] + r1*dx;
      r2 = rho_c[l][k] + r2*dy;
      r3 = rho_c[l][k] + r3*dz;
    }
    r1d[0][k] = r1;
    r1d[1][k] = r2;
    r1d[2][k] = r3;
  }
}

/* ----------------------------------------------------------------------
   charge assignment into drho1d
   dx,dy,dz = distance of particle from "lower left" grid point
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::compute_drho1d_thr(FFT_SCALAR * const * const dr1d, const FFT_SCALAR &dx,
                                    const FFT_SCALAR &dy, const FFT_SCALAR &dz,
                                    const int ord, FFT_SCALAR * const * const drho_c)
{
  int k,l;
  FFT_SCALAR r1,r2,r3;

  for (k = (1-ord)/2; k <= ord/2; k++) {
    r1 = r2 = r3 = ZEROF;

    for (l = ord-2; l >= 0; l--) {
      r1 = drho_c[l][k] + r1*dx;
      r2 = drho_c[l][k] + r2*dy;
      r3 = drho_c[l][k] + r3*dz;
    }
    dr1d[0][k] = r1;
    dr1d[1][k] = r2;
    dr1d[2][k] = r3;
  }
}

/* ----------------------------------------------------------------------
  find 2 H atoms bonded to O atom i
  compute position xM of fictitious charge site for O atom
  also return local indices iH1,iH2 of H atoms
------------------------------------------------------------------------- */

void PPPMDispTIP4POMP::find_M_thr(int i, int &iH1, int &iH2, dbl3_t &xM)
{
  iH1 = atom->map(atom->tag[i] + 1);
  iH2 = atom->map(atom->tag[i] + 2);

  if (iH1 == -1 || iH2 == -1) error->one(FLERR,"TIP4P hydrogen is missing");
  if (atom->type[iH1] != typeH || atom->type[iH2] != typeH)
    error->one(FLERR,"TIP4P hydrogen has incorrect atom type");

  // set iH1,iH2 to index of closest image to O

  iH1 = domain->closest_image(i,iH1);
  iH2 = domain->closest_image(i,iH2);

  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];

  double delx1 = x[iH1].x - x[i].x;
  double dely1 = x[iH1].y - x[i].y;
  double delz1 = x[iH1].z - x[i].z;

  double delx2 = x[iH2].x - x[i].x;
  double dely2 = x[iH2].y - x[i].y;
  double delz2 = x[iH2].z - x[i].z;

  xM.x = x[i].x + alpha * 0.5 * (delx1 + delx2);
  xM.y = x[i].y + alpha * 0.5 * (dely1 + dely2);
  xM.z = x[i].z + alpha * 0.5 * (delz1 + delz2);
}
