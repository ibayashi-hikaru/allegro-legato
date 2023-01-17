/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(msm,MSM);
// clang-format on
#else

#ifndef LMP_MSM_H
#define LMP_MSM_H

#include "kspace.h"

namespace LAMMPS_NS {

class MSM : public KSpace {
 public:
  MSM(class LAMMPS *);
  virtual ~MSM();
  void init();
  void setup();
  virtual void settings(int, char **);
  virtual void compute(int, int);
  virtual double memory_usage();

 protected:
  int me, nprocs;
  double precision;
  int nfactors;
  int *factors;
  double qqrd2e;
  double cutoff;
  double volume;
  double *delxinv, *delyinv, *delzinv;
  double h_x, h_y, h_z;
  double C_p;

  int *nx_msm, *ny_msm, *nz_msm;
  int *nxlo_in, *nylo_in, *nzlo_in;
  int *nxhi_in, *nyhi_in, *nzhi_in;
  int *nxlo_out, *nylo_out, *nzlo_out;
  int *nxhi_out, *nyhi_out, *nzhi_out;
  int *ngrid, *active_flag;
  int *alpha, *betax, *betay, *betaz;
  int nxlo_out_all, nylo_out_all, nzlo_out_all;
  int nxhi_out_all, nyhi_out_all, nzhi_out_all;
  int nxlo_direct, nxhi_direct, nylo_direct;
  int nyhi_direct, nzlo_direct, nzhi_direct;
  int nmax_direct;
  int nlower, nupper;
  int peratom_allocate_flag;
  int levels;

  MPI_Comm *world_levels;

  double ****qgrid;
  double ****egrid;
  double ****v0grid, ****v1grid, ****v2grid;
  double ****v3grid, ****v4grid, ****v5grid;
  double **g_direct;
  double **v0_direct, **v1_direct, **v2_direct;
  double **v3_direct, **v4_direct, **v5_direct;
  double *g_direct_top;
  double *v0_direct_top, *v1_direct_top, *v2_direct_top;
  double *v3_direct_top, *v4_direct_top, *v5_direct_top;

  double **phi1d, **dphi1d;

  int procgrid[3];            // procs assigned in each dim of 3d grid
  int myloc[3];               // which proc I am in each dim
  int ***procneigh_levels;    // my 6 neighboring procs, 0/1 = left/right

  class GridComm *gcall;    // GridComm class for finest level grid
  class GridComm **gc;      // GridComm classes for each hierarchical level

  double *gcall_buf1, *gcall_buf2;
  double **gc_buf1, **gc_buf2;
  int ngcall_buf1, ngcall_buf2, npergrid;
  int *ngc_buf1, *ngc_buf2;

  int current_level;

  int **part2grid;    // storage for particle -> grid mapping
  int nmax;

  int triclinic;
  double *boxlo;

  void set_grid_global();
  void set_proc_grid(int);
  void set_grid_local();
  void setup_grid();
  double estimate_1d_error(double, double);
  double estimate_3d_error();
  double estimate_total_error();
  void allocate();
  void allocate_peratom();
  void deallocate();
  void deallocate_peratom();
  void allocate_levels();
  void deallocate_levels();
  int factorable(int, int &, int &);
  void particle_map();
  void make_rho();
  virtual void direct(int);
  void direct_peratom(int);
  void direct_top(int);
  void direct_peratom_top(int);
  void restriction(int);
  void prolongation(int);
  void grid_swap_forward(int, double ***&);
  void grid_swap_reverse(int, double ***&);
  void fieldforce();
  void fieldforce_peratom();
  void compute_phis(const double &, const double &, const double &);
  void compute_phis_and_dphis(const double &, const double &, const double &);
  inline double compute_phi(const double &);
  inline double compute_dphi(const double &);
  void get_g_direct();
  void get_virial_direct();
  void get_g_direct_top(int);
  void get_virial_direct_top(int);

  // grid communication

  void pack_forward_grid(int, void *, int, int *);
  void unpack_forward_grid(int, void *, int, int *);
  void pack_reverse_grid(int, void *, int, int *);
  void unpack_reverse_grid(int, void *, int, int *);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Cannot (yet) use MSM with 2d simulation

This feature is not yet supported.

E: MSM can only currently be used with comm_style brick

This is a current restriction in LAMMPS.

E: Kspace style requires atom attribute q

The atom style defined does not have these attributes.

W: Slab correction not needed for MSM

Slab correction is intended to be used with Ewald or PPPM and is not needed by MSM.

E: MSM order must be 4, 6, 8, or 10

This is a limitation of the MSM implementation in LAMMPS:
the MSM order can only be 4, 6, 8, or 10.

E: Cannot (yet) use single precision with MSM (remove -DFFT_SINGLE from Makefile and recompile)

Single precision cannot be used with MSM.

E: KSpace style is incompatible with Pair style

Setting a kspace style requires that a pair style with matching
long-range Coulombic or dispersion components be used.

E: Must use 'kspace_modify pressure/scalar no' to obtain per-atom virial with kspace_style MSM

The kspace scalar pressure option cannot be used to obtain per-atom virial.

E: KSpace accuracy must be > 0

The kspace accuracy designated in the input must be greater than zero.

W: Number of MSM mesh points changed to be a multiple of 2

MSM requires that the number of grid points in each direction be a multiple
of two and the number of grid points in one or more directions have been
adjusted to meet this requirement.

W: Adjusting Coulombic cutoff for MSM, new cutoff = %g

The adjust/cutoff command is turned on and the Coulombic cutoff has been
adjusted to match the user-specified accuracy.

E: Too many MSM grid levels

The max number of MSM grid levels is hardwired to 10.

W: MSM mesh too small, increasing to 2 points in each direction

Self-explanatory.

E: MSM grid is too large

The global MSM grid is larger than OFFSET in one or more dimensions.
OFFSET is currently set to 16384.  You likely need to decrease the
requested accuracy.

E: Non-numeric box dimensions - simulation unstable

The box size has apparently blown up.

E: Out of range atoms - cannot compute MSM

One or more atoms are attempting to map their charge to a MSM grid point
that is not owned by a processor.  This is likely for one of two
reasons, both of them bad.  First, it may mean that an atom near the
boundary of a processor's sub-domain has moved more than 1/2 the
"neighbor skin distance"_neighbor.html without neighbor lists being
rebuilt and atoms being migrated to new processors.  This also means
you may be missing pairwise interactions that need to be computed.
The solution is to change the re-neighboring criteria via the
"neigh_modify"_neigh_modify command.  The safest settings are "delay 0
every 1 check yes".  Second, it may mean that an atom has moved far
outside a processor's sub-domain or even the entire simulation box.
This indicates bad physics, e.g. due to highly overlapping atoms, too
large a timestep, etc.

*/
