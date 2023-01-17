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

#ifndef LMP_FIX_QEQ_H
#define LMP_FIX_QEQ_H

#include "fix.h"

#define EV_TO_KCAL_PER_MOL 14.4
#define DANGER_ZONE 0.90
#define MIN_CAP 50
#define SAFE_ZONE 1.2
#define MIN_NBRS 100

namespace LAMMPS_NS {

class FixQEq : public Fix {
 public:
  FixQEq(class LAMMPS *, int, char **);
  ~FixQEq();
  int setmask();
  void init_list(int, class NeighList *);
  void setup_pre_force(int);
  void setup_pre_force_respa(int, int);
  void pre_force_respa(int, int, int);
  void min_pre_force(int);

  virtual double compute_scalar();

  // derived child classes must provide these functions

  virtual void init() = 0;
  virtual void pre_force(int) = 0;

  virtual int pack_forward_comm(int, int *, double *, int, int *);
  virtual void unpack_forward_comm(int, int, double *);
  virtual int pack_reverse_comm(int, int, double *);
  virtual void unpack_reverse_comm(int, int *, double *);
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  double memory_usage();

 protected:
  int nevery;
  int nlocal, nall, m_fill;
  int n_cap, nmax, m_cap;
  int pack_flag;
  int nlevels_respa;
  class NeighList *list;

  int matvecs;
  double qeq_time;

  double swa, swb;             // lower/upper Taper cutoff radius
  double Tap[8];               // Taper function
  double tolerance;            // tolerance for the norm of the rel residual in CG
  int maxiter;                 // maximum number of QEq iterations
  int maxwarn;                 // print warning when max iterations was reached
  double cutoff, cutoff_sq;    // neighbor cutoff

  double *chi, *eta, *gamma, *zeta, *zcore;    // qeq parameters
  double *chizj;
  double **shld;
  int streitz_flag, reax_flag;

  bigint ngroup;

  // fictitious charges

  double *s, *t;
  double **s_hist, **t_hist;
  int nprev;

  typedef struct {
    int n, m;
    int *firstnbr;
    int *numnbrs;
    int *jlist;
    double *val;
  } sparse_matrix;

  sparse_matrix H;
  double *Hdia_inv;
  double *b_s, *b_t;
  double *p, *q, *r, *d;

  // streitz-mintmire

  double alpha;

  // damped dynamics

  double *qf, *q1, *q2, qdamp, qstep;

  // fire
  double *qv;

  void calculate_Q();

  double parallel_norm(double *, int);
  double parallel_dot(double *, double *, int);
  double parallel_vector_acc(double *, int);

  void vector_sum(double *, double, double *, double, double *, int);
  void vector_add(double *, double, double *, int);

  void init_storage();
  void read_file(char *);
  void allocate_storage();
  void deallocate_storage();
  void reallocate_storage();
  void allocate_matrix();
  void deallocate_matrix();
  void reallocate_matrix();

  virtual int CG(double *, double *);
  virtual void sparse_matvec(sparse_matrix *, double *, double *);
};

}    // namespace LAMMPS_NS

#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: QEQ with 'newton pair off' not supported

See the newton command.  This is a restriction to use the QEQ fixes.

W: Fix qeq CG convergence failed (%g) after %d iterations at %ld step

Self-explanatory.

E: Cannot open fix qeq parameter file %s

The specified file cannot be opened.  Check that the path and name are
correct.

E: Invalid fix qeq parameter file

Element index > number of atom types.

*/
