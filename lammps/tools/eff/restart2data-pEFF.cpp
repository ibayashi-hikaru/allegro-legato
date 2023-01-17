/* -----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------ */

// Convert a LAMMPS binary restart file into an ASCII text data file
//
// Syntax: restart2data restart-file data-file (input-file)
//         input-file is optional
//         if specified it will contain LAMMPS input script commands
//           for mass and force field info
//         only a few force field styles support this option
//
// this serial code must be compiled on a platform that can read the binary
//   restart file since binary formats are not compatible across all platforms
// restart-file can have a '%' character to indicate a multiproc restart
//   file as written by LAMMPS

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define MAX_GROUP 32

// same as write_restart.cpp

enum{VERSION,UNITS,NTIMESTEP,DIMENSION,NPROCS,PROCGRID_0,PROCGRID_1,PROCGRID_2,
       NEWTON_PAIR,NEWTON_BOND,XPERIODIC,YPERIODIC,ZPERIODIC,
       BOUNDARY_00,BOUNDARY_01,BOUNDARY_10,BOUNDARY_11,BOUNDARY_20,BOUNDARY_21,
       ATOM_STYLE,NATOMS,NTYPES,
       NBONDS,NBONDTYPES,BOND_PER_ATOM,
       NANGLES,NANGLETYPES,ANGLE_PER_ATOM,
       NDIHEDRALS,NDIHEDRALTYPES,DIHEDRAL_PER_ATOM,
       NIMPROPERS,NIMPROPERTYPES,IMPROPER_PER_ATOM,
       BOXLO_0,BOXHI_0,BOXLO_1,BOXHI_1,BOXLO_2,BOXHI_2,
       SPECIAL_LJ_1,SPECIAL_LJ_2,SPECIAL_LJ_3,
       SPECIAL_COUL_1,SPECIAL_COUL_2,SPECIAL_COUL_3,
       XY,XZ,YZ};
enum{MASS,SHAPE,DIPOLE};
enum{PAIR,BOND,ANGLE,DIHEDRAL,IMPROPER};

static const char * const cg_type_list[] = 
  {"none", "lj9_6", "lj12_4", "lj12_6"};

// ---------------------------------------------------------------------
// Data class to hold problem
// ---------------------------------------------------------------------

class Data {
 public:

  // global settings

  char *version;
  int ntimestep;
  int nprocs;
  char *unit_style;
  int dimension;
  int px,py,pz;
  int newton_pair,newton_bond;
  int xperiodic,yperiodic,zperiodic;
  int boundary[3][2];

  char *atom_style;
  int style_angle,style_atomic,style_bond,style_charge,style_dipole;
  int style_dpd,style_ellipsoid,style_full,style_granular;
  int style_hybrid,style_molecular,style_peri,style_electron;

  int natoms,nbonds,nangles,ndihedrals,nimpropers;
  int ntypes,nbondtypes,nangletypes,ndihedraltypes,nimpropertypes;
  int bond_per_atom,angle_per_atom,dihedral_per_atom,improper_per_atom;
  int triclinic;

  double xlo,xhi,ylo,yhi,zlo,zhi,xy,xz,yz;
  double special_lj[4],special_coul[4];

  double cut_lj_global,cut_coul_global,kappa;
  int offset_flag,mix_flag;

  // force fields

  char *pair_style,*bond_style,*angle_style,*dihedral_style,*improper_style;

  double *pair_born_A,*pair_born_rho,*pair_born_sigma;
  double *pair_born_C,*pair_born_D;
  double *pair_buck_A,*pair_buck_rho,*pair_buck_C;
  double *pair_colloid_A12,*pair_colloid_sigma;
  double *pair_colloid_d1,*pair_colloid_d2;
  double *pair_dipole_epsilon,*pair_dipole_sigma;
  double *pair_dpd_a0,*pair_dpd_gamma;
  double *pair_charmm_epsilon,*pair_charmm_sigma;
  double *pair_charmm_eps14,*pair_charmm_sigma14;
  double *pair_class2_epsilon,*pair_class2_sigma;
  double *pair_gb_epsilon,*pair_gb_sigma;
  double *pair_gb_epsa,*pair_gb_epsb,*pair_gb_epsc;
  double *pair_lj_epsilon,*pair_lj_sigma;
  double **pair_cg_epsilon,**pair_cg_sigma;
  int **pair_cg_cmm_type, **pair_setflag;
  double **pair_cut_coul, **pair_cut_lj, **pair_cut_eff;
  double *pair_ljexpand_epsilon,*pair_ljexpand_sigma,*pair_ljexpand_shift;
  double *pair_ljgromacs_epsilon,*pair_ljgromacs_sigma;
  double *pair_ljsmooth_epsilon,*pair_ljsmooth_sigma;
  double *pair_morse_d0,*pair_morse_alpha,*pair_morse_r0;
  double *pair_soft_start,*pair_soft_stop;
  double *pair_yukawa_A;

  double *bond_class2_r0,*bond_class2_k2,*bond_class2_k3,*bond_class2_k4;
  double *bond_fene_k,*bond_fene_r0,*bond_fene_epsilon,*bond_fene_sigma;
  double *bond_feneexpand_k,*bond_feneexpand_r0;
  double *bond_feneexpand_epsilon,*bond_feneexpand_sigma;
  double *bond_feneexpand_shift;
  double *bond_harmonic_k,*bond_harmonic_r0;
  double *bond_morse_d0,*bond_morse_alpha,*bond_morse_r0;
  double *bond_nonlinear_epsilon,*bond_nonlinear_r0,*bond_nonlinear_lamda;
  double *bond_quartic_k,*bond_quartic_b1,*bond_quartic_b2;
  double *bond_quartic_rc,*bond_quartic_u0;

  double *angle_charmm_k,*angle_charmm_theta0;
  double *angle_charmm_k_ub,*angle_charmm_r_ub;
  double *angle_class2_theta0;
  double *angle_class2_k2,*angle_class2_k3,*angle_class2_k4;
  double *angle_class2_bb_k,*angle_class2_bb_r1,*angle_class2_bb_r2;
  double *angle_class2_ba_k1,*angle_class2_ba_k2;
  double *angle_class2_ba_r1,*angle_class2_ba_r2;
  double *angle_cosine_k;
  double *angle_cosine_squared_k,*angle_cosine_squared_theta0;
  double *angle_harmonic_k,*angle_harmonic_theta0;
  double *angle_cg_cmm_epsilon,*angle_cg_cmm_sigma;
  int *angle_cg_cmm_type;

  double *dihedral_charmm_k,*dihedral_charmm_weight;
  int *dihedral_charmm_multiplicity,*dihedral_charmm_sign;
  double *dihedral_class2_k1,*dihedral_class2_k2,*dihedral_class2_k3;
  double *dihedral_class2_phi1,*dihedral_class2_phi2,*dihedral_class2_phi3;
  double *dihedral_class2_mbt_f1,*dihedral_class2_mbt_f2;
  double *dihedral_class2_mbt_f3,*dihedral_class2_mbt_r0;
  double *dihedral_class2_ebt_f1_1,*dihedral_class2_ebt_f2_1;
  double *dihedral_class2_ebt_f3_1,*dihedral_class2_ebt_r0_1;
  double *dihedral_class2_ebt_f1_2,*dihedral_class2_ebt_f2_2;
  double *dihedral_class2_ebt_f3_2,*dihedral_class2_ebt_r0_2;
  double *dihedral_class2_at_f1_1,*dihedral_class2_at_f2_1;
  double *dihedral_class2_at_f3_1,*dihedral_class2_at_theta0_1;
  double *dihedral_class2_at_f1_2,*dihedral_class2_at_f2_2;
  double *dihedral_class2_at_f3_2,*dihedral_class2_at_theta0_2;
  double *dihedral_class2_aat_k;
  double *dihedral_class2_aat_theta0_1,*dihedral_class2_aat_theta0_2;
  double *dihedral_class2_bb13_k;
  double *dihedral_class2_bb13_r10,*dihedral_class2_bb13_r30;
  double *dihedral_harmonic_k;
  int *dihedral_harmonic_multiplicity,*dihedral_harmonic_sign;
  double *dihedral_helix_aphi,*dihedral_helix_bphi,*dihedral_helix_cphi;
  double *dihedral_multi_a1,*dihedral_multi_a2,*dihedral_multi_a3;
  double *dihedral_multi_a4,*dihedral_multi_a5;
  double *dihedral_opls_k1,*dihedral_opls_k2;
  double *dihedral_opls_k3,*dihedral_opls_k4;

  double *improper_class2_k0,*improper_class2_chi0;
  double *improper_class2_aa_k1,*improper_class2_aa_k2,*improper_class2_aa_k3;
  double *improper_class2_aa_theta0_1,*improper_class2_aa_theta0_2;
  double *improper_class2_aa_theta0_3;
  double *improper_cvff_k;
  int *improper_cvff_sign,*improper_cvff_multiplicity;
  double *improper_harmonic_k,*improper_harmonic_chi;

  // atom quantities

  int iatoms,ibonds,iangles,idihedrals,iimpropers;

  double *mass,*shape,*dipole;
  double *x,*y,*z,*vx,*vy,*vz;
  double *omegax,*omegay,*omegaz;
  int *tag,*type,*mask,*image;
  int *molecule;
  int *spin;
  double *ervel, *erforce;
  double *q,*mux,*muy,*muz,*radius,*density,*vfrac,*rmass;
  double *s0,*x0x,*x0y,*x0z;
  double *quatw,*quati,*quatj,*quatk,*angmomx,*angmomy,*angmomz;
  int *bond_type,*angle_type,*dihedral_type,*improper_type;
  int *bond_atom1,*bond_atom2;
  int *angle_atom1,*angle_atom2,*angle_atom3;
  int *dihedral_atom1,*dihedral_atom2,*dihedral_atom3,*dihedral_atom4;
  int *improper_atom1,*improper_atom2,*improper_atom3,*improper_atom4;
  
  // functions

  Data();
  void stats();
  void write(FILE *fp, FILE *fp2=NULL);

  void write_atom_angle(FILE *, int, int, int, int);
  void write_atom_atomic(FILE *, int, int, int, int);
  void write_atom_bond(FILE *, int, int, int, int);
  void write_atom_charge(FILE *, int, int, int, int);
  void write_atom_dipole(FILE *, int, int, int, int);
  void write_atom_dpd(FILE *, int, int, int, int);
  void write_atom_ellipsoid(FILE *, int, int, int, int);
  void write_atom_full(FILE *, int, int, int, int);
  void write_atom_granular(FILE *, int, int, int, int);
  void write_atom_molecular(FILE *, int, int, int, int);
  void write_atom_peri(FILE *, int, int, int, int);
  void write_atom_electron(FILE *, int, int, int, int);

  void write_atom_angle_extra(FILE *, int);
  void write_atom_atomic_extra(FILE *, int);
  void write_atom_bond_extra(FILE *, int);
  void write_atom_charge_extra(FILE *, int);
  void write_atom_dipole_extra(FILE *, int);
  void write_atom_dpd_extra(FILE *, int);
  void write_atom_ellipsoid_extra(FILE *, int);
  void write_atom_full_extra(FILE *, int);
  void write_atom_granular_extra(FILE *, int);
  void write_atom_molecular_extra(FILE *, int);
  void write_atom_peri_extra(FILE *, int);
  void write_atom_electron_extra(FILE *, int);

  void write_vel_angle(FILE *, int);
  void write_vel_atomic(FILE *, int);
  void write_vel_bond(FILE *, int);
  void write_vel_charge(FILE *, int);
  void write_vel_dipole(FILE *, int);
  void write_vel_dpd(FILE *, int);
  void write_vel_ellipsoid(FILE *, int);
  void write_vel_full(FILE *, int);
  void write_vel_granular(FILE *, int);
  void write_vel_molecular(FILE *, int);
  void write_vel_peri(FILE *, int);
  void write_vel_electron(FILE *, int);

  void write_vel_angle_extra(FILE *, int);
  void write_vel_atomic_extra(FILE *, int);
  void write_vel_bond_extra(FILE *, int);
  void write_vel_charge_extra(FILE *, int);
  void write_vel_dipole_extra(FILE *, int);
  void write_vel_dpd_extra(FILE *, int);
  void write_vel_ellipsoid_extra(FILE *, int);
  void write_vel_full_extra(FILE *, int);
  void write_vel_granular_extra(FILE *, int);
  void write_vel_molecular_extra(FILE *, int);
  void write_vel_peri_extra(FILE *, int);
  void write_vel_electron_extra(FILE *, int);
};

// ---------------------------------------------------------------------
// function prototypes
// ---------------------------------------------------------------------

void header(FILE *, Data &);
void set_style(char *, Data &, int);
void groups(FILE *);
void type_arrays(FILE *, Data &);
void force_fields(FILE *, Data &);
void modify(FILE *);
void pair(FILE *fp, Data &data, char *style, int flag);
void bond(FILE *fp, Data &data);
void angle(FILE *fp, Data &data);
void dihedral(FILE *fp, Data &data);
void improper(FILE *fp, Data &data);
int atom(double *, Data &data);

void allocate_angle(Data &data);
void allocate_atomic(Data &data);
void allocate_bond(Data &data);
void allocate_charge(Data &data);
void allocate_dipole(Data &data);
void allocate_dpd(Data &data);
void allocate_ellipsoid(Data &data);
void allocate_full(Data &data);
void allocate_granular(Data &data);
void allocate_molecular(Data &data);
void allocate_peri(Data &data);
void allocate_electron(Data &data);

int atom_angle(double *, Data &, int);
int atom_atomic(double *, Data &, int);
int atom_bond(double *, Data &, int);
int atom_charge(double *, Data &, int);
int atom_dipole(double *, Data &, int);
int atom_dpd(double *, Data &, int);
int atom_ellipsoid(double *, Data &, int);
int atom_full(double *, Data &, int);
int atom_granular(double *, Data &, int);
int atom_molecular(double *, Data &, int);
int atom_peri(double *, Data &, int);
int atom_electron(double *, Data &, int);

int read_int(FILE *fp);
double read_double(FILE *fp);
char *read_char(FILE *fp);

// ---------------------------------------------------------------------
// main program
// ---------------------------------------------------------------------

int main (int argc, char **argv)
{
  // syntax error check

    if ((argc != 3) && (argc !=4)) {
    printf("Syntax: restart2data restart-file data-file (input-file)\n");
    return 1;
  }

  // if restart file contains '%', file = filename with % replaced by "base"
  // else file = single file

  int multiproc;
  char *file,*ptr;

  if (ptr = strchr(argv[1],'%')) {
    multiproc = 1;
    file = new char[strlen(argv[1]) + 16];
    *ptr = '\0';
    sprintf(file,"%s%s%s",argv[1],"base",ptr+1);
  } else {
    multiproc = 0;
    file = argv[1];
  }

  // open single restart file or base file for multiproc case

  printf("Reading restart file ...\n");
  FILE *fp = fopen(file,"rb");
  if (fp == NULL) {
    printf("ERROR: Cannot open restart file %s\n",file);
    return 1;
  }

  // read beginning of restart file

  Data data;

  header(fp,data);
  groups(fp);
  type_arrays(fp,data);
  force_fields(fp,data);
  modify(fp);

  // read atoms from single or multiple restart files

  double *buf = NULL;
  int n,m;
  int maxbuf = 0;
  data.iatoms = data.ibonds = data.iangles = 
    data.idihedrals = data.iimpropers = 0;

  for (int iproc = 0; iproc < data.nprocs; iproc++) {
    if (multiproc) {
      fclose(fp);
      sprintf(file,"%s%d%s",argv[1],iproc,ptr+1);
      fp = fopen(file,"rb");
      if (fp == NULL) {
        printf("ERROR: Cannot open restart file %s\n",file);
        return 1;
      }
    }
    n = read_int(fp);

    if (n > maxbuf) {
      maxbuf = n;
      delete [] buf;
      buf = new double[maxbuf];
    }

    fread(buf,sizeof(double),n,fp);

    m = 0;
    while (m < n) m += atom(&buf[m],data);
  }

  fclose(fp);

  // print out stats

  data.stats();

  // write out data file and no input file

  if (argc == 3) {
    printf("Writing data file ...\n");
    fp = fopen(argv[2],"w");
    if (fp == NULL) {
      printf("ERROR: Cannot open data file %s\n",argv[2]);
      return 1;
    }
    data.write(fp);
    fclose(fp);

  // write out data file and input file

  } else {
    printf("Writing data file ...\n");
    fp = fopen(argv[2],"w");
    if (fp == NULL) {
      printf("ERROR: Cannot open data file %s\n",argv[2]);
      return 1;
    }
    printf("Writing input file ...\n");
    FILE *fp2 = fopen(argv[3],"w");
    if (fp2 == NULL) {
      printf("ERROR: Cannot open input file %s\n",argv[3]);
      return 1;
    }

    data.write(fp,fp2);
    fclose(fp);
    fclose(fp2);
  }
  
  return 0;
}

// ---------------------------------------------------------------------
// read header of restart file
// ---------------------------------------------------------------------

void header(FILE *fp, Data &data)
{
  char *version = "7 Jul 2009";

  data.triclinic = 0;

  int flag;
  flag = read_int(fp);

  while (flag >= 0) {

    if (flag == VERSION) {
      data.version = read_char(fp);
      if (strcmp(version,data.version) != 0) {
	char *str = "Restart file version does not match restart2data version";
	printf("WARNING %s\n",str);
	printf("  restart2data version = %s\n",version);
      }
    }
    else if (flag == UNITS) data.unit_style = read_char(fp);
    else if (flag == NTIMESTEP) data.ntimestep = read_int(fp);
    else if (flag == DIMENSION) data.dimension = read_int(fp);
    else if (flag == NPROCS) data.nprocs = read_int(fp);
    else if (flag == PROCGRID_0) data.px = read_int(fp);
    else if (flag == PROCGRID_1) data.py = read_int(fp);
    else if (flag == PROCGRID_2) data.pz = read_int(fp);
    else if (flag == NEWTON_PAIR) data.newton_pair = read_int(fp);
    else if (flag == NEWTON_BOND) data.newton_bond = read_int(fp);
    else if (flag == XPERIODIC) data.xperiodic = read_int(fp);
    else if (flag == YPERIODIC) data.yperiodic = read_int(fp);
    else if (flag == ZPERIODIC) data.zperiodic = read_int(fp);
    else if (flag == BOUNDARY_00) data.boundary[0][0] = read_int(fp);
    else if (flag == BOUNDARY_01) data.boundary[0][1] = read_int(fp);
    else if (flag == BOUNDARY_10) data.boundary[1][0] = read_int(fp);
    else if (flag == BOUNDARY_11) data.boundary[1][1] = read_int(fp);
    else if (flag == BOUNDARY_20) data.boundary[2][0] = read_int(fp);
    else if (flag == BOUNDARY_21) data.boundary[2][1] = read_int(fp);

    // if atom_style = hybrid:
    //   set data_style_hybrid to # of sub-styles
    //   read additional sub-class arguments
    //   set sub-styles to 1 to N

    else if (flag == ATOM_STYLE) {
      data.style_angle = data.style_atomic = data.style_bond = 
	data.style_charge = data.style_dipole =	data.style_dpd =
	data.style_ellipsoid = data.style_full = data.style_granular =
	data.style_hybrid = data.style_molecular = data.style_peri = data.style_electron = 0;

      data.atom_style = read_char(fp);
      set_style(data.atom_style,data,1);

      if (strcmp(data.atom_style,"hybrid") == 0) {
	int nwords = read_int(fp);
	set_style(data.atom_style,data,nwords);
	char *substyle;
	for (int i = 1; i <= nwords; i++) {
	  substyle = read_char(fp);
	  set_style(substyle,data,i);
	}
      }
    }

    else if (flag == NATOMS) data.natoms = static_cast<int> (read_double(fp));
    else if (flag == NTYPES) data.ntypes = read_int(fp);
    else if (flag == NBONDS) data.nbonds = read_int(fp);
    else if (flag == NBONDTYPES) data.nbondtypes = read_int(fp);
    else if (flag == BOND_PER_ATOM) data.bond_per_atom = read_int(fp);
    else if (flag == NANGLES) data.nangles = read_int(fp);
    else if (flag == NANGLETYPES) data.nangletypes = read_int(fp);
    else if (flag == ANGLE_PER_ATOM) data.angle_per_atom = read_int(fp);
    else if (flag == NDIHEDRALS) data.ndihedrals = read_int(fp);
    else if (flag == NDIHEDRALTYPES) data.ndihedraltypes = read_int(fp);
    else if (flag == DIHEDRAL_PER_ATOM) data.dihedral_per_atom = read_int(fp);
    else if (flag == NIMPROPERS) data.nimpropers = read_int(fp);
    else if (flag == NIMPROPERTYPES) data.nimpropertypes = read_int(fp);
    else if (flag == IMPROPER_PER_ATOM) data.improper_per_atom = read_int(fp);
    else if (flag == BOXLO_0) data.xlo = read_double(fp);
    else if (flag == BOXHI_0) data.xhi = read_double(fp);
    else if (flag == BOXLO_1) data.ylo = read_double(fp);
    else if (flag == BOXHI_1) data.yhi = read_double(fp);
    else if (flag == BOXLO_2) data.zlo = read_double(fp);
    else if (flag == BOXHI_2) data.zhi = read_double(fp);
    else if (flag == SPECIAL_LJ_1) data.special_lj[1] = read_double(fp);
    else if (flag == SPECIAL_LJ_2) data.special_lj[2] = read_double(fp);
    else if (flag == SPECIAL_LJ_3) data.special_lj[3] = read_double(fp);
    else if (flag == SPECIAL_COUL_1) data.special_coul[1] = read_double(fp);
    else if (flag == SPECIAL_COUL_2) data.special_coul[2] = read_double(fp);
    else if (flag == SPECIAL_COUL_3) data.special_coul[3] = read_double(fp);
    else if (flag == XY) {
      data.triclinic = 1;
      data.xy = read_double(fp);
    } else if (flag == XZ) {
      data.triclinic = 1;
      data.xz = read_double(fp);
    } else if (flag == YZ) {
      data.triclinic = 1;
      data.yz = read_double(fp);
    } else {
      printf("ERROR: Invalid flag in header section of restart file %d\n",
	     flag);
      exit(1);
    }

    flag = read_int(fp);
  }
}

// ---------------------------------------------------------------------
// set atom style to flag
// ---------------------------------------------------------------------

void set_style(char *name, Data &data, int flag)
{
  if (strcmp(name,"angle") == 0) data.style_angle = flag;
  else if (strcmp(name,"atomic") == 0) data.style_atomic = flag;
  else if (strcmp(name,"bond") == 0) data.style_bond = flag;
  else if (strcmp(name,"charge") == 0) data.style_charge = flag;
  else if (strcmp(name,"dipole") == 0) data.style_dipole = flag;
  else if (strcmp(name,"dpd") == 0) data.style_dpd = flag;
  else if (strcmp(name,"ellipsoid") == 0) data.style_ellipsoid = flag;
  else if (strcmp(name,"full") == 0) data.style_full = flag;
  else if (strcmp(name,"granular") == 0) data.style_granular = flag;
  else if (strcmp(name,"hybrid") == 0) data.style_hybrid = flag;
  else if (strcmp(name,"molecular") == 0) data.style_molecular = flag;
  else if (strcmp(name,"peri") == 0) data.style_peri = flag;
  else if (strcmp(name,"electron") == 0) data.style_electron = flag;
  else {
    printf("ERROR: Unknown atom style %s\n",name);
    exit(1);
  }
}

// ---------------------------------------------------------------------
// read group info from restart file, just ignore it
// ---------------------------------------------------------------------

void groups(FILE *fp)
{
  int ngroup = read_int(fp);

  int n;
  char *name;

  // use count to not change restart format with deleted groups
  // remove this on next major release

  int count = 0;
  for (int i = 0; i < MAX_GROUP; i++) {
    name = read_char(fp);
    delete [] name;
    count++;
    if (count == ngroup) break;
  }
}

// ---------------------------------------------------------------------
// read type arrays from restart file
// ---------------------------------------------------------------------

void type_arrays(FILE *fp, Data &data)
{
  data.mass = NULL;
  data.shape = NULL;
  data.dipole = NULL;

  int flag;
  flag = read_int(fp);

  while (flag >= 0) {

    if (flag == MASS) {
      data.mass = new double[data.ntypes+1];
      fread(&data.mass[1],sizeof(double),data.ntypes,fp);
    } else if (flag == SHAPE) {
      data.shape = new double[3*(data.ntypes+1)];
      fread(&data.shape[3],sizeof(double),3*data.ntypes,fp);
    } else if (flag == DIPOLE) {
      data.dipole = new double[data.ntypes+1];
      fread(&data.dipole[1],sizeof(double),data.ntypes,fp);
    } else {
      printf("ERROR: Invalid flag in type arrays section of restart file %d\n",
	     flag);
      exit(1);
    }

    flag = read_int(fp);
  }
}

// ---------------------------------------------------------------------
// read force-field info from restart file
// ---------------------------------------------------------------------

void force_fields(FILE *fp, Data &data)
{
  data.pair_style = data.bond_style = data.angle_style =
    data.dihedral_style = data.improper_style = NULL;

  int flag;
  flag = read_int(fp);

  while (flag >= 0) {

    if (flag == PAIR) {
      data.pair_style = read_char(fp);
      pair(fp,data,data.pair_style,1);
    } else if (flag == BOND) {
      data.bond_style = read_char(fp);
      bond(fp,data);
    } else if (flag == ANGLE) {
      data.angle_style = read_char(fp);
      angle(fp,data);
    } else if (flag == DIHEDRAL) {
      data.dihedral_style = read_char(fp);
      dihedral(fp,data);
    } else if (flag == IMPROPER) {
      data.improper_style = read_char(fp);
      improper(fp,data);
    } else {
      printf("ERROR: Invalid flag in force fields section of restart file %d\n",
	     flag);
      exit(1);
    }

    flag = read_int(fp);
  }
}

// ---------------------------------------------------------------------
// read fix info from restart file, just ignore it
// ---------------------------------------------------------------------

void modify(FILE *fp)
{
  char *buf;
  int n;

  // nfix = # of fix entries with state

  int nfix = read_int(fp);

  // read each entry with id string, style string, chunk of data

  for (int i = 0; i < nfix; i++) {
    buf = read_char(fp); delete [] buf;
    buf = read_char(fp); delete [] buf;
    buf = read_char(fp); delete [] buf;
  }

  // nfix = # of fix entries with peratom info

  int nfix_peratom = read_int(fp);

  // read each entry with id string, style string, maxsize of one atom data

  for (int i = 0; i < nfix_peratom; i++) {
    buf = read_char(fp); delete [] buf;
    buf = read_char(fp); delete [] buf;
    n = read_int(fp);
  }
}

// ---------------------------------------------------------------------
// read atom info from restart file and store in data struct
// ---------------------------------------------------------------------

int atom(double *buf, Data &data)
{
  // allocate per-atom arrays

  if (data.iatoms == 0) {

    // common to all atom styles

    data.x = new double[data.natoms];
    data.y = new double[data.natoms];
    data.z = new double[data.natoms];
    data.tag = new int[data.natoms];
    data.type = new int[data.natoms];
    data.mask = new int[data.natoms];
    data.image = new int[data.natoms];
    data.vx = new double[data.natoms];
    data.vy = new double[data.natoms];
    data.vz = new double[data.natoms];

    // style-specific arrays
    // don't worry about re-allocating if style = hybrid

    if (data.style_angle) allocate_angle(data);
    if (data.style_atomic) allocate_atomic(data);
    if (data.style_bond) allocate_bond(data);
    if (data.style_charge) allocate_charge(data);
    if (data.style_dipole) allocate_dipole(data);
    if (data.style_dpd) allocate_dpd(data);
    if (data.style_ellipsoid) allocate_ellipsoid(data);
    if (data.style_full) allocate_full(data);
    if (data.style_granular) allocate_granular(data);
    if (data.style_molecular) allocate_molecular(data);
    if (data.style_peri) allocate_peri(data);
    if (data.style_electron) allocate_electron(data);
  }

  // read atom quantities from buf
  // if hybrid, loop over all sub-styles in order listed
  // if hybrid, loop index k will match style setting to insure correct order

  int nloop = 1;
  if (data.style_hybrid) nloop = data.style_hybrid;

  int iatoms = data.iatoms;
  int m = 0;
  for (int k = 1; k <= nloop; k++) {
    if (k == data.style_angle) m += atom_angle(&buf[m],data,iatoms);
    if (k == data.style_atomic) m += atom_atomic(&buf[m],data,iatoms);
    if (k == data.style_bond) m += atom_bond(&buf[m],data,iatoms);
    if (k == data.style_charge) m += atom_charge(&buf[m],data,iatoms);
    if (k == data.style_dipole) m += atom_dipole(&buf[m],data,iatoms);
    if (k == data.style_dpd) m += atom_dpd(&buf[m],data,iatoms);
    if (k == data.style_ellipsoid) m += atom_ellipsoid(&buf[m],data,iatoms);
    if (k == data.style_full) m += atom_full(&buf[m],data,iatoms);
    if (k == data.style_granular) m += atom_granular(&buf[m],data,iatoms);
    if (k == data.style_molecular) m += atom_molecular(&buf[m],data,iatoms);
    if (k == data.style_peri) m += atom_peri(&buf[m],data,iatoms);
    if (k == data.style_electron) m += atom_electron(&buf[m],data,iatoms);
  }

  data.iatoms++;
  m = static_cast<int> (buf[0]);
  return m;
}

// ---------------------------------------------------------------------
// read one atom's info from buffer
// one routine per atom style
// ---------------------------------------------------------------------

int atom_angle(double *buf, Data &data, int iatoms)
{
  int type,atom1,atom2,atom3;

  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];
  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);
  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  data.molecule[iatoms] = static_cast<int> (buf[m++]);

  int n = static_cast<int> (buf[m++]);
  for (int k = 0; k < n; k++) {
    type = static_cast<int> (buf[m++]);
    atom1 = static_cast<int> (buf[m++]);
    if (data.newton_bond || data.tag[iatoms] < atom1) {
      data.bond_type[data.ibonds] = type;
      data.bond_atom1[data.ibonds] = data.tag[iatoms];
      data.bond_atom2[data.ibonds] = atom1;
      data.ibonds++;
    }
  }

  n = static_cast<int> (buf[m++]);
  for (int k = 0; k < n; k++) {
    type = static_cast<int> (buf[m++]);
    atom1 = static_cast<int> (buf[m++]);
    atom2 = static_cast<int> (buf[m++]);
    atom3 = static_cast<int> (buf[m++]);
    if (data.newton_bond || data.tag[iatoms] == atom2) {
      data.angle_type[data.iangles] = type;
      data.angle_atom1[data.iangles] = atom1;
      data.angle_atom2[data.iangles] = atom2;
      data.angle_atom3[data.iangles] = atom3;
	  data.iangles++;
    }
  }

  return m;
}

int atom_atomic(double *buf, Data &data, int iatoms)
{
  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];
  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);
  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  return m;
}

int atom_bond(double *buf, Data &data, int iatoms)
{
  int type,atom1;

  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];
  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);
  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  data.molecule[iatoms] = static_cast<int> (buf[m++]);

  int n = static_cast<int> (buf[m++]);
  for (int k = 0; k < n; k++) {
    type = static_cast<int> (buf[m++]);
    atom1 = static_cast<int> (buf[m++]);
    if (data.newton_bond || data.tag[iatoms] < atom1) {
      data.bond_type[data.ibonds] = type;
      data.bond_atom1[data.ibonds] = data.tag[iatoms];
      data.bond_atom2[data.ibonds] = atom1;
      data.ibonds++;
    }
  }

  return m;
}

int atom_charge(double *buf, Data &data, int iatoms)
{
  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];
  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);
  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  data.q[iatoms] = buf[m++];

  return m;
}

int atom_dipole(double *buf, Data &data, int iatoms)
{
  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];
  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);
  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  data.q[iatoms] = buf[m++];
  data.mux[iatoms] = buf[m++];
  data.muy[iatoms] = buf[m++];
  data.muz[iatoms] = buf[m++];
  
  return m;
}

int atom_dpd(double *buf, Data &data, int iatoms)
{
  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];
  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);
  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  return m;
}

int atom_ellipsoid(double *buf, Data &data, int iatoms)
{
  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];
  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);
  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  data.quatw[iatoms] = buf[m++];
  data.quati[iatoms] = buf[m++];
  data.quatj[iatoms] = buf[m++];
  data.quatk[iatoms] = buf[m++];
  data.angmomx[iatoms] = buf[m++];
  data.angmomy[iatoms] = buf[m++];
  data.angmomz[iatoms] = buf[m++];
  
  return m;
}

int atom_granular(double *buf, Data &data, int iatoms)
{
  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];
  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);
  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  data.radius[iatoms] = buf[m++];
  data.density[iatoms] = buf[m++];
  data.omegax[iatoms] = buf[m++];
  data.omegay[iatoms] = buf[m++];
  data.omegaz[iatoms] = buf[m++];

  return m;
}

int atom_full(double *buf, Data &data, int iatoms)
{
  int type,atom1,atom2,atom3,atom4;

  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];
  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);
  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  data.q[iatoms] = buf[m++];
  data.molecule[iatoms] = static_cast<int> (buf[m++]);

  int n = static_cast<int> (buf[m++]);
  for (int k = 0; k < n; k++) {
    type = static_cast<int> (buf[m++]);
    atom1 = static_cast<int> (buf[m++]);
    if (data.newton_bond || data.tag[iatoms] < atom1) {
      data.bond_type[data.ibonds] = type;
      data.bond_atom1[data.ibonds] = data.tag[iatoms];
      data.bond_atom2[data.ibonds] = atom1;
      data.ibonds++;
    }
  }

  n = static_cast<int> (buf[m++]);
  for (int k = 0; k < n; k++) {
    type = static_cast<int> (buf[m++]);
    atom1 = static_cast<int> (buf[m++]);
    atom2 = static_cast<int> (buf[m++]);
    atom3 = static_cast<int> (buf[m++]);
    if (data.newton_bond || data.tag[iatoms] == atom2) {
      data.angle_type[data.iangles] = type;
      data.angle_atom1[data.iangles] = atom1;
      data.angle_atom2[data.iangles] = atom2;
      data.angle_atom3[data.iangles] = atom3;
      data.iangles++;
    }
  }

  n = static_cast<int> (buf[m++]);
  for (int k = 0; k < n; k++) {
    type = static_cast<int> (buf[m++]);
    atom1 = static_cast<int> (buf[m++]);
    atom2 = static_cast<int> (buf[m++]);
    atom3 = static_cast<int> (buf[m++]);
    atom4 = static_cast<int> (buf[m++]);
    if (data.newton_bond || data.tag[iatoms] == atom2) {
      data.dihedral_type[data.idihedrals] = type;
      data.dihedral_atom1[data.idihedrals] = atom1;
      data.dihedral_atom2[data.idihedrals] = atom2;
      data.dihedral_atom3[data.idihedrals] = atom3;
      data.dihedral_atom4[data.idihedrals] = atom4;
      data.idihedrals++;
    }
  }
    
  n = static_cast<int> (buf[m++]);
  for (int k = 0; k < n; k++) {
    type = static_cast<int> (buf[m++]);
    atom1 = static_cast<int> (buf[m++]);
    atom2 = static_cast<int> (buf[m++]);
    atom3 = static_cast<int> (buf[m++]);
    atom4 = static_cast<int> (buf[m++]);
    if (data.newton_bond || data.tag[iatoms] == atom2) {
      data.improper_type[data.iimpropers] = type;
      data.improper_atom1[data.iimpropers] = atom1;
      data.improper_atom2[data.iimpropers] = atom2;
      data.improper_atom3[data.iimpropers] = atom3;
      data.improper_atom4[data.iimpropers] = atom4;
      data.iimpropers++;
    }
  }

  return m;
}

int atom_molecular(double *buf, Data &data, int iatoms)
{
  int type,atom1,atom2,atom3,atom4;

  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];
  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);
  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  data.molecule[iatoms] = static_cast<int> (buf[m++]);

  int n = static_cast<int> (buf[m++]);
  for (int k = 0; k < n; k++) {
    type = static_cast<int> (buf[m++]);
    atom1 = static_cast<int> (buf[m++]);
    if (data.newton_bond || data.tag[iatoms] < atom1) {
      data.bond_type[data.ibonds] = type;
      data.bond_atom1[data.ibonds] = data.tag[iatoms];
      data.bond_atom2[data.ibonds] = atom1;
      data.ibonds++;
    }
  }

  n = static_cast<int> (buf[m++]);
  for (int k = 0; k < n; k++) {
    type = static_cast<int> (buf[m++]);
    atom1 = static_cast<int> (buf[m++]);
    atom2 = static_cast<int> (buf[m++]);
    atom3 = static_cast<int> (buf[m++]);
    if (data.newton_bond || data.tag[iatoms] == atom2) {
      data.angle_type[data.iangles] = type;
      data.angle_atom1[data.iangles] = atom1;
      data.angle_atom2[data.iangles] = atom2;
      data.angle_atom3[data.iangles] = atom3;
      data.iangles++;
    }
  }

  n = static_cast<int> (buf[m++]);
  for (int k = 0; k < n; k++) {
    type = static_cast<int> (buf[m++]);
    atom1 = static_cast<int> (buf[m++]);
    atom2 = static_cast<int> (buf[m++]);
    atom3 = static_cast<int> (buf[m++]);
    atom4 = static_cast<int> (buf[m++]);
    if (data.newton_bond || data.tag[iatoms] == atom2) {
      data.dihedral_type[data.idihedrals] = type;
      data.dihedral_atom1[data.idihedrals] = atom1;
      data.dihedral_atom2[data.idihedrals] = atom2;
      data.dihedral_atom3[data.idihedrals] = atom3;
      data.dihedral_atom4[data.idihedrals] = atom4;
      data.idihedrals++;
    }
  }

  n = static_cast<int> (buf[m++]);
  for (int k = 0; k < n; k++) {
    type = static_cast<int> (buf[m++]);
    atom1 = static_cast<int> (buf[m++]);
    atom2 = static_cast<int> (buf[m++]);
    atom3 = static_cast<int> (buf[m++]);
    atom4 = static_cast<int> (buf[m++]);
    if (data.newton_bond || data.tag[iatoms] == atom2) {
      data.improper_type[data.iimpropers] = type;
      data.improper_atom1[data.iimpropers] = atom1;
      data.improper_atom2[data.iimpropers] = atom2;
      data.improper_atom3[data.iimpropers] = atom3;
      data.improper_atom4[data.iimpropers] = atom4;
      data.iimpropers++;
    }
  }

  return m;
}

int atom_peri(double *buf, Data &data, int iatoms)
{
  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];
  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);
  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  data.vfrac[iatoms] = buf[m++];
  data.density[iatoms] = buf[m++];
  data.rmass[iatoms] = buf[m++];
  data.s0[iatoms] = buf[m++];
  data.x0x[iatoms] = buf[m++];
  data.x0y[iatoms] = buf[m++];
  data.x0z[iatoms] = buf[m++];

  return m;
}

int atom_electron(double *buf, Data &data, int iatoms)
{
  int m = 1;
  data.x[iatoms] = buf[m++];
  data.y[iatoms] = buf[m++];
  data.z[iatoms] = buf[m++];

  data.tag[iatoms] = static_cast<int> (buf[m++]);
  data.type[iatoms] = static_cast<int> (buf[m++]);
  data.mask[iatoms] = static_cast<int> (buf[m++]);
  data.image[iatoms] = static_cast<int> (buf[m++]);

  data.vx[iatoms] = buf[m++];
  data.vy[iatoms] = buf[m++];
  data.vz[iatoms] = buf[m++];

  data.q[iatoms] = buf[m++];
  data.spin[iatoms] = static_cast<int> (buf[m++]);
  data.radius[iatoms] = buf[m++];
  data.ervel[iatoms] = buf[m++];

  return m;
}


// ---------------------------------------------------------------------
// per-atom memory allocation routines
// one routine per atom style
// ---------------------------------------------------------------------

void allocate_angle(Data &data)
{
  data.molecule = new int[data.natoms];
  data.bond_type = new int[data.nbonds];
  data.bond_atom1 = new int[data.nbonds];
  data.bond_atom2 = new int[data.nbonds];
  data.angle_type = new int[data.nangles];
  data.angle_atom1 = new int[data.nangles];
  data.angle_atom2 = new int[data.nangles];
  data.angle_atom3 = new int[data.nangles];
}

void allocate_atomic(Data &data) {}

void allocate_bond(Data &data)
{
  data.molecule = new int[data.natoms];
  data.bond_type = new int[data.nbonds];
  data.bond_atom1 = new int[data.nbonds];
  data.bond_atom2 = new int[data.nbonds];
}

void allocate_charge(Data &data)
{
  data.q = new double[data.natoms];
}

void allocate_dipole(Data &data)
{
  data.q = new double[data.natoms];
  data.mux = new double[data.natoms];
  data.muy = new double[data.natoms];
  data.muz = new double[data.natoms];
}

void allocate_dpd(Data &data) {}

void allocate_full(Data &data)
{
  data.q = new double[data.natoms];
  data.molecule = new int[data.natoms];
  data.bond_type = new int[data.nbonds];
  data.bond_atom1 = new int[data.nbonds];
  data.bond_atom2 = new int[data.nbonds];
  data.angle_type = new int[data.nangles];
  data.angle_atom1 = new int[data.nangles];
  data.angle_atom2 = new int[data.nangles];
  data.angle_atom3 = new int[data.nangles];
  data.dihedral_type = new int[data.ndihedrals];
  data.dihedral_atom1 = new int[data.ndihedrals];
  data.dihedral_atom2 = new int[data.ndihedrals];
  data.dihedral_atom3 = new int[data.ndihedrals];
  data.dihedral_atom4 = new int[data.ndihedrals];
  data.improper_type = new int[data.nimpropers];
  data.improper_atom1 = new int[data.nimpropers];
  data.improper_atom2 = new int[data.nimpropers];
  data.improper_atom3 = new int[data.nimpropers];
  data.improper_atom4 = new int[data.nimpropers];
}

void allocate_ellipsoid(Data &data)
{
  data.quatw = new double[data.natoms];
  data.quati = new double[data.natoms];
  data.quatj = new double[data.natoms];
  data.quatk = new double[data.natoms];
  data.angmomx = new double[data.natoms];
  data.angmomy = new double[data.natoms];
  data.angmomz = new double[data.natoms];
}

void allocate_granular(Data &data)
{
  data.radius = new double[data.natoms];
  data.density = new double[data.natoms];
  data.omegax = new double[data.natoms];
  data.omegay = new double[data.natoms];
  data.omegaz = new double[data.natoms];
}

void allocate_molecular(Data &data)
{
  data.molecule = new int[data.natoms];
  data.bond_type = new int[data.nbonds];
  data.bond_atom1 = new int[data.nbonds];
  data.bond_atom2 = new int[data.nbonds];
  data.angle_type = new int[data.nangles];
  data.angle_atom1 = new int[data.nangles];
  data.angle_atom2 = new int[data.nangles];
  data.angle_atom3 = new int[data.nangles];
  data.dihedral_type = new int[data.ndihedrals];
  data.dihedral_atom1 = new int[data.ndihedrals];
  data.dihedral_atom2 = new int[data.ndihedrals];
  data.dihedral_atom3 = new int[data.ndihedrals];
  data.dihedral_atom4 = new int[data.ndihedrals];
  data.improper_type = new int[data.nimpropers];
  data.improper_atom1 = new int[data.nimpropers];
  data.improper_atom2 = new int[data.nimpropers];
  data.improper_atom3 = new int[data.nimpropers];
  data.improper_atom4 = new int[data.nimpropers];
}

void allocate_peri(Data &data)
{
  data.vfrac = new double[data.natoms];
  data.density = new double[data.natoms];
  data.rmass = new double[data.natoms];
  data.s0 = new double[data.natoms];
  data.x0x = new double[data.natoms];
  data.x0y = new double[data.natoms];
  data.x0z = new double[data.natoms];
}

void allocate_electron(Data &data)
{
  data.q = new double[data.natoms];
  data.spin = new int[data.natoms];
  data.radius = new double[data.natoms];
  data.ervel = new double[data.natoms];
}


// ---------------------------------------------------------------------
// pair coeffs
// one section for each pair style
// flag = 1, read all coeff info and allocation arrays
// flag = 0, just read global settings (when called recursively by hybrid)
// ---------------------------------------------------------------------

void pair(FILE *fp, Data &data, char *style, int flag)
{
  int i,j,m;
  int itmp;
  double rtmp;

  if (strcmp(style,"none") == 0) {

  } else if (strcmp(style,"airebo") == 0) {

  } else if (strcmp(style,"born/coul/long") == 0) {
    double cut_lj_global = read_double(fp);
    double cut_coul = read_double(fp);
    int offset_flag = read_int(fp);
    int mix_flag = read_int(fp);

    if (!flag) return;

    data.pair_born_A = new double[data.ntypes+1];
    data.pair_born_rho = new double[data.ntypes+1];
    data.pair_born_sigma = new double[data.ntypes+1];
    data.pair_born_C = new double[data.ntypes+1];
    data.pair_born_D = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_born_A[i] = read_double(fp);
	    data.pair_born_rho[i] = read_double(fp);
	    data.pair_born_sigma[i] = read_double(fp);
	    data.pair_born_C[i] = read_double(fp);
	    data.pair_born_D[i] = read_double(fp);
	    double cut_lj = read_double(fp);
	  } else {
	    double born_A = read_double(fp);
	    double born_rho = read_double(fp);
	    double born_sigma = read_double(fp);
	    double born_C = read_double(fp);
	    double born_D = read_double(fp);
	    double cut_lj = read_double(fp);
	  }
	}
      }

  } else if ((strcmp(style,"buck") == 0)  ||
	   (strcmp(style,"buck/coul/cut") == 0) ||
	   (strcmp(style,"buck/coul/long") == 0) ||
	   (strcmp(style,"buck/coul") == 0)) {

    if (strcmp(style,"buck") == 0) {
      m = 0;
      double cut_lj_global = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"buck/coul/cut") == 0) {
      m = 1;
      double cut_lj_global = read_double(fp);
      double cut_lj_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"buck/coul/long") == 0) {
      m = 0;
      double cut_lj_global = read_double(fp);
      double cut_lj_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"buck/coul") == 0) {
      m = 0;
      double cut_buck_global = read_double(fp);
      double cut_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
      int ewald_order = read_int(fp);
    }

    if (!flag) return;

    data.pair_buck_A = new double[data.ntypes+1];
    data.pair_buck_rho = new double[data.ntypes+1];
    data.pair_buck_C = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_buck_A[i] = read_double(fp);
	    data.pair_buck_rho[i] = read_double(fp);
	    data.pair_buck_C[i] = read_double(fp);
	    double cut_lj = read_double(fp);
	    if (m) double cut_coul = read_double(fp);
	  } else {
	    double buck_A = read_double(fp);
	    double buck_rho = read_double(fp);
	    double buck_C = read_double(fp);
	    double cut_lj = read_double(fp);
	    if (m) double cut_coul = read_double(fp);
	  }
	}
      }

  } else if (strcmp(style,"colloid") == 0) {

    double cut_global = read_double(fp);
    int offset_flag = read_int(fp);
    int mix_flag = read_int(fp);

    if (!flag) return;

    data.pair_colloid_A12 = new double[data.ntypes+1];
    data.pair_colloid_sigma = new double[data.ntypes+1];
    data.pair_colloid_d1 = new double[data.ntypes+1];
    data.pair_colloid_d2 = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_colloid_A12[i] = read_double(fp);
	    data.pair_colloid_sigma[i] = read_double(fp);
	    data.pair_colloid_d1[i] = read_double(fp);
	    data.pair_colloid_d2[i] = read_double(fp);
	    double cut_lj = read_double(fp);
	  } else {
	    double colloid_A12 = read_double(fp);
	    double colloid_sigma = read_double(fp);
	    double colloid_d1 = read_double(fp);
	    double colloid_d2 = read_double(fp);
	    double cut_lj = read_double(fp);
	  }
	}
      }
    
  } else if ((strcmp(style,"coul/cut") == 0) ||
	     (strcmp(style,"coul/debye") == 0) ||
	     (strcmp(style,"coul/long") == 0)) {

    if (strcmp(style,"coul/cut") == 0) {
      double cut_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"coul/debye") == 0) {
      m = 1;
      double cut_coul = read_double(fp);
      double kappa = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"coul/long") == 0) {
      double cut_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    }

    if (!flag) return;

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    double cut_coul = read_double(fp);
	  } else {
	    double cut_coul = read_double(fp);
	  }
	}
      }

  } else if (strcmp(style,"eff/cut") == 0) {

    if (strcmp(style,"eff/cut") == 0) {
      double cut_eff = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    }

    if (!flag) return;

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
        itmp = read_int(fp);
        if (i == j && itmp == 0) {
          printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
          exit(1);
        }
        if (itmp) {
          if (i == j) {
            double cut_eff = read_double(fp);
          } else {
            double cut_eff = read_double(fp);
          }
        }
      }

  }  else if (strcmp(style,"dipole/cut") == 0) {

    double cut_lj_global = read_double(fp);
    double cut_coul_global = read_double(fp);
    int offset_flag = read_int(fp);
    int mix_flag = read_int(fp);

    if (!flag) return;

    data.pair_dipole_epsilon = new double[data.ntypes+1];
    data.pair_dipole_sigma = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_dipole_epsilon[i] = read_double(fp);
	    data.pair_dipole_sigma[i] = read_double(fp);
	    double cut_lj = read_double(fp);
	    double cut_coul = read_double(fp);
	  } else {
	    double dipole_epsilon = read_double(fp);
	    double dipole_sigma = read_double(fp);
	    double cut_lj = read_double(fp);
	    double cut_coul = read_double(fp);
	  }
	}
      }

  } else if (strcmp(style,"dpd") == 0) {

    double temperature = read_double(fp);
    double cut_global = read_double(fp);
    int seed = read_int(fp);
    int mix_flag = read_int(fp);

    if (!flag) return;

    data.pair_dpd_a0 = new double[data.ntypes+1];
    data.pair_dpd_gamma = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_dpd_a0[i] = read_double(fp);
	    data.pair_dpd_gamma[i] = read_double(fp);
	    double cut = read_double(fp);
	  } else {
	    double dpd_a0 = read_double(fp);
	    double dpd_gamma = read_double(fp);
	    double cut = read_double(fp);
	  }
	}
      }

  } else if (strcmp(style,"eam") == 0) {
  } else if (strcmp(style,"eam/opt") == 0) {
  } else if (strcmp(style,"eam/alloy") == 0) {
  } else if (strcmp(style,"eam/alloy/opt") == 0) {
  } else if (strcmp(style,"eam/fs") == 0) {
  } else if (strcmp(style,"eam/fs/opt") == 0) {

  } else if (strcmp(style,"gayberne") == 0) {

    double gamma = read_double(fp);
    double upsilon = read_double(fp);
    double mu = read_double(fp);
    double cut_global = read_double(fp);
    int offset_flag = read_int(fp);
    int mix_flag = read_int(fp);

    if (!flag) return;

    data.pair_gb_epsilon = new double[data.ntypes+1];
    data.pair_gb_sigma = new double[data.ntypes+1];
    data.pair_gb_epsa = new double[data.ntypes+1];
    data.pair_gb_epsb = new double[data.ntypes+1];
    data.pair_gb_epsc = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++) {
      itmp = read_int(fp);
      if (itmp) {
	data.pair_gb_epsa[i] = read_double(fp);
	data.pair_gb_epsb[i] = read_double(fp);
	data.pair_gb_epsc[i] = read_double(fp);
	data.pair_gb_epsa[i] = pow(data.pair_gb_epsa[i],-mu);
	data.pair_gb_epsb[i] = pow(data.pair_gb_epsb[i],-mu);
	data.pair_gb_epsc[i] = pow(data.pair_gb_epsc[i],-mu);
      }
      
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_gb_epsilon[i] = read_double(fp);
	    data.pair_gb_sigma[i] = read_double(fp);
	    double cut = read_double(fp);
	  } else {
	    double gb_epsilon = read_double(fp);
	    double gb_sigma = read_double(fp);
	    double cut = read_double(fp);
	  }
	}
      }
    }

  } else if ((strcmp(style,"gran/hooke") == 0) ||
	   (strcmp(style,"gran/hooke/history") == 0) ||
	   (strcmp(style,"gran/hertz/history") == 0)) {

    double kn = read_double(fp);
    double kt = read_double(fp);
    double gamman = read_double(fp);
    double gammat = read_double(fp);
    double xmu = read_double(fp);
    int dampflag = read_int(fp);

  } else if ((strcmp(style,"lj/charmm/coul/charmm") == 0) ||
	   (strcmp(style,"lj/charmm/coul/charmm/implicit") == 0) ||
	   (strcmp(style,"lj/charmm/coul/long") == 0) ||
	   (strcmp(style,"lj/charmm/coul/long/opt") == 0)) {

    if (strcmp(style,"lj/charmm/coul/charmm") == 0) {
      double cut_lj_inner = read_double(fp);
      double cut_lj = read_double(fp);
      double cut_coul_inner = read_double(fp);
      double cut_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"lj/charmm/coul/charmm/implicit") == 0) {
      double cut_lj_inner = read_double(fp);
      double cut_lj = read_double(fp);
      double cut_coul_inner = read_double(fp);
      double cut_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if ((strcmp(style,"lj/charmm/coul/long") == 0) ||
	       (strcmp(style,"lj/charmm/coul/long/opt") == 0)) {
      double cut_lj_inner = read_double(fp);
      double cut_lj = read_double(fp);
      double cut_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    }

    if (!flag) return;

    data.pair_charmm_epsilon = new double[data.ntypes+1];
    data.pair_charmm_sigma = new double[data.ntypes+1];
    data.pair_charmm_eps14 = new double[data.ntypes+1];
    data.pair_charmm_sigma14 = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_charmm_epsilon[i] = read_double(fp);
	    data.pair_charmm_sigma[i] = read_double(fp);
	    data.pair_charmm_eps14[i] = read_double(fp);
	    data.pair_charmm_sigma14[i] = read_double(fp);
	  } else {
	    double charmm_epsilon = read_double(fp);
	    double charmm_sigma = read_double(fp);
	    double charmm_eps14 = read_double(fp);
	    double charmm_sigma14 = read_double(fp);
	  }
	}
      }

  } else if ((strcmp(style,"lj/class2") == 0) ||
	   (strcmp(style,"lj/class2/coul/cut") == 0) ||
	   (strcmp(style,"lj/class2/coul/long") == 0)) {

    if (strcmp(style,"lj/class2") == 0) {
      m = 0;
      double cut_lj_global = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"lj/class2/coul/cut") == 0) {
      m = 1;
      double cut_lj_global = read_double(fp);
      double cut_lj_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"lj/class2/coul/long") == 0) {
      m = 0;
      double cut_lj_global = read_double(fp);
      double cut_lj_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    }

    if (!flag) return;

    data.pair_class2_epsilon = new double[data.ntypes+1];
    data.pair_class2_sigma = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_class2_epsilon[i] = read_double(fp);
	    data.pair_class2_sigma[i] = read_double(fp);
	    double cut_lj = read_double(fp);
	    if (m) double cut_coul = read_double(fp);
	  } else {
	    double class2_epsilon = read_double(fp);
	    double class2_sigma = read_double(fp);
	    double cut_lj = read_double(fp);
	    if (m) double cut_coul = read_double(fp);
	  }
	}
      }

  } else if ((strcmp(style,"lj/cut") == 0) ||
	   (strcmp(style,"lj/cut/opt") == 0) ||
	   (strcmp(style,"lj/cut/coul/cut") == 0) ||
	   (strcmp(style,"lj/cut/coul/debye") == 0) ||
	   (strcmp(style,"lj/cut/coul/long") == 0) ||
	   (strcmp(style,"lj/cut/coul/long/tip4p") == 0) ||
	   (strcmp(style,"lj/coul") == 0)) {

    if ((strcmp(style,"lj/cut") == 0) || (strcmp(style,"lj/cut/opt") == 0)) {
      m = 0;
      double cut_lj_global = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"lj/cut/coul/cut") == 0) {
      m = 1;
      double cut_lj_global = read_double(fp);
      double cut_lj_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"lj/cut/coul/debye") == 0) {
      m = 1;
      double cut_lj_global = read_double(fp);
      double cut_lj_coul = read_double(fp);
      double kappa = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"lj/cut/coul/long") == 0) {
      m = 0;
      double cut_lj_global = read_double(fp);
      double cut_lj_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"lj/cut/coul/long/tip4p") == 0) {
      m = 0;
      int typeO = read_int(fp);
      int typeH = read_int(fp);
      int typeB = read_int(fp);
      int typeA = read_int(fp);
      double qdist = read_double(fp);
      double cut_lj_global = read_double(fp);
      double cut_lj_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"lj/coul") == 0) {
      m = 0;
      double cut_lj_global = read_double(fp);
      double cut_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
      int ewald_order = read_int(fp);
    }

    if (!flag) return;

    data.pair_lj_epsilon = new double[data.ntypes+1];
    data.pair_lj_sigma = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_lj_epsilon[i] = read_double(fp);
	    data.pair_lj_sigma[i] = read_double(fp);
	    double cut_lj = read_double(fp);
	    if (m) double cut_coul = read_double(fp);
	  } else {
	    double lj_epsilon = read_double(fp);
	    double lj_sigma = read_double(fp);
	    double cut_lj = read_double(fp);
	    if (m) double cut_coul = read_double(fp);
	  }
	}
      }

  } else if (strcmp(style,"lj/expand") == 0) {

    double cut_global = read_double(fp);
    int offset_flag = read_int(fp);
    int mix_flag = read_int(fp);

    if (!flag) return;

    data.pair_ljexpand_epsilon = new double[data.ntypes+1];
    data.pair_ljexpand_sigma = new double[data.ntypes+1];
    data.pair_ljexpand_shift = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_ljexpand_epsilon[i] = read_double(fp);
	    data.pair_ljexpand_sigma[i] = read_double(fp);
	    data.pair_ljexpand_shift[i] = read_double(fp);
	    double cut_lj = read_double(fp);
	  } else {
	    double ljexpand_epsilon = read_double(fp);
	    double ljexpand_sigma = read_double(fp);
	    double ljexpand_shift = read_double(fp);
	    double cut_lj = read_double(fp);
	  }
	}
      }

  } else if ((strcmp(style,"lj/gromacs") == 0) ||
	     (strcmp(style,"lj/gromacs/coul/gromacs") == 0)) {

    if (strcmp(style,"lj/gromacs") == 0) {
      m = 1;
      double cut_inner_global = read_double(fp);
      double cut_global = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    } else if (strcmp(style,"lj/gromacs/coul/gromacs") == 0) {
      m = 0;
      double cut_lj_inner_global = read_double(fp);
      double cut_lj = read_double(fp);
      double cut_coul_inner_global = read_double(fp);
      double cut_coul = read_double(fp);
      int offset_flag = read_int(fp);
      int mix_flag = read_int(fp);
    }

    if (!flag) return;

    data.pair_ljgromacs_epsilon = new double[data.ntypes+1];
    data.pair_ljgromacs_sigma = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_ljgromacs_epsilon[i] = read_double(fp);
	    data.pair_ljgromacs_sigma[i] = read_double(fp);
	    if (m) {
	      double cut_inner = read_double(fp);
	      double cut = read_double(fp);
	    }
	    } else {
	    double ljgromacs_epsilon = read_double(fp);
	    double ljgromacs_sigma = read_double(fp);
	    if (m) {
	      double cut_inner = read_double(fp);
	      double cut = read_double(fp);
	    }
	  }
	}
      }

  } else if (strcmp(style,"lj/smooth") == 0) {

    double cut_inner_global = read_double(fp);
    double cut_global = read_double(fp);
    int offset_flag = read_int(fp);
    int mix_flag = read_int(fp);

    if (!flag) return;

    data.pair_ljsmooth_epsilon = new double[data.ntypes+1];
    data.pair_ljsmooth_sigma = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_ljsmooth_epsilon[i] = read_double(fp);
	    data.pair_ljsmooth_sigma[i] = read_double(fp);
	    double cut_inner = read_double(fp);
	    double cut = read_double(fp);
	  } else {
	    double ljsmooth_epsilon = read_double(fp);
	    double ljsmooth_sigma = read_double(fp);
	    double cut_inner = read_double(fp);
	    double cut = read_double(fp);
	  }
	}
      }

  } else if (strcmp(style,"meam") == 0) {

  } else if ((strcmp(style,"morse") == 0) ||
	     (strcmp(style,"morse/opt") == 0)) {

    double cut_global = read_double(fp);
    int offset_flag = read_int(fp);
    int mix_flag = read_int(fp);

    if (!flag) return;

    data.pair_morse_d0 = new double[data.ntypes+1];
    data.pair_morse_alpha = new double[data.ntypes+1];
    data.pair_morse_r0 = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_morse_d0[i] = read_double(fp);
	    data.pair_morse_alpha[i] = read_double(fp);
	    data.pair_morse_r0[i] = read_double(fp);
	    double cut = read_double(fp);
	  } else {
	    double morse_d0 = read_double(fp);
	    double morse_alpha = read_double(fp);
	    double morse_r0 = read_double(fp);
	    double cut = read_double(fp);
	  }
	}
      }

  } else if (strcmp(style,"reax") == 0) {

  } else if (strcmp(style,"soft") == 0) {

    double cut_global = read_double(fp);
    int mix_flag = read_int(fp);

    if (!flag) return;

    data.pair_soft_start = new double[data.ntypes+1];
    data.pair_soft_stop = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_soft_start[i] = read_double(fp);
	    data.pair_soft_stop[i] = read_double(fp);
	    double cut = read_double(fp);
	  } else {
	    double soft_start = read_double(fp);
	    double soft_stop = read_double(fp);
	    double cut = read_double(fp);
	  }
	}
      }

  } else if (strcmp(style,"sw") == 0) {

  } else if (strcmp(style,"table") == 0) {

    int tabstyle = read_int(fp);
    int n = read_int(fp);

  } else if (strcmp(style,"tersoff") == 0) {

  } else if (strcmp(style,"yukawa") == 0) {

    double kappa = read_double(fp);
    double cut_global = read_double(fp);
    int offset_flag = read_int(fp);
    int mix_flag = read_int(fp);

    if (!flag) return;

    data.pair_yukawa_A = new double[data.ntypes+1];

    for (i = 1; i <= data.ntypes; i++)
      for (j = i; j <= data.ntypes; j++) {
	itmp = read_int(fp);
	if (i == j && itmp == 0) {
	  printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
	  exit(1);
	}
	if (itmp) {
	  if (i == j) {
	    data.pair_yukawa_A[i] = read_double(fp);
	    double cut = read_double(fp);
	  } else {
	    double yukawa_A = read_double(fp);
	    double cut = read_double(fp);
	  }
	}
      }

  } else if ((strcmp(style,"cg/cmm") == 0) ||
             (strcmp(style,"cg/cmm/coul/cut") == 0) ||
             (strcmp(style,"cg/cmm/coul/long") == 0) ) {
    m = 0;
    data.cut_lj_global = read_double(fp);
    data.cut_coul_global = read_double(fp);
    data.kappa = read_double(fp);
    data.offset_flag = read_int(fp);
    data.mix_flag = read_int(fp);

    if (!flag) return;

    const int numtyp=data.ntypes+1;
    data.pair_cg_cmm_type = new int*[numtyp];
    data.pair_setflag = new int*[numtyp];
    data.pair_cg_epsilon = new double*[numtyp];
    data.pair_cg_sigma = new double*[numtyp];
    data.pair_cut_lj = new double*[numtyp];
    if  ((strcmp(style,"cg/cmm/coul/cut") == 0) ||
             (strcmp(style,"cg/cmm/coul/long") == 0) ) {
      data.pair_cut_coul = new double*[numtyp];
      m=1;
    } else {
      data.pair_cut_coul = NULL;
      m=0;
    }
    
    for (i = 1; i <= data.ntypes; i++) {
      data.pair_cg_cmm_type[i] = new int[numtyp];
      data.pair_setflag[i] = new int[numtyp];
      data.pair_cg_epsilon[i] = new double[numtyp];
      data.pair_cg_sigma[i] = new double[numtyp];
      data.pair_cut_lj[i] = new double[numtyp];
      if ((strcmp(style,"cg/cmm/coul/cut") == 0) ||
          (strcmp(style,"cg/cmm/coul/long") == 0) ) {
        data.pair_cut_coul[i] = new double[numtyp];
      }

      for (j = i; j <= data.ntypes; j++) {
        itmp = read_int(fp);
        data.pair_setflag[i][j] = itmp;        
        if (i == j && itmp == 0) {
          printf("ERROR: Pair coeff %d,%d is not in restart file\n",i,j);
          exit(1);
        }
        if (itmp) {
          data.pair_cg_cmm_type[i][j] = read_int(fp);
          data.pair_cg_epsilon[i][j] = read_double(fp);
          data.pair_cg_sigma[i][j] = read_double(fp);
          data.pair_cut_lj[i][j] = read_double(fp);
          if (m) {
	    data.pair_cut_lj[i][j] = read_double(fp);
	    data.pair_cut_coul[i][j] = read_double(fp);
	  }
        } 
      }
    }

  } else if ((strcmp(style,"hybrid") == 0) ||
	     (strcmp(style,"hybrid/overlay") == 0)) {

    // for each substyle of hybrid,
    //   read its settings by calling pair() recursively with flag = 0
    //   so that coeff array allocation is skipped

    int nstyles = read_int(fp);
    for (int i = 0; i < nstyles; i++) {
      char *substyle = read_char(fp);
      pair(fp,data,substyle,0);
    }

  } else {
    printf("ERROR: Unknown pair style %s\n",style);
    exit(1);
  }
}

// ---------------------------------------------------------------------
// bond coeffs
// one section for each bond style
// ---------------------------------------------------------------------

void bond(FILE *fp, Data &data)
{
  if (strcmp(data.bond_style,"none") == 0) {

  } else if (strcmp(data.bond_style,"class2") == 0) {

    data.bond_class2_r0 = new double[data.nbondtypes+1];
    data.bond_class2_k2 = new double[data.nbondtypes+1];
    data.bond_class2_k3 = new double[data.nbondtypes+1];
    data.bond_class2_k4 = new double[data.nbondtypes+1];
    fread(&data.bond_class2_r0[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_class2_k2[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_class2_k3[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_class2_k4[1],sizeof(double),data.nbondtypes,fp);

  } else if (strcmp(data.bond_style,"fene") == 0) {

    data.bond_fene_k = new double[data.nbondtypes+1];
    data.bond_fene_r0 = new double[data.nbondtypes+1];
    data.bond_fene_epsilon = new double[data.nbondtypes+1];
    data.bond_fene_sigma = new double[data.nbondtypes+1];
    fread(&data.bond_fene_k[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_fene_r0[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_fene_epsilon[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_fene_sigma[1],sizeof(double),data.nbondtypes,fp);

  } else if (strcmp(data.bond_style,"fene/expand") == 0) {

    data.bond_feneexpand_k = new double[data.nbondtypes+1];
    data.bond_feneexpand_r0 = new double[data.nbondtypes+1];
    data.bond_feneexpand_epsilon = new double[data.nbondtypes+1];
    data.bond_feneexpand_sigma = new double[data.nbondtypes+1];
    data.bond_feneexpand_shift = new double[data.nbondtypes+1];
    fread(&data.bond_feneexpand_k[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_feneexpand_r0[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_feneexpand_epsilon[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_feneexpand_sigma[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_feneexpand_shift[1],sizeof(double),data.nbondtypes,fp);

  } else if (strcmp(data.bond_style,"harmonic") == 0) {

    data.bond_harmonic_k = new double[data.nbondtypes+1];
    data.bond_harmonic_r0 = new double[data.nbondtypes+1];
    fread(&data.bond_harmonic_k[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_harmonic_r0[1],sizeof(double),data.nbondtypes,fp);

  } else if (strcmp(data.bond_style,"morse") == 0) {

    data.bond_morse_d0 = new double[data.nbondtypes+1];
    data.bond_morse_alpha = new double[data.nbondtypes+1];
    data.bond_morse_r0 = new double[data.nbondtypes+1];
    fread(&data.bond_morse_d0[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_morse_alpha[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_morse_r0[1],sizeof(double),data.nbondtypes,fp);

  } else if (strcmp(data.bond_style,"nonlinear") == 0) {

    data.bond_nonlinear_epsilon = new double[data.nbondtypes+1];
    data.bond_nonlinear_r0 = new double[data.nbondtypes+1];
    data.bond_nonlinear_lamda = new double[data.nbondtypes+1];
    fread(&data.bond_nonlinear_epsilon[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_nonlinear_r0[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_nonlinear_lamda[1],sizeof(double),data.nbondtypes,fp);

  } else if (strcmp(data.bond_style,"quartic") == 0) {

    data.bond_quartic_k = new double[data.nbondtypes+1];
    data.bond_quartic_b1 = new double[data.nbondtypes+1];
    data.bond_quartic_b2 = new double[data.nbondtypes+1];
    data.bond_quartic_rc = new double[data.nbondtypes+1];
    data.bond_quartic_u0 = new double[data.nbondtypes+1];
    fread(&data.bond_quartic_k[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_quartic_b1[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_quartic_b2[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_quartic_rc[1],sizeof(double),data.nbondtypes,fp);
    fread(&data.bond_quartic_u0[1],sizeof(double),data.nbondtypes,fp);

  } else if (strcmp(data.bond_style,"hybrid") == 0) {

    int nstyles = read_int(fp);
    for (int i = 0; i < nstyles; i++)
      char *substyle = read_char(fp);

  } else {
    printf("ERROR: Unknown bond style %s\n",data.bond_style);
    exit(1);
  }
}

// ---------------------------------------------------------------------
// angle coeffs
// one section for each angle style
// ---------------------------------------------------------------------

void angle(FILE *fp, Data &data)
{
  if (strcmp(data.angle_style,"none") == 0) {

  } else if (strcmp(data.angle_style,"charmm") == 0) {

    data.angle_charmm_k = new double[data.nangletypes+1];
    data.angle_charmm_theta0 = new double[data.nangletypes+1];
    data.angle_charmm_k_ub = new double[data.nangletypes+1];
    data.angle_charmm_r_ub = new double[data.nangletypes+1];
    fread(&data.angle_charmm_k[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_charmm_theta0[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_charmm_k_ub[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_charmm_r_ub[1],sizeof(double),data.nangletypes,fp);

  } else if (strcmp(data.angle_style,"class2") == 0) {

    data.angle_class2_theta0 = new double[data.nangletypes+1];
    data.angle_class2_k2 = new double[data.nangletypes+1];
    data.angle_class2_k3 = new double[data.nangletypes+1];
    data.angle_class2_k4 = new double[data.nangletypes+1];

    data.angle_class2_bb_k = new double[data.nangletypes+1];
    data.angle_class2_bb_r1 = new double[data.nangletypes+1];
    data.angle_class2_bb_r2 = new double[data.nangletypes+1];

    data.angle_class2_ba_k1 = new double[data.nangletypes+1];
    data.angle_class2_ba_k2 = new double[data.nangletypes+1];
    data.angle_class2_ba_r1 = new double[data.nangletypes+1];
    data.angle_class2_ba_r2 = new double[data.nangletypes+1];

    fread(&data.angle_class2_theta0[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_class2_k2[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_class2_k3[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_class2_k4[1],sizeof(double),data.nangletypes,fp);

    fread(&data.angle_class2_bb_k[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_class2_bb_r1[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_class2_bb_r2[1],sizeof(double),data.nangletypes,fp);

    fread(&data.angle_class2_ba_k1[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_class2_ba_k2[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_class2_ba_r1[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_class2_ba_r2[1],sizeof(double),data.nangletypes,fp);

  } else if (strcmp(data.angle_style,"cosine") == 0) {

    data.angle_cosine_k = new double[data.nangletypes+1];
    fread(&data.angle_cosine_k[1],sizeof(double),data.nangletypes,fp);

  } else if ((strcmp(data.angle_style,"cosine/squared") == 0) ||
             (strcmp(data.angle_style,"cosine/delta") == 0)) {

    data.angle_cosine_squared_k = new double[data.nangletypes+1];
    data.angle_cosine_squared_theta0 = new double[data.nangletypes+1];
    fread(&data.angle_cosine_squared_k[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_cosine_squared_theta0[1],
	  sizeof(double),data.nangletypes,fp);

  } else if (strcmp(data.angle_style,"harmonic") == 0) {

    data.angle_harmonic_k = new double[data.nangletypes+1];
    data.angle_harmonic_theta0 = new double[data.nangletypes+1];
    fread(&data.angle_harmonic_k[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_harmonic_theta0[1],sizeof(double),data.nangletypes,fp);

  } else if (strcmp(data.angle_style,"cg/cmm") == 0) {

    data.angle_harmonic_k = new double[data.nangletypes+1];
    data.angle_harmonic_theta0 = new double[data.nangletypes+1];
    data.angle_cg_cmm_epsilon = new double[data.nangletypes+1];
    data.angle_cg_cmm_sigma = new double[data.nangletypes+1];
    double *angle_cg_cmm_rcut = new double[data.nangletypes+1];
    data.angle_cg_cmm_type = new int[data.nangletypes+1];
    
    fread(&data.angle_harmonic_k[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_harmonic_theta0[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_cg_cmm_epsilon[1],sizeof(double),data.nangletypes,fp);
    fread(&data.angle_cg_cmm_sigma[1],sizeof(double),data.nangletypes,fp);
    fread(angle_cg_cmm_rcut,sizeof(double),data.nangletypes,fp);
    fread(&data.angle_cg_cmm_type[1],sizeof(int),data.nangletypes,fp);

  } else if (strcmp(data.angle_style,"hybrid") == 0) {

    int nstyles = read_int(fp);
    for (int i = 0; i < nstyles; i++)
      char *substyle = read_char(fp);

  } else {
    printf("ERROR: Unknown angle style %s\n",data.angle_style);
    exit(1);
  }
}

// ---------------------------------------------------------------------
// dihedral coeffs
// one section for each dihedral style
// ---------------------------------------------------------------------

void dihedral(FILE *fp, Data &data)
{
  if (strcmp(data.dihedral_style,"none") == 0) {

  } else if (strcmp(data.dihedral_style,"charmm") == 0) {

    data.dihedral_charmm_k = new double[data.ndihedraltypes+1];
    data.dihedral_charmm_multiplicity = new int[data.ndihedraltypes+1];
    data.dihedral_charmm_sign = new int[data.ndihedraltypes+1];
    data.dihedral_charmm_weight = new double[data.ndihedraltypes+1];
    fread(&data.dihedral_charmm_k[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_charmm_multiplicity[1],sizeof(int),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_charmm_sign[1],sizeof(int),data.ndihedraltypes,fp);
    fread(&data.dihedral_charmm_weight[1],sizeof(double),
	  data.ndihedraltypes,fp);

  } else if (strcmp(data.dihedral_style,"class2") == 0) {

    data.dihedral_class2_k1 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_k2 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_k3 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_phi1 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_phi2 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_phi3 = new double[data.ndihedraltypes+1];

    data.dihedral_class2_mbt_f1 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_mbt_f2 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_mbt_f3 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_mbt_r0 = new double[data.ndihedraltypes+1];

    data.dihedral_class2_ebt_f1_1 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_ebt_f2_1 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_ebt_f3_1 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_ebt_r0_1 = new double[data.ndihedraltypes+1];

    data.dihedral_class2_ebt_f1_2 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_ebt_f2_2 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_ebt_f3_2 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_ebt_r0_2 = new double[data.ndihedraltypes+1];

    data.dihedral_class2_at_f1_1 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_at_f2_1 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_at_f3_1 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_at_theta0_1 = new double[data.ndihedraltypes+1];

    data.dihedral_class2_at_f1_2 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_at_f2_2 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_at_f3_2 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_at_theta0_2 = new double[data.ndihedraltypes+1];

    data.dihedral_class2_aat_k = new double[data.ndihedraltypes+1];
    data.dihedral_class2_aat_theta0_1 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_aat_theta0_2 = new double[data.ndihedraltypes+1];

    data.dihedral_class2_bb13_k = new double[data.ndihedraltypes+1];
    data.dihedral_class2_bb13_r10 = new double[data.ndihedraltypes+1];
    data.dihedral_class2_bb13_r30 = new double[data.ndihedraltypes+1];

    fread(&data.dihedral_class2_k1[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_k2[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_k3[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_phi1[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_phi2[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_phi3[1],sizeof(double),data.ndihedraltypes,fp);

    fread(&data.dihedral_class2_mbt_f1[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_mbt_f2[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_mbt_f3[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_mbt_r0[1],sizeof(double),
	  data.ndihedraltypes,fp);

    fread(&data.dihedral_class2_ebt_f1_1[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_ebt_f2_1[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_ebt_f3_1[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_ebt_r0_1[1],sizeof(double),
	  data.ndihedraltypes,fp);

    fread(&data.dihedral_class2_ebt_f1_2[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_ebt_f2_2[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_ebt_f3_2[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_ebt_r0_2[1],sizeof(double),
	  data.ndihedraltypes,fp);

    fread(&data.dihedral_class2_at_f1_1[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_at_f2_1[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_at_f3_1[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_at_theta0_1[1],sizeof(double),
	  data.ndihedraltypes,fp);

    fread(&data.dihedral_class2_at_f1_2[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_at_f2_2[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_at_f3_2[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_at_theta0_2[1],sizeof(double),
	  data.ndihedraltypes,fp);

    fread(&data.dihedral_class2_aat_k[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_aat_theta0_1[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_aat_theta0_2[1],sizeof(double),
	  data.ndihedraltypes,fp);

    fread(&data.dihedral_class2_bb13_k[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_bb13_r10[1],sizeof(double),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_class2_bb13_r30[1],sizeof(double),
	  data.ndihedraltypes,fp);

  } else if (strcmp(data.dihedral_style,"harmonic") == 0) {

    data.dihedral_harmonic_k = new double[data.ndihedraltypes+1];
    data.dihedral_harmonic_multiplicity = new int[data.ndihedraltypes+1];
    data.dihedral_harmonic_sign = new int[data.ndihedraltypes+1];
    fread(&data.dihedral_harmonic_k[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_harmonic_multiplicity[1],sizeof(int),
	  data.ndihedraltypes,fp);
    fread(&data.dihedral_harmonic_sign[1],sizeof(int),data.ndihedraltypes,fp);

  } else if (strcmp(data.dihedral_style,"helix") == 0) {

    data.dihedral_helix_aphi = new double[data.ndihedraltypes+1];
    data.dihedral_helix_bphi = new double[data.ndihedraltypes+1];
    data.dihedral_helix_cphi = new double[data.ndihedraltypes+1];
    fread(&data.dihedral_helix_aphi[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_helix_bphi[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_helix_cphi[1],sizeof(double),data.ndihedraltypes,fp);

  } else if (strcmp(data.dihedral_style,"multi/harmonic") == 0) {

    data.dihedral_multi_a1 = new double[data.ndihedraltypes+1];
    data.dihedral_multi_a2 = new double[data.ndihedraltypes+1];
    data.dihedral_multi_a3 = new double[data.ndihedraltypes+1];
    data.dihedral_multi_a4 = new double[data.ndihedraltypes+1];
    data.dihedral_multi_a5 = new double[data.ndihedraltypes+1];
    fread(&data.dihedral_multi_a1[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_multi_a2[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_multi_a3[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_multi_a4[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_multi_a5[1],sizeof(double),data.ndihedraltypes,fp);

  } else if (strcmp(data.dihedral_style,"opls") == 0) {

    data.dihedral_opls_k1 = new double[data.ndihedraltypes+1];
    data.dihedral_opls_k2 = new double[data.ndihedraltypes+1];
    data.dihedral_opls_k3 = new double[data.ndihedraltypes+1];
    data.dihedral_opls_k4 = new double[data.ndihedraltypes+1];
    fread(&data.dihedral_opls_k1[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_opls_k2[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_opls_k3[1],sizeof(double),data.ndihedraltypes,fp);
    fread(&data.dihedral_opls_k4[1],sizeof(double),data.ndihedraltypes,fp);

  } else if (strcmp(data.dihedral_style,"hybrid") == 0) {

    int nstyles = read_int(fp);
    for (int i = 0; i < nstyles; i++)
      char *substyle = read_char(fp);

  } else {
    printf("ERROR: Unknown dihedral style %s\n",data.dihedral_style);
    exit(1);
  }
}

// ---------------------------------------------------------------------
// improper coeffs
// one section for each improper style
// ---------------------------------------------------------------------

void improper(FILE *fp, Data &data)
{
  if (strcmp(data.improper_style,"none") == 0) {

  } else if (strcmp(data.improper_style,"class2") == 0) {

    data.improper_class2_k0 = new double[data.nimpropertypes+1];
    data.improper_class2_chi0 = new double[data.nimpropertypes+1];

    data.improper_class2_aa_k1 = new double[data.nimpropertypes+1];
    data.improper_class2_aa_k2 = new double[data.nimpropertypes+1];
    data.improper_class2_aa_k3 = new double[data.nimpropertypes+1];
    data.improper_class2_aa_theta0_1 = new double[data.nimpropertypes+1];
    data.improper_class2_aa_theta0_2 = new double[data.nimpropertypes+1];
    data.improper_class2_aa_theta0_3 = new double[data.nimpropertypes+1];

    fread(&data.improper_class2_k0[1],sizeof(double),
	  data.nimpropertypes,fp);
    fread(&data.improper_class2_chi0[1],sizeof(double),
	  data.nimpropertypes,fp);

    fread(&data.improper_class2_aa_k1[1],sizeof(double),
	  data.nimpropertypes,fp);
    fread(&data.improper_class2_aa_k2[1],sizeof(double),
	  data.nimpropertypes,fp);
    fread(&data.improper_class2_aa_k3[1],sizeof(double),
	  data.nimpropertypes,fp);
    fread(&data.improper_class2_aa_theta0_1[1],sizeof(double),
	  data.nimpropertypes,fp);
    fread(&data.improper_class2_aa_theta0_2[1],sizeof(double),
	  data.nimpropertypes,fp);
    fread(&data.improper_class2_aa_theta0_3[1],sizeof(double),
	  data.nimpropertypes,fp);

  } else if (strcmp(data.improper_style,"cvff") == 0) {

    data.improper_cvff_k = new double[data.nimpropertypes+1];
    data.improper_cvff_sign = new int[data.nimpropertypes+1];
    data.improper_cvff_multiplicity = new int[data.nimpropertypes+1];
    fread(&data.improper_cvff_k[1],sizeof(double),data.nimpropertypes,fp);
    fread(&data.improper_cvff_sign[1],sizeof(int),data.nimpropertypes,fp);
    fread(&data.improper_cvff_multiplicity[1],sizeof(int),
	  data.nimpropertypes,fp);

  } else if (strcmp(data.improper_style,"harmonic") == 0) {

    data.improper_harmonic_k = new double[data.nimpropertypes+1];
    data.improper_harmonic_chi = new double[data.nimpropertypes+1];
    fread(&data.improper_harmonic_k[1],sizeof(double),data.nimpropertypes,fp);
    fread(&data.improper_harmonic_chi[1],sizeof(double),
	  data.nimpropertypes,fp);

  } else if (strcmp(data.improper_style,"hybrid") == 0) {

    int nstyles = read_int(fp);
    for (int i = 0; i < nstyles; i++)
      char *substyle = read_char(fp);

  } else {
    printf("ERROR: Unknown improper style %s\n",data.improper_style);
    exit(1);
  }
}

// ---------------------------------------------------------------------
// initialize Data
// ---------------------------------------------------------------------

Data::Data() {}

// ---------------------------------------------------------------------
// print out stats on problem
// ---------------------------------------------------------------------

void Data::stats()
{
  printf("  Restart file version = %s\n",version);
  printf("  Ntimestep = %d\n",ntimestep);
  printf("  Nprocs = %d\n",nprocs);
  printf("  Natoms = %d\n",natoms);
  printf("  Nbonds = %d\n",nbonds);
  printf("  Nangles = %d\n",nangles);
  printf("  Ndihedrals = %d\n",ndihedrals);
  printf("  Nimpropers = %d\n",nimpropers);
  printf("  Unit style = %s\n",unit_style);
  printf("  Atom style = %s\n",atom_style);
  printf("  Pair style = %s\n",pair_style);
  printf("  Bond style = %s\n",bond_style);
  printf("  Angle style = %s\n",angle_style);
  printf("  Dihedral style = %s\n",dihedral_style);
  printf("  Improper style = %s\n",improper_style);
  printf("  Xlo xhi = %g %g\n",xlo,xhi);
  printf("  Ylo yhi = %g %g\n",ylo,yhi);
  printf("  Zlo zhi = %g %g\n",zlo,zhi);
  if (triclinic) printf("  Xy xz yz = %g %g %g\n",xy,xz,yz);
  printf("  Periodicity = %d %d %d\n",xperiodic,yperiodic,zperiodic);
  printf("  Boundary = %d %d, %d %d, %d %d\n",boundary[0][0],boundary[0][1],
	 boundary[1][0],boundary[1][1],boundary[2][0],boundary[2][1]);
}

// ---------------------------------------------------------------------
// write the data file
// ---------------------------------------------------------------------

void Data::write(FILE *fp, FILE *fp2)
{
  fprintf(fp,"LAMMPS data file from restart file: timestep = %d, procs = %d\n",
	  ntimestep,nprocs);
  
  fprintf(fp,"\n");

  fprintf(fp,"%d atoms\n",natoms);
  if (nbonds) fprintf(fp,"%d bonds\n",nbonds);
  if (nangles) fprintf(fp,"%d angles\n",nangles);
  if (ndihedrals) fprintf(fp,"%d dihedrals\n",ndihedrals);
  if (nimpropers) fprintf(fp,"%d impropers\n",nimpropers);

  fprintf(fp,"\n");

  fprintf(fp,"%d atom types\n",ntypes);
  if (nbondtypes) fprintf(fp,"%d bond types\n",nbondtypes);
  if (nangletypes) fprintf(fp,"%d angle types\n",nangletypes);
  if (ndihedraltypes) fprintf(fp,"%d dihedral types\n",ndihedraltypes);
  if (nimpropertypes) fprintf(fp,"%d improper types\n",nimpropertypes);

  fprintf(fp,"\n");
  
  fprintf(fp,"%-1.16e %-1.16e xlo xhi\n",xlo,xhi);
  fprintf(fp,"%-1.16e %-1.16e ylo yhi\n",ylo,yhi);
  fprintf(fp,"%-1.16e %-1.16e zlo zhi\n",zlo,zhi);
  if (triclinic) fprintf(fp,"%-1.16e %-1.16e %-1.16e xy xz yz\n",xy,xz,yz);

  // write ff styles to input file

  if (fp2) {
    fprintf(fp2,"# LAMMPS input file from restart file: timestep = %d, procs = %d\n\n",
	  ntimestep,nprocs);
    if (pair_style) fprintf(fp2,"pair_style %s\n",pair_style);
    if (bond_style) fprintf(fp2,"bond_style %s\n",bond_style);
    if (angle_style) fprintf(fp2,"angle_style %s\n",angle_style);
    if (dihedral_style) fprintf(fp2,"dihedral_style %s\n",dihedral_style);
    if (improper_style) fprintf(fp2,"improper_style %s\n",improper_style);
    fprintf(fp2,"special_bonds %g %g %g %g %g %g\n",
            special_lj[1],special_lj[2],special_lj[3],
            special_lj[1],special_coul[2],special_coul[3]);
    fprintf(fp2,"\n");
  }

  // mass to either data file or input file

  if (mass) {
    if (fp2) {
      fprintf(fp2,"\n");
      for (int i = 1; i <= ntypes; i++) fprintf(fp2,"mass %d %g\n",i,mass[i]);
      fprintf(fp2,"\n");
    } else {
      fprintf(fp,"\nMasses\n\n");
      for (int i = 1; i <= ntypes; i++) fprintf(fp,"%d %g\n",i,mass[i]);
    }
  }

  // shape and dipole to data file
  // convert shape from radius to diameter

  if (shape) {
    fprintf(fp,"\nShapes\n\n");
    for (int i = 1; i <= ntypes; i++)
      fprintf(fp,"%d %g %g %g\n",i,
	      2.0*shape[3*i+0],2.0*shape[3*i+1],2.0*shape[3*i+2]);
  }

  if (dipole) {
    fprintf(fp,"\nDipoles\n\n");
    for (int i = 1; i <= ntypes; i++) fprintf(fp,"%d %g\n",i,dipole[i]);
  }

  // pair coeffs to data file

  if (pair_style && fp2 == NULL) {
    if ((strcmp(pair_style,"none") != 0) && 
	(strcmp(pair_style,"airebo") != 0) &&
	(strcmp(pair_style,"coul/cut") != 0) &&
	(strcmp(pair_style,"coul/debye") != 0) &&
        (strcmp(pair_style,"eff/cut") != 0) &&
	(strcmp(pair_style,"coul/long") != 0) &&
	(strcmp(pair_style,"eam") != 0) &&
	(strcmp(pair_style,"eam/opt") != 0) &&
	(strcmp(pair_style,"eam/alloy") != 0) &&
	(strcmp(pair_style,"eam/alloy/opt") != 0) &&
	(strcmp(pair_style,"eam/fs") != 0) &&
	(strcmp(pair_style,"eam/fs/opt") != 0) &&
	(strcmp(pair_style,"gran/history") != 0) &&
	(strcmp(pair_style,"gran/no_history") != 0) &&
	(strcmp(pair_style,"gran/hertzian") != 0) &&
	(strcmp(pair_style,"meam") != 0) &&
	(strcmp(pair_style,"reax") != 0) &&
	(strcmp(pair_style,"sw") != 0) &&
	(strcmp(pair_style,"table") != 0) &&
	(strcmp(pair_style,"tersoff") != 0) &&
	(strcmp(pair_style,"hybrid") != 0) &&
	(strcmp(pair_style,"hybrid/overlay") != 0))
      fprintf(fp,"\nPair Coeffs\n\n");
    
    if (strcmp(pair_style,"born/coul/long") == 0) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g %g %g %g\n",i,
		pair_born_A[i],pair_born_rho[i],pair_born_sigma[i],
		pair_born_C[i],pair_born_D[i]);

    } else if ((strcmp(pair_style,"buck") == 0) || 
	(strcmp(pair_style,"buck/coul/cut") == 0) ||
	(strcmp(pair_style,"buck/coul/long") == 0) ||
	(strcmp(pair_style,"buck/long") == 0)) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g %g\n",i,
		pair_buck_A[i],pair_buck_rho[i],pair_buck_C[i]);
      
    } else if (strcmp(pair_style,"colloid") == 0) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g %g %g\n",i,
		pair_colloid_A12[i],pair_colloid_sigma[i],
		pair_colloid_d2[i],pair_colloid_d2[i]);

    } else if (strcmp(pair_style,"dipole/cut") == 0) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		pair_dipole_epsilon[i],pair_dipole_sigma[i]);
      
    } else if (strcmp(pair_style,"dpd") == 0) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		pair_dpd_a0[i],pair_dpd_gamma[i]);
      
    } else if (strcmp(pair_style,"gayberne") == 0) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g %g %g %g %g %g %g\n",i,
		pair_gb_epsilon[i],pair_gb_sigma[i],
		pair_gb_epsa[i],pair_gb_epsb[i],pair_gb_epsc[i],
		pair_gb_epsa[i],pair_gb_epsb[i],pair_gb_epsc[i]);

    } else if ((strcmp(pair_style,"lj/charmm/coul/charmm") == 0) ||
	       (strcmp(pair_style,"lj/charmm/coul/charmm/implicit") == 0) ||
	       (strcmp(pair_style,"lj/charmm/coul/long") == 0) ||
	       (strcmp(pair_style,"lj/charmm/coul/long") == 0)) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g %g %g\n",i,
		pair_charmm_epsilon[i],pair_charmm_sigma[i],
		pair_charmm_eps14[i],pair_charmm_sigma14[i]);
      
    } else if ((strcmp(pair_style,"lj/class2") == 0) || 
	       (strcmp(pair_style,"lj/class2/coul/cut") == 0) ||
	       (strcmp(pair_style,"lj/class2/coul/long") == 0)) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		pair_class2_epsilon[i],pair_class2_sigma[i]);
      
    } else if ((strcmp(pair_style,"lj/cut") == 0) || 
	       (strcmp(pair_style,"lj/cut/opt") == 0) ||
	       (strcmp(pair_style,"lj/cut/coul/cut") == 0) ||
	       (strcmp(pair_style,"lj/cut/coul/debye") == 0) ||
	       (strcmp(pair_style,"lj/cut/coul/long") == 0) ||
	       (strcmp(pair_style,"lj/cut/coul/long/tip4p") == 0) |
	       (strcmp(pair_style,"lj/coul") == 0)) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		pair_lj_epsilon[i],pair_lj_sigma[i]);

    } else if (strcmp(pair_style,"lj/expand") == 0) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g %g\n",i,
		pair_ljexpand_epsilon[i],pair_ljexpand_sigma[i],
		pair_ljexpand_shift[i]);

    } else if ((strcmp(pair_style,"lj/gromacs") == 0) ||
	       (strcmp(pair_style,"lj/gromacs/coul/gromacs") == 0)) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		pair_ljgromacs_epsilon[i],pair_ljgromacs_sigma[i]);

    } else if (strcmp(pair_style,"lj/smooth") == 0) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		pair_ljsmooth_epsilon[i],pair_ljsmooth_sigma[i]);
      
    } else if ((strcmp(pair_style,"morse") == 0) ||
	       (strcmp(pair_style,"morse/opt") == 0)) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g %g\n",i,
		pair_morse_d0[i],pair_morse_alpha[i],pair_morse_r0[i]);
      
    } else if (strcmp(pair_style,"soft") == 0) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		pair_soft_start[i],pair_soft_stop[i]);
      
    } else if (strcmp(pair_style,"yukawa") == 0) {
      for (int i = 1; i <= ntypes; i++)
	fprintf(fp,"%d %g\n",i,
		pair_yukawa_A[i]);

    } else if ((strcmp(pair_style,"cg/cmm") == 0) || 
               (strcmp(pair_style,"cg/cmm/coul/cut") == 0) ||
               (strcmp(pair_style,"cg/cmm/coul/long") == 0)) {
      printf("ERROR: Cannot write pair_style %s to data file\n",
	     pair_style);
      exit(1);
    }
  }

  // pair coeffs to input file
  // only supported styles = cg/cmm

  if (pair_style && fp2) {
    if ((strcmp(pair_style,"cg/cmm") == 0) || 
	(strcmp(pair_style,"cg/cmm/coul/cut") == 0) ||
	(strcmp(pair_style,"cg/cmm/coul/long") == 0)) {
      for (int i = 1; i <= ntypes; i++) {
	for (int j = i; j <= ntypes; j++) {
	  fprintf(fp2,"pair_coeff %d %d %s %g %g\n",i,j,
		  cg_type_list[pair_cg_cmm_type[i][j]],
		  pair_cg_epsilon[i][j],pair_cg_sigma[i][j]);
	}
      }

    } else {
      printf("ERROR: Cannot write pair_style %s to input file\n",
	     pair_style);
      exit(1);
    }
  }

  // bond coeffs to data file

  if (bond_style && fp2 == NULL) {
    if ((strcmp(bond_style,"none") != 0) && 
	(strcmp(bond_style,"hybrid") != 0))
      fprintf(fp,"\nBond Coeffs\n\n");

    if (strcmp(bond_style,"class2") == 0) {
      for (int i = 1; i <= nbondtypes; i++)
	fprintf(fp,"%d %g %g %g %g\n",i,
		bond_class2_r0[i],bond_class2_k2[i],
		bond_class2_k3[i],bond_class2_k4[i]);

    } else if (strcmp(bond_style,"fene") == 0) {
      for (int i = 1; i <= nbondtypes; i++)
	fprintf(fp,"%d %g %g %g %g\n",i,
		bond_fene_k[i],bond_fene_r0[i],
		bond_fene_epsilon[i],bond_fene_sigma[i]);

    } else if (strcmp(bond_style,"fene/expand") == 0) {
      for (int i = 1; i <= nbondtypes; i++)
	fprintf(fp,"%d %g %g %g %g %g\n",i,
		bond_feneexpand_k[i],bond_feneexpand_r0[i],
		bond_feneexpand_epsilon[i],bond_feneexpand_sigma[i],
		bond_feneexpand_shift[i]);

    } else if (strcmp(bond_style,"harmonic") == 0) {
      for (int i = 1; i <= nbondtypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		bond_harmonic_k[i],bond_harmonic_r0[i]);

    } else if (strcmp(bond_style,"morse") == 0) {
      for (int i = 1; i <= nbondtypes; i++)
	fprintf(fp,"%d %g %g %g\n",i,
		bond_morse_d0[i],bond_morse_alpha[i],bond_morse_r0[i]);

    } else if (strcmp(bond_style,"nonlinear") == 0) {
      for (int i = 1; i <= nbondtypes; i++)
	fprintf(fp,"%d %g %g %g\n",i,
		bond_nonlinear_epsilon[i],bond_nonlinear_r0[i],
		bond_nonlinear_lamda[i]);

    } else if (strcmp(bond_style,"quartic") == 0) {
      for (int i = 1; i <= nbondtypes; i++)
        fprintf(fp,"%d %g %g %g %g %g\n",i,
                bond_quartic_k[i],bond_quartic_b1[i],bond_quartic_b2[i],
                bond_quartic_rc[i],bond_quartic_u0[i]);
    }
  }

  // bond coeffs to input file
  // only supported styles = harmonic

  if (bond_style && fp2) {
    if (strcmp(bond_style,"harmonic") == 0) {
      for (int i = 1; i <= nbondtypes; i++)
	fprintf(fp2,"bond_coeff  %d %g %g\n",i,
		bond_harmonic_k[i],bond_harmonic_r0[i]);

    } else {
      printf("ERROR: Cannot write bond_style %s to input file\n",
	     bond_style);
      exit(1);
    }
  }

  // angle coeffs to data file

  if (angle_style && fp2 == NULL) {
    double PI = 3.1415926;           // convert back to degrees

    if ((strcmp(angle_style,"none") != 0) && 
	(strcmp(angle_style,"hybrid") != 0))
      fprintf(fp,"\nAngle Coeffs\n\n");
    
    if (strcmp(angle_style,"charmm") == 0) {
      for (int i = 1; i <= nangletypes; i++)
	fprintf(fp,"%d %g %g %g %g\n",i,
		angle_charmm_k[i],angle_charmm_theta0[i]/PI*180.0,
		angle_charmm_k_ub[i],angle_charmm_r_ub[i]);

    } else if (strcmp(angle_style,"class2") == 0) {
      for (int i = 1; i <= nangletypes; i++)
	fprintf(fp,"%d %g %g %g %g\n",i,
		angle_class2_theta0[i]/PI*180.0,angle_class2_k2[i], 
		angle_class2_k3[i],angle_class2_k4[i]);
      
      fprintf(fp,"\nBondBond Coeffs\n\n");
      for (int i = 1; i <= nangletypes; i++)
	fprintf(fp,"%d %g %g %g\n",i,
		angle_class2_bb_k[i],
		angle_class2_bb_r1[i],angle_class2_bb_r2[i]);
      
      fprintf(fp,"\nBondAngle Coeffs\n\n");
      for (int i = 1; i <= nangletypes; i++)
	fprintf(fp,"%d %g %g %g %g\n",i,
		angle_class2_ba_k1[i],angle_class2_ba_k2[i],
		angle_class2_ba_r1[i],angle_class2_ba_r2[i]);

    } else if (strcmp(angle_style,"cosine") == 0) {
      for (int i = 1; i <= nangletypes; i++)
	fprintf(fp,"%d %g\n",i,angle_cosine_k[i]);
      
    } else if ((strcmp(angle_style,"cosine/squared") == 0) ||
               (strcmp(angle_style,"cosine/delta") == 0)) {
      for (int i = 1; i <= nangletypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		angle_cosine_squared_k[i],
		angle_cosine_squared_theta0[i]/PI*180.0);
      
    } else if (strcmp(angle_style,"harmonic") == 0) {
      for (int i = 1; i <= nangletypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		angle_harmonic_k[i],angle_harmonic_theta0[i]/PI*180.0);
      
    } else if (strcmp(angle_style,"cg/cmm") == 0) {
      for (int i = 1; i <= nangletypes; i++)
	fprintf(fp,"%d %g %g %s %g %g\n",i,
		angle_harmonic_k[i],angle_harmonic_theta0[i]/PI*180.0,
		cg_type_list[angle_cg_cmm_type[i]],angle_cg_cmm_epsilon[i],
		angle_cg_cmm_sigma[i]);
    }
  }

  // angle coeffs to input file
  // only supported styles = cosine/squared, harmonic, cg/cmm

  if (angle_style && fp2) {
    double PI = 3.1415926;           // convert back to degrees

    if ((strcmp(angle_style,"cosine/squared") == 0) ||
        (strcmp(angle_style,"cosine/delta") == 0)) {
      for (int i = 1; i <= nangletypes; i++)
	fprintf(fp2,"angle_coeffs  %d %g %g\n",i,
		angle_cosine_squared_k[i],
		angle_cosine_squared_theta0[i]/PI*180.0);
      
    } else if (strcmp(angle_style,"harmonic") == 0) {
      for (int i = 1; i <= nangletypes; i++)
	fprintf(fp2,"angle_coeffs  %d %g %g\n",i,
		angle_harmonic_k[i],angle_harmonic_theta0[i]/PI*180.0);
      
    } else if (strcmp(angle_style,"cg/cmm") == 0) {
      for (int i = 1; i <= nangletypes; i++)
	fprintf(fp2,"angle_coeffs  %d %g %g %s %g %g\n",i,
		angle_harmonic_k[i],angle_harmonic_theta0[i]/PI*180.0,
		cg_type_list[angle_cg_cmm_type[i]],angle_cg_cmm_epsilon[i],
		angle_cg_cmm_sigma[i]);
      
    } else {
      printf("ERROR: Cannot write angle_style %s to input file\n",
	     angle_style);
      exit(1);
    }
  }

  if (dihedral_style) {
    double PI = 3.1415926;           // convert back to degrees

    if ((strcmp(dihedral_style,"none") != 0) && 
	(strcmp(dihedral_style,"hybrid") != 0))
      fprintf(fp,"\nDihedral Coeffs\n\n");

    if (strcmp(dihedral_style,"charmm") == 0) {
      for (int i = 1; i <= ndihedraltypes; i++)
	fprintf(fp,"%d %g %d %d %g\n",i,
		dihedral_charmm_k[i],dihedral_charmm_multiplicity[i],
		dihedral_charmm_sign[i],dihedral_charmm_weight[i]);

    } else if (strcmp(dihedral_style,"class2") == 0) {
      for (int i = 1; i <= ndihedraltypes; i++)
	fprintf(fp,"%d %g %g %g %g %g %g\n",i,
		dihedral_class2_k1[i],
		dihedral_class2_phi1[i]/PI*180.0,
		dihedral_class2_k2[i],
		dihedral_class2_phi2[i]/PI*180.0,
		dihedral_class2_k3[i],
		dihedral_class2_phi3[i]/PI*180.0);

      fprintf(fp,"\nMiddleBondTorsion Coeffs\n\n");
      for (int i = 1; i <= ndihedraltypes; i++)
	fprintf(fp,"%d %g %g %g %g\n",i,
		dihedral_class2_mbt_f1[i],dihedral_class2_mbt_f2[i],
		dihedral_class2_mbt_f3[i],dihedral_class2_mbt_r0[i]);

      fprintf(fp,"\nEndBondTorsion Coeffs\n\n");
      for (int i = 1; i <= ndihedraltypes; i++)
	fprintf(fp,"%d %g %g %g %g %g %g %g %g\n",i,
		dihedral_class2_ebt_f1_1[i],dihedral_class2_ebt_f2_1[i],
		dihedral_class2_ebt_f3_1[i],
		dihedral_class2_ebt_f1_2[i],dihedral_class2_ebt_f2_2[i],
		dihedral_class2_ebt_f3_2[i],
		dihedral_class2_ebt_r0_1[i],
		dihedral_class2_ebt_r0_2[i]);

      fprintf(fp,"\nAngleTorsion Coeffs\n\n");
      for (int i = 1; i <= ndihedraltypes; i++)
	fprintf(fp,"%d %g %g %g %g %g %g %g %g\n",i,
		dihedral_class2_at_f1_1[i],dihedral_class2_at_f2_1[i],
		dihedral_class2_at_f3_1[i],
		dihedral_class2_at_f1_2[i],dihedral_class2_at_f2_2[i],
		dihedral_class2_at_f3_2[i],
		dihedral_class2_at_theta0_1[i]/PI*180.0,
		dihedral_class2_at_theta0_2[i]/PI*180.0);

      fprintf(fp,"\nAngleAngleTorsion Coeffs\n\n");
      for (int i = 1; i <= ndihedraltypes; i++)
	fprintf(fp,"%d %g %g %g\n",i,
		dihedral_class2_aat_k[i],
		dihedral_class2_aat_theta0_1[i]/PI*180.0,
		dihedral_class2_aat_theta0_2[i]/PI*180.0);

      fprintf(fp,"\nBondBond13 Coeffs\n\n");
      for (int i = 1; i <= ndihedraltypes; i++)
	fprintf(fp,"%d %g %g %g\n",i,
		dihedral_class2_bb13_k[i],
		dihedral_class2_bb13_r10[i],dihedral_class2_bb13_r30[i]);

    } else if (strcmp(dihedral_style,"harmonic") == 0) {
      for (int i = 1; i <= ndihedraltypes; i++)
	fprintf(fp,"%d %g %d %d\n",i,
		dihedral_harmonic_k[i],dihedral_harmonic_multiplicity[i],
		dihedral_harmonic_sign[i]);

    } else if (strcmp(dihedral_style,"helix") == 0) {
      for (int i = 1; i <= ndihedraltypes; i++) 
	fprintf(fp,"%d %g %g %g\n",i,dihedral_helix_aphi[i],
		dihedral_helix_bphi[i],dihedral_helix_cphi[i]);

    } else if (strcmp(dihedral_style,"multi/harmonic") == 0) {
      for (int i = 1; i <= ndihedraltypes; i++)
	fprintf(fp,"%d %g %g %g %g %g\n",i,
		dihedral_multi_a1[i],dihedral_multi_a2[i],
		dihedral_multi_a3[i],dihedral_multi_a4[i],
		dihedral_multi_a5[i]);

    } else if (strcmp(dihedral_style,"opls") == 0) {
      for (int i = 1; i <= ndihedraltypes; i++)        // restore factor of 2
	fprintf(fp,"%d %g %g %g %g\n",i,
		2.0*dihedral_opls_k1[i],2.0*dihedral_opls_k2[i],
		2.0*dihedral_opls_k3[i],2.0*dihedral_opls_k4[i]);
    }
  }

  if (improper_style) {
    double PI = 3.1415926;           // convert back to degrees

    if ((strcmp(improper_style,"none") != 0) && 
	(strcmp(improper_style,"hybrid") != 0))
      fprintf(fp,"\nImproper Coeffs\n\n");

    if (strcmp(improper_style,"class2") == 0) {
      for (int i = 1; i <= nimpropertypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		improper_class2_k0[i],improper_class2_chi0[i]/PI*180.0);

      fprintf(fp,"\nAngleAngle Coeffs\n\n");
      for (int i = 1; i <= nimpropertypes; i++)
	fprintf(fp,"%d %g %g %g %g %g %g\n",i,
		improper_class2_aa_k1[i],improper_class2_aa_k2[i],
		improper_class2_aa_k3[i],
		improper_class2_aa_theta0_1[i]/PI*180.0,
		improper_class2_aa_theta0_2[i]/PI*180.0,
		improper_class2_aa_theta0_3[i]/PI*180.0);

    } else if (strcmp(improper_style,"cvff") == 0) {
      for (int i = 1; i <= nimpropertypes; i++)
	fprintf(fp,"%d %g %d %d\n",i,
		improper_cvff_k[i],improper_cvff_sign[i],
		improper_cvff_multiplicity[i]);

    } else if (strcmp(improper_style,"harmonic") == 0) {
      for (int i = 1; i <= nimpropertypes; i++)
	fprintf(fp,"%d %g %g\n",i,
		improper_harmonic_k[i],improper_harmonic_chi[i]/PI*180.0);
    }
  }

  if (natoms) {
    fprintf(fp,"\nAtoms\n\n");

    int ix,iy,iz;
    for (int i = 0; i < natoms; i++) {

      ix = (image[i] & 1023) - 512;
      iy = (image[i] >> 10 & 1023) - 512;
      iz = (image[i] >> 20) - 512;

      if (style_hybrid == 0) {
	if (style_angle) write_atom_angle(fp,i,ix,iy,iz);
	if (style_atomic) write_atom_atomic(fp,i,ix,iy,iz);
	if (style_bond) write_atom_bond(fp,i,ix,iy,iz);
	if (style_charge) write_atom_charge(fp,i,ix,iy,iz);
	if (style_dipole) write_atom_dipole(fp,i,ix,iy,iz);
	if (style_dpd) write_atom_dpd(fp,i,ix,iy,iz);
	if (style_ellipsoid) write_atom_ellipsoid(fp,i,ix,iy,iz);
	if (style_full) write_atom_full(fp,i,ix,iy,iz);
	if (style_granular) write_atom_granular(fp,i,ix,iy,iz);
	if (style_molecular) write_atom_molecular(fp,i,ix,iy,iz);
	if (style_peri) write_atom_peri(fp,i,ix,iy,iz);
        if (style_electron) write_atom_electron(fp,i,ix,iy,iz);
	fprintf(fp,"\n");

      } else {
	fprintf(fp,"%d %d %-1.16e %-1.16e %-1.16e",
		tag[i],type[i],x[i],y[i],z[i]);
	for (int k = 1; k <= style_hybrid; k++) {
	  if (k == style_angle) write_atom_angle_extra(fp,i);
	  if (k == style_atomic) write_atom_atomic_extra(fp,i);
	  if (k == style_bond) write_atom_bond_extra(fp,i);
	  if (k == style_charge) write_atom_charge_extra(fp,i);
	  if (k == style_dipole) write_atom_dipole_extra(fp,i);
	  if (k == style_dpd) write_atom_dpd_extra(fp,i);
	  if (k == style_ellipsoid) write_atom_ellipsoid_extra(fp,i);
	  if (k == style_full) write_atom_full_extra(fp,i);
	  if (k == style_granular) write_atom_granular_extra(fp,i);
	  if (k == style_molecular) write_atom_molecular_extra(fp,i);
	  if (k == style_peri) write_atom_peri_extra(fp,i);
          if (k == style_electron) write_atom_electron_extra(fp,i);
	}
	fprintf(fp," %d %d %d\n",ix,iy,iz);
      }
    }
  }

  if (natoms) {
    fprintf(fp,"\nVelocities\n\n");
    for (int i = 0; i < natoms; i++)

      if (style_hybrid == 0) {
	if (style_angle) write_vel_angle(fp,i);
	if (style_atomic) write_vel_atomic(fp,i);
	if (style_bond) write_vel_bond(fp,i);
	if (style_charge) write_vel_charge(fp,i);
	if (style_dipole) write_vel_dipole(fp,i);
	if (style_dpd) write_vel_dpd(fp,i);
	if (style_ellipsoid) write_vel_ellipsoid(fp,i);
	if (style_full) write_vel_full(fp,i);
	if (style_granular) write_vel_granular(fp,i);
	if (style_molecular) write_vel_molecular(fp,i);
	if (style_peri) write_vel_peri(fp,i);
        if (style_electron) write_vel_electron(fp,i);
	fprintf(fp,"\n");

      } else {
	fprintf(fp,"%d %-1.16e %-1.16e %-1.16e",tag[i],vx[i],vy[i],vz[i]);
	for (int k = 1; k <= style_hybrid; k++) {
	  if (k == style_angle) write_vel_angle_extra(fp,i);
	  if (k == style_atomic) write_vel_atomic_extra(fp,i);
	  if (k == style_bond) write_vel_bond_extra(fp,i);
	  if (k == style_charge) write_vel_charge_extra(fp,i);
	  if (k == style_dipole) write_vel_dipole_extra(fp,i);
	  if (k == style_dpd) write_vel_dpd_extra(fp,i);
	  if (k == style_ellipsoid) write_vel_ellipsoid_extra(fp,i);
	  if (k == style_full) write_vel_full_extra(fp,i);
	  if (k == style_granular) write_vel_granular_extra(fp,i);
	  if (k == style_molecular) write_vel_molecular_extra(fp,i);
	  if (k == style_peri) write_vel_peri_extra(fp,i);
          if (k == style_electron) write_vel_electron_extra(fp,i);
	}
	fprintf(fp,"\n");
      }
  }

  if (nbonds) {
    fprintf(fp,"\nBonds\n\n");
    for (int i = 0; i < nbonds; i++)
      fprintf(fp,"%d %d %d %d\n",
	      i+1,bond_type[i],bond_atom1[i],bond_atom2[i]);
  }

  if (nangles) {
    fprintf(fp,"\nAngles\n\n");
    for (int i = 0; i < nangles; i++)
      fprintf(fp,"%d %d %d %d %d\n",
	      i+1,angle_type[i],angle_atom1[i],angle_atom2[i],angle_atom3[i]);
  }

  if (ndihedrals) {
    fprintf(fp,"\nDihedrals\n\n");
    for (int i = 0; i < ndihedrals; i++)
      fprintf(fp,"%d %d %d %d %d %d\n",
	      i+1,dihedral_type[i],dihedral_atom1[i],dihedral_atom2[i],
	      dihedral_atom3[i],dihedral_atom4[i]);
  }

  if (nimpropers) {
    fprintf(fp,"\nImpropers\n\n");
    for (int i = 0; i < nimpropers; i++)
      fprintf(fp,"%d %d %d %d %d %d\n",
	      i+1,improper_type[i],improper_atom1[i],improper_atom2[i],
	      improper_atom3[i],improper_atom4[i]);
  }
}

// ---------------------------------------------------------------------
// per-atom write routines
// one routine per atom style
// ---------------------------------------------------------------------

void Data::write_atom_angle(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %d %-1.16e %-1.16e %-1.16e %d %d %d",
	  tag[i],molecule[i],type[i],x[i],y[i],z[i],ix,iy,iz);
}

void Data::write_atom_atomic(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %-1.16e %-1.16e %-1.16e %d %d %d",
	  tag[i],type[i],x[i],y[i],z[i],ix,iy,iz);
}

void Data::write_atom_bond(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %d %-1.16e %-1.16e %-1.16e %d %d %d",
	  tag[i],molecule[i],type[i],x[i],y[i],z[i],ix,iy,iz);
}

void Data::write_atom_charge(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d",
	  tag[i],type[i],q[i],x[i],y[i],z[i],ix,iy,iz);
}

void Data::write_atom_dipole(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d",
	  tag[i],type[i],q[i],x[i],y[i],z[i],mux[i],muy[i],muz[i],ix,iy,iz);
}

void Data::write_atom_dpd(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %-1.16e %-1.16e %-1.16e %d %d %d",
	  tag[i],type[i],x[i],y[i],z[i],ix,iy,iz);
}

void Data::write_atom_ellipsoid(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d",
	  tag[i],type[i],x[i],y[i],z[i],
	  quatw[i],quati[i],quatj[i],quatk[i],ix,iy,iz);
}

void Data::write_atom_full(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %d %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d",
	  tag[i],molecule[i],type[i],q[i],x[i],y[i],z[i],ix,iy,iz);
}

void Data::write_atom_granular(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d",
	  tag[i],type[i],2.0*radius[i],density[i],x[i],y[i],z[i],ix,iy,iz);
}

void Data::write_atom_molecular(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %d %-1.16e %-1.16e %-1.16e %d %d %d",
	  tag[i],molecule[i],type[i],x[i],y[i],z[i],ix,iy,iz);
}

void Data::write_atom_peri(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d",
	  tag[i],type[i],vfrac[i],density[i],rmass[i],x[i],y[i],z[i],ix,iy,iz);
}

void Data::write_atom_electron(FILE *fp, int i, int ix, int iy, int iz)
{
  fprintf(fp,"%d %d %-1.16e %d %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d",
          tag[i],type[i],q[i],spin[i],radius[i],x[i],y[i],z[i],ix,iy,iz);
}

// ---------------------------------------------------------------------
// per-atom write routines of extra quantities unique to style
// one routine per atom style
// ---------------------------------------------------------------------

void Data::write_atom_angle_extra(FILE *fp, int i)
{
  fprintf(fp," %d",molecule[i]);
}

void Data::write_atom_atomic_extra(FILE *fp, int i) {}

void Data::write_atom_bond_extra(FILE *fp, int i)
{
  fprintf(fp," %d",molecule[i]);
}

void Data::write_atom_charge_extra(FILE *fp, int i)
{
  fprintf(fp," %-1.16e",q[i]);
}

void Data::write_atom_dipole_extra(FILE *fp, int i)
{
  fprintf(fp," %-1.16e %-1.16e %-1.16e %-1.16e",q[i],mux[i],muy[i],muz[i]);
}

void Data::write_atom_dpd_extra(FILE *fp, int i) {}

void Data::write_atom_ellipsoid_extra(FILE *fp, int i)
{
  fprintf(fp," %-1.16e %-1.16e %-1.16e %-1.16e",quatw[i],quati[i],quatj[i],quatk[i]);
}

void Data::write_atom_full_extra(FILE *fp, int i)
{
  fprintf(fp," %d %-1.16e",molecule[i],q[i]);
}

void Data::write_atom_granular_extra(FILE *fp, int i)
{
  fprintf(fp," %-1.16e %-1.16e",2.0*radius[i],density[i]);
}

void Data::write_atom_molecular_extra(FILE *fp, int i)
{
  fprintf(fp," %d",molecule[i]);
}

void Data::write_atom_peri_extra(FILE *fp, int i)
{
  fprintf(fp," %-1.16e %-1.16e %-1.16e",vfrac[i],density[i],rmass[i]);
}

void Data::write_atom_electron_extra(FILE *fp, int i)
{
  fprintf(fp," %-1.16e %d %-1.16e",q[i],spin[i],radius[i]);
}

// ---------------------------------------------------------------------
// per-atom velocity write routines
// one routine per atom style
// ---------------------------------------------------------------------

void Data::write_vel_angle(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e",tag[i],vx[i],vy[i],vz[i]);
}

void Data::write_vel_atomic(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e",tag[i],vx[i],vy[i],vz[i]);
}

void Data::write_vel_bond(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e",tag[i],vx[i],vy[i],vz[i]);
}

void Data::write_vel_charge(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e",tag[i],vx[i],vy[i],vz[i]);
}

void Data::write_vel_dipole(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e",tag[i],vx[i],vy[i],vz[i]);
}

void Data::write_vel_dpd(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e",tag[i],vx[i],vy[i],vz[i]);
}

void Data::write_vel_ellipsoid(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e",
	  tag[i],vx[i],vy[i],vz[i],angmomx[i],angmomy[i],angmomz[i]);
}

void Data::write_vel_full(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e",tag[i],vx[i],vy[i],vz[i]);
}

void Data::write_vel_granular(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e",
	  tag[i],vx[i],vy[i],vz[i],omegax[i],omegay[i],omegaz[i]);
}

void Data::write_vel_molecular(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e",tag[i],vx[i],vy[i],vz[i]);
}

void Data::write_vel_peri(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e",tag[i],vx[i],vy[i],vz[i]);
}

void Data::write_vel_electron(FILE *fp, int i)
{
  fprintf(fp,"%d %-1.16e %-1.16e %-1.16e %-1.16e",tag[i],vx[i],vy[i],vz[i],ervel[i]);
}

// ---------------------------------------------------------------------
// per-atom velocity write routines of extra quantities unique to style
// one routine per atom style
// ---------------------------------------------------------------------

void Data::write_vel_angle_extra(FILE *fp, int i) {}
void Data::write_vel_atomic_extra(FILE *fp, int i) {}
void Data::write_vel_bond_extra(FILE *fp, int i) {}
void Data::write_vel_charge_extra(FILE *fp, int i) {}
void Data::write_vel_dipole_extra(FILE *fp, int i) {}
void Data::write_vel_dpd_extra(FILE *fp, int i) {}

void Data::write_vel_ellipsoid_extra(FILE *fp, int i)
{
  fprintf(fp," %-1.16e %-1.16e %-1.16e",angmomx[i],angmomy[i],angmomz[i]);
}

void Data::write_vel_full_extra(FILE *fp, int i) {}

void Data::write_vel_granular_extra(FILE *fp, int i)
{
  fprintf(fp," %-1.16e %-1.16e %-1.16e",omegax[i],omegay[i],omegaz[i]);
}

void Data::write_vel_molecular_extra(FILE *fp, int i) {}
void Data::write_vel_peri_extra(FILE *fp, int i) {}
void Data::write_vel_electron_extra(FILE *fp, int i) 
{
  fprintf(fp," %-1.16e",ervel[i]);
}
// ---------------------------------------------------------------------
// binary reads from restart file
// ---------------------------------------------------------------------

int read_int(FILE *fp)
{
  int value;
  fread(&value,sizeof(int),1,fp);
  return value;
}

double read_double(FILE *fp)
{
  double value;
  fread(&value,sizeof(double),1,fp);
  return value;
}

char *read_char(FILE *fp)
{
  int n;
  fread(&n,sizeof(int),1,fp);
  if (n == 0) return NULL;
  char *value = new char[n];
  fread(value,sizeof(char),n,fp);
  return value;
}
