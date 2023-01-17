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
   Contributing author (ratio and subset) : Jake Gissinger (U Colorado)
------------------------------------------------------------------------- */

#include "create_atoms.h"

#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "irregular.h"
#include "lattice.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "random_mars.h"
#include "random_park.h"
#include "region.h"
#include "special.h"
#include "variable.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

#define BIG 1.0e30
#define EPSILON 1.0e-6
#define LB_FACTOR 1.1

enum{BOX,REGION,SINGLE,RANDOM};
enum{ATOM,MOLECULE};
enum{COUNT,INSERT,INSERT_SELECTED};
enum{NONE,RATIO,SUBSET};

/* ---------------------------------------------------------------------- */

CreateAtoms::CreateAtoms(LAMMPS *lmp) : Command(lmp), basistype(nullptr) {}

/* ---------------------------------------------------------------------- */

void CreateAtoms::command(int narg, char **arg)
{
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  if (domain->box_exist == 0)
    error->all(FLERR,"Create_atoms command before simulation box is defined");
  if (modify->nfix_restart_peratom)
    error->all(FLERR,"Cannot create_atoms after "
               "reading restart file with per-atom info");

  // check for compatible lattice

  int latsty = domain->lattice->style;
  if (domain->dimension == 2) {
    if (latsty == Lattice::SC || latsty == Lattice::BCC
        || latsty == Lattice::FCC || latsty == Lattice::HCP
        || latsty == Lattice::DIAMOND)
      error->all(FLERR,"Lattice style incompatible with simulation dimension");
  } else {
    if (latsty == Lattice::SQ ||latsty == Lattice::SQ2
        || latsty == Lattice::HEX)
      error->all(FLERR,"Lattice style incompatible with simulation dimension");
  }

  // parse arguments

  if (narg < 2) error->all(FLERR,"Illegal create_atoms command");
  ntype = utils::inumeric(FLERR,arg[0],false,lmp);

  int iarg;
  if (strcmp(arg[1],"box") == 0) {
    style = BOX;
    iarg = 2;
    nregion = -1;
  } else if (strcmp(arg[1],"region") == 0) {
    style = REGION;
    if (narg < 3) error->all(FLERR,"Illegal create_atoms command");
    nregion = domain->find_region(arg[2]);
    if (nregion == -1) error->all(FLERR,
                                  "Create_atoms region ID does not exist");
    domain->regions[nregion]->init();
    domain->regions[nregion]->prematch();
    iarg = 3;;
  } else if (strcmp(arg[1],"single") == 0) {
    style = SINGLE;
    if (narg < 5) error->all(FLERR,"Illegal create_atoms command");
    xone[0] = utils::numeric(FLERR,arg[2],false,lmp);
    xone[1] = utils::numeric(FLERR,arg[3],false,lmp);
    xone[2] = utils::numeric(FLERR,arg[4],false,lmp);
    iarg = 5;
  } else if (strcmp(arg[1],"random") == 0) {
    style = RANDOM;
    if (narg < 5) error->all(FLERR,"Illegal create_atoms command");
    nrandom = utils::inumeric(FLERR,arg[2],false,lmp);
    seed = utils::inumeric(FLERR,arg[3],false,lmp);
    if (strcmp(arg[4],"NULL") == 0) nregion = -1;
    else {
      nregion = domain->find_region(arg[4]);
      if (nregion == -1) error->all(FLERR,
                                    "Create_atoms region ID does not exist");
      domain->regions[nregion]->init();
      domain->regions[nregion]->prematch();
    }
    iarg = 5;
  } else error->all(FLERR,"Illegal create_atoms command");

  // process optional keywords

  int scaleflag = 1;
  remapflag = 0;
  mode = ATOM;
  int molseed;
  varflag = 0;
  vstr = xstr = ystr = zstr = nullptr;
  quatone[0] = quatone[1] = quatone[2] = 0.0;
  subsetflag = NONE;
  int subsetseed;

  nbasis = domain->lattice->nbasis;
  basistype = new int[nbasis];
  for (int i = 0; i < nbasis; i++) basistype[i] = ntype;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"basis") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal create_atoms command");
      int ibasis = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      int itype = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if (ibasis <= 0 || ibasis > nbasis || itype <= 0 || itype > atom->ntypes)
        error->all(FLERR,"Invalid basis setting in create_atoms command");
      basistype[ibasis-1] = itype;
      iarg += 3;
    } else if (strcmp(arg[iarg],"remap") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal create_atoms command");
      if (strcmp(arg[iarg+1],"yes") == 0) remapflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) remapflag = 0;
      else error->all(FLERR,"Illegal create_atoms command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"mol") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal create_atoms command");
      int imol = atom->find_molecule(arg[iarg+1]);
      if (imol == -1) error->all(FLERR,"Molecule template ID for "
                                 "create_atoms does not exist");
      if (atom->molecules[imol]->nset > 1 && me == 0)
        error->warning(FLERR,"Molecule template for "
                       "create_atoms has multiple molecules");
      mode = MOLECULE;
      onemol = atom->molecules[imol];
      molseed = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg],"units") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal create_atoms command");
      if (strcmp(arg[iarg+1],"box") == 0) scaleflag = 0;
      else if (strcmp(arg[iarg+1],"lattice") == 0) scaleflag = 1;
      else error->all(FLERR,"Illegal create_atoms command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"var") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal create_atoms command");
      delete [] vstr;
      vstr = utils::strdup(arg[iarg+1]);
      varflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"set") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal create_atoms command");
      if (strcmp(arg[iarg+1],"x") == 0) {
        delete [] xstr;
        xstr = utils::strdup(arg[iarg+2]);
      } else if (strcmp(arg[iarg+1],"y") == 0) {
        delete [] ystr;
        ystr = utils::strdup(arg[iarg+2]);
      } else if (strcmp(arg[iarg+1],"z") == 0) {
        delete [] zstr;
        zstr = utils::strdup(arg[iarg+2]);
      } else error->all(FLERR,"Illegal create_atoms command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"rotate") == 0) {
      if (iarg+5 > narg) error->all(FLERR,"Illegal create_atoms command");
      double thetaone;
      double axisone[3];
      thetaone = utils::numeric(FLERR,arg[iarg+1],false,lmp) / 180.0 * MY_PI;;
      axisone[0] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      axisone[1] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      axisone[2] = utils::numeric(FLERR,arg[iarg+4],false,lmp);
      if (axisone[0] == 0.0 && axisone[1] == 0.0 && axisone[2] == 0.0)
        error->all(FLERR,"Illegal create_atoms command");
      if (domain->dimension == 2 && (axisone[0] != 0.0 || axisone[1] != 0.0))
        error->all(FLERR,"Invalid create_atoms rotation vector for 2d model");
      MathExtra::norm3(axisone);
      MathExtra::axisangle_to_quat(axisone,thetaone,quatone);
      iarg += 5;
    } else if (strcmp(arg[iarg],"ratio") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal create_atoms command");
      subsetflag = RATIO;
      subsetfrac = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      subsetseed = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if (subsetfrac <= 0.0 || subsetfrac > 1.0 || subsetseed <= 0)
        error->all(FLERR,"Illegal create_atoms command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"subset") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal create_atoms command");
      subsetflag = SUBSET;
      nsubset = utils::bnumeric(FLERR,arg[iarg+1],false,lmp);
      subsetseed = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      if (nsubset <= 0 || subsetseed <= 0)
        error->all(FLERR,"Illegal create_atoms command");
      iarg += 3;
    } else error->all(FLERR,"Illegal create_atoms command");
  }

  // error checks

  if (mode == ATOM && (ntype <= 0 || ntype > atom->ntypes))
    error->all(FLERR,"Invalid atom type in create_atoms command");

  if (style == RANDOM) {
    if (nrandom < 0) error->all(FLERR,"Illegal create_atoms command");
    if (seed <= 0) error->all(FLERR,"Illegal create_atoms command");
  }

  // error check and further setup for mode = MOLECULE

  ranmol = nullptr;
  if (mode == MOLECULE) {
    if (onemol->xflag == 0)
      error->all(FLERR,"Create_atoms molecule must have coordinates");
    if (onemol->typeflag == 0)
      error->all(FLERR,"Create_atoms molecule must have atom types");
    if (ntype+onemol->ntypes <= 0 || ntype+onemol->ntypes > atom->ntypes)
      error->all(FLERR,"Invalid atom type in create_atoms mol command");
    if (onemol->tag_require && !atom->tag_enable)
      error->all(FLERR,
                 "Create_atoms molecule has atom IDs, but system does not");
    onemol->check_attributes(0);

    // create_atoms uses geoemetric center of molecule for insertion

    onemol->compute_center();

    // molecule random number generator, different for each proc

    ranmol = new RanMars(lmp,molseed+me);
  }

  ranlatt = nullptr;
  if (subsetflag != NONE) ranlatt = new RanMars(lmp,subsetseed+me);

  // error check and further setup for variable test

  if (!vstr && (xstr || ystr || zstr))
    error->all(FLERR,"Incomplete use of variables in create_atoms command");
  if (vstr && (!xstr && !ystr && !zstr))
    error->all(FLERR,"Incomplete use of variables in create_atoms command");

  if (varflag) {
    vvar = input->variable->find(vstr);
    if (vvar < 0)
      error->all(FLERR,"Variable name for create_atoms does not exist");
    if (!input->variable->equalstyle(vvar))
      error->all(FLERR,"Variable for create_atoms is invalid style");

    if (xstr) {
      xvar = input->variable->find(xstr);
      if (xvar < 0)
        error->all(FLERR,"Variable name for create_atoms does not exist");
      if (!input->variable->internalstyle(xvar))
        error->all(FLERR,"Variable for create_atoms is invalid style");
    }
    if (ystr) {
      yvar = input->variable->find(ystr);
      if (yvar < 0)
        error->all(FLERR,"Variable name for create_atoms does not exist");
      if (!input->variable->internalstyle(yvar))
        error->all(FLERR,"Variable for create_atoms is invalid style");
    }
    if (zstr) {
      zvar = input->variable->find(zstr);
      if (zvar < 0)
        error->all(FLERR,"Variable name for create_atoms does not exist");
      if (!input->variable->internalstyle(zvar))
        error->all(FLERR,"Variable for create_atoms is invalid style");
    }
  }

  // demand non-none lattice be defined for BOX and REGION
  // else setup scaling for SINGLE and RANDOM
  // could use domain->lattice->lattice2box() to do conversion of
  //   lattice to box, but not consistent with other uses of units=lattice
  // triclinic remapping occurs in add_single()

  if (style == BOX || style == REGION) {
    if (nbasis == 0)
      error->all(FLERR,"Cannot create atoms with undefined lattice");
  } else if (scaleflag == 1) {
    xone[0] *= domain->lattice->xlattice;
    xone[1] *= domain->lattice->ylattice;
    xone[2] *= domain->lattice->zlattice;
  }

  // set bounds for my proc in sublo[3] & subhi[3]
  // if periodic and style = BOX or REGION, i.e. using lattice:
  //   should create exactly 1 atom when 2 images are both "on" the boundary
  //   either image may be slightly inside/outside true box due to round-off
  //   if I am lo proc, decrement lower bound by EPSILON
  //     this will insure lo image is created
  //   if I am hi proc, decrement upper bound by 2.0*EPSILON
  //     this will insure hi image is not created
  //   thus insertion box is EPSILON smaller than true box
  //     and is shifted away from true boundary
  //     which is where atoms are likely to be generated

  triclinic = domain->triclinic;

  double epsilon[3];
  if (triclinic) epsilon[0] = epsilon[1] = epsilon[2] = EPSILON;
  else {
    epsilon[0] = domain->prd[0] * EPSILON;
    epsilon[1] = domain->prd[1] * EPSILON;
    epsilon[2] = domain->prd[2] * EPSILON;
  }

  if (triclinic == 0) {
    sublo[0] = domain->sublo[0]; subhi[0] = domain->subhi[0];
    sublo[1] = domain->sublo[1]; subhi[1] = domain->subhi[1];
    sublo[2] = domain->sublo[2]; subhi[2] = domain->subhi[2];
  } else {
    sublo[0] = domain->sublo_lamda[0]; subhi[0] = domain->subhi_lamda[0];
    sublo[1] = domain->sublo_lamda[1]; subhi[1] = domain->subhi_lamda[1];
    sublo[2] = domain->sublo_lamda[2]; subhi[2] = domain->subhi_lamda[2];
  }

  if (style == BOX || style == REGION) {
    if (comm->layout != Comm::LAYOUT_TILED) {
      if (domain->xperiodic) {
        if (comm->myloc[0] == 0) sublo[0] -= epsilon[0];
        if (comm->myloc[0] == comm->procgrid[0]-1) subhi[0] -= 2.0*epsilon[0];
      }
      if (domain->yperiodic) {
        if (comm->myloc[1] == 0) sublo[1] -= epsilon[1];
        if (comm->myloc[1] == comm->procgrid[1]-1) subhi[1] -= 2.0*epsilon[1];
      }
      if (domain->zperiodic) {
        if (comm->myloc[2] == 0) sublo[2] -= epsilon[2];
        if (comm->myloc[2] == comm->procgrid[2]-1) subhi[2] -= 2.0*epsilon[2];
      }
    } else {
      if (domain->xperiodic) {
        if (comm->mysplit[0][0] == 0.0) sublo[0] -= epsilon[0];
        if (comm->mysplit[0][1] == 1.0) subhi[0] -= 2.0*epsilon[0];
      }
      if (domain->yperiodic) {
        if (comm->mysplit[1][0] == 0.0) sublo[1] -= epsilon[1];
        if (comm->mysplit[1][1] == 1.0) subhi[1] -= 2.0*epsilon[1];
      }
      if (domain->zperiodic) {
        if (comm->mysplit[2][0] == 0.0) sublo[2] -= epsilon[2];
        if (comm->mysplit[2][1] == 1.0) subhi[2] -= 2.0*epsilon[2];
      }
    }
  }

  // Record wall time for atom creation

  MPI_Barrier(world);
  double time1 = MPI_Wtime();

  // clear ghost count and any ghost bonus data internal to AtomVec
  // same logic as beginning of Comm::exchange()
  // do it now b/c creating atoms will overwrite ghost atoms

  atom->nghost = 0;
  atom->avec->clear_bonus();

  // add atoms/molecules in one of 3 ways

  bigint natoms_previous = atom->natoms;
  int nlocal_previous = atom->nlocal;

  if (style == SINGLE) add_single();
  else if (style == RANDOM) add_random();
  else add_lattice();

  // init per-atom fix/compute/variable values for created atoms

  atom->data_fix_compute_variable(nlocal_previous,atom->nlocal);

  // set new total # of atoms and error check

  bigint nblocal = atom->nlocal;
  MPI_Allreduce(&nblocal,&atom->natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (atom->natoms < 0 || atom->natoms >= MAXBIGINT)
    error->all(FLERR,"Too many total atoms");

  // add IDs for newly created atoms
  // check that atom IDs are valid

  if (atom->tag_enable) atom->tag_extend();
  atom->tag_check();

  // if global map exists, reset it
  // invoke map_init() b/c atom count has grown

  if (atom->map_style != Atom::MAP_NONE) {
    atom->map_init();
    atom->map_set();
  }

  // for MOLECULE mode:
  // molecule can mean just a mol ID or bonds/angles/etc or mol templates
  // set molecule IDs for created atoms if atom->molecule_flag is set
  // reset new molecule bond,angle,etc and special values if defined
  // send atoms to new owning procs via irregular comm
  //   since not all atoms I created will be within my sub-domain
  // perform special list build if needed

  if (mode == MOLECULE) {

    int molecule_flag = atom->molecule_flag;
    int molecular = atom->molecular;
    tagint *molecule = atom->molecule;

    // molcreate = # of molecules I created

    tagint molcreate = (atom->nlocal - nlocal_previous) / onemol->natoms;

    // increment total bonds,angles,etc

    bigint nmolme = molcreate;
    bigint nmoltotal;
    MPI_Allreduce(&nmolme,&nmoltotal,1,MPI_LMP_BIGINT,MPI_SUM,world);
    atom->nbonds += nmoltotal * onemol->nbonds;
    atom->nangles += nmoltotal * onemol->nangles;
    atom->ndihedrals += nmoltotal * onemol->ndihedrals;
    atom->nimpropers += nmoltotal * onemol->nimpropers;

    // if atom style template
    // maxmol = max molecule ID across all procs, for previous atoms
    // moloffset = max molecule ID for all molecules owned by previous procs
    //             including molecules existing before this creation

    tagint moloffset = 0;
    if (molecule_flag) {
      tagint max = 0;
      for (int i = 0; i < nlocal_previous; i++) max = MAX(max,molecule[i]);
      tagint maxmol;
      MPI_Allreduce(&max,&maxmol,1,MPI_LMP_TAGINT,MPI_MAX,world);
      MPI_Scan(&molcreate,&moloffset,1,MPI_LMP_TAGINT,MPI_SUM,world);
      moloffset = moloffset - molcreate + maxmol;
    }

    // loop over molecules I created
    // set their molecule ID
    // reset their bond,angle,etc and special values

    int natoms = onemol->natoms;
    tagint offset = 0;

    tagint *tag = atom->tag;
    int *num_bond = atom->num_bond;
    int *num_angle = atom->num_angle;
    int *num_dihedral = atom->num_dihedral;
    int *num_improper = atom->num_improper;
    tagint **bond_atom = atom->bond_atom;
    tagint **angle_atom1 = atom->angle_atom1;
    tagint **angle_atom2 = atom->angle_atom2;
    tagint **angle_atom3 = atom->angle_atom3;
    tagint **dihedral_atom1 = atom->dihedral_atom1;
    tagint **dihedral_atom2 = atom->dihedral_atom2;
    tagint **dihedral_atom3 = atom->dihedral_atom3;
    tagint **dihedral_atom4 = atom->dihedral_atom4;
    tagint **improper_atom1 = atom->improper_atom1;
    tagint **improper_atom2 = atom->improper_atom2;
    tagint **improper_atom3 = atom->improper_atom3;
    tagint **improper_atom4 = atom->improper_atom4;
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    int ilocal = nlocal_previous;
    for (int i = 0; i < molcreate; i++) {
      if (tag) offset = tag[ilocal]-1;
      for (int m = 0; m < natoms; m++) {
        if (molecule_flag) {
          if (onemol->moleculeflag) {
            molecule[ilocal] = moloffset + onemol->molecule[m];
          } else {
            molecule[ilocal] = moloffset + 1;
          }
        }
        if (molecular == Atom::TEMPLATE) {
          atom->molindex[ilocal] = 0;
          atom->molatom[ilocal] = m;
        } else if (molecular != Atom::ATOMIC) {
          if (onemol->bondflag)
            for (int j = 0; j < num_bond[ilocal]; j++)
              bond_atom[ilocal][j] += offset;
          if (onemol->angleflag)
            for (int j = 0; j < num_angle[ilocal]; j++) {
              angle_atom1[ilocal][j] += offset;
              angle_atom2[ilocal][j] += offset;
              angle_atom3[ilocal][j] += offset;
            }
          if (onemol->dihedralflag)
            for (int j = 0; j < num_dihedral[ilocal]; j++) {
              dihedral_atom1[ilocal][j] += offset;
              dihedral_atom2[ilocal][j] += offset;
              dihedral_atom3[ilocal][j] += offset;
              dihedral_atom4[ilocal][j] += offset;
            }
          if (onemol->improperflag)
            for (int j = 0; j < num_improper[ilocal]; j++) {
              improper_atom1[ilocal][j] += offset;
              improper_atom2[ilocal][j] += offset;
              improper_atom3[ilocal][j] += offset;
              improper_atom4[ilocal][j] += offset;
            }
          if (onemol->specialflag)
            for (int j = 0; j < nspecial[ilocal][2]; j++)
              special[ilocal][j] += offset;
        }
        ilocal++;
      }
      if (molecule_flag) {
        if (onemol->moleculeflag) {
          moloffset += onemol->nmolecules;
        } else {
          moloffset++;
        }
      }
    }

    // perform irregular comm to migrate atoms to new owning procs

    double **x = atom->x;
    imageint *image = atom->image;
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);

    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->reset_box();
    Irregular *irregular = new Irregular(lmp);
    irregular->migrate_atoms(1);
    delete irregular;
    if (domain->triclinic) domain->lamda2x(atom->nlocal);
  }

  // clean up

  delete ranmol;
  delete ranlatt;

  delete [] basistype;
  delete [] vstr;
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;

  // for MOLECULE mode:
  // create special bond lists for molecular systems,
  //   but not for atom style template
  // only if onemol added bonds but not special info

  if (mode == MOLECULE) {
    if (atom->molecular == Atom::MOLECULAR && onemol->bondflag && !onemol->specialflag) {
      Special special(lmp);
      special.build();

    }
  }

  // print status

  MPI_Barrier(world);
  if (me == 0) {
    utils::logmesg(lmp,"Created {} atoms\n", atom->natoms - natoms_previous);
    if (scaleflag) domain->print_box("  using lattice units in ");
    else domain->print_box("  using box units in ");
    utils::logmesg(lmp,"  create_atoms CPU = {:.3f} seconds\n",
                   MPI_Wtime() - time1);
  }
}

/* ----------------------------------------------------------------------
   add single atom with coords at xone if it's in my sub-box
   if triclinic, xone is in lamda coords
------------------------------------------------------------------------- */

void CreateAtoms::add_single()
{
  // remap atom if requested

  if (remapflag) {
    imageint imagetmp = ((imageint) IMGMAX << IMG2BITS) |
      ((imageint) IMGMAX << IMGBITS) | IMGMAX;
    domain->remap(xone,imagetmp);
  }

  // if triclinic, convert to lamda coords (0-1)
  // with remapflag set and periodic dims,
  //   resulting coord must satisfy 0.0 <= coord < 1.0

  double lamda[3],*coord;
  if (triclinic) {
    domain->x2lamda(xone,lamda);
    if (remapflag) {
      if (domain->xperiodic && (lamda[0] < 0.0 || lamda[0] >= 1.0))
        lamda[0] = 0.0;
      if (domain->yperiodic && (lamda[1] < 0.0 || lamda[1] >= 1.0))
        lamda[1] = 0.0;
      if (domain->zperiodic && (lamda[2] < 0.0 || lamda[2] >= 1.0))
        lamda[2] = 0.0;
    }
    coord = lamda;
  } else coord = xone;

  // if atom/molecule is in my subbox, create it

  if (coord[0] >= sublo[0] && coord[0] < subhi[0] &&
      coord[1] >= sublo[1] && coord[1] < subhi[1] &&
      coord[2] >= sublo[2] && coord[2] < subhi[2]) {
    if (mode == ATOM) atom->avec->create_atom(ntype,xone);
    else if (quatone[0] == 0.0 && quatone[1] == 0.0 && quatone[2] == 0.0)
      add_molecule(xone);
    else add_molecule(xone,quatone);
  }
}

/* ----------------------------------------------------------------------
   add Nrandom atoms at random locations
------------------------------------------------------------------------- */

void CreateAtoms::add_random()
{
  double xlo,ylo,zlo,xhi,yhi,zhi,zmid;
  double lamda[3],*coord;
  double *boxlo,*boxhi;

  // random number generator, same for all procs
  // warm up the generator 30x to avoid correlations in first-particle
  // positions if runs are repeated with consecutive seeds

  RanPark *random = new RanPark(lmp,seed);
  for (int ii=0; ii < 30; ii++) random->uniform();

  // bounding box for atom creation
  // in real units, even if triclinic
  // only limit bbox by region if its bboxflag is set (interior region)

  if (triclinic == 0) {
    xlo = domain->boxlo[0]; xhi = domain->boxhi[0];
    ylo = domain->boxlo[1]; yhi = domain->boxhi[1];
    zlo = domain->boxlo[2]; zhi = domain->boxhi[2];
    zmid = zlo + 0.5*(zhi-zlo);
  } else {
    xlo = domain->boxlo_bound[0]; xhi = domain->boxhi_bound[0];
    ylo = domain->boxlo_bound[1]; yhi = domain->boxhi_bound[1];
    zlo = domain->boxlo_bound[2]; zhi = domain->boxhi_bound[2];
    zmid = zlo + 0.5*(zhi-zlo);
    boxlo = domain->boxlo_lamda;
    boxhi = domain->boxhi_lamda;
  }

  if (nregion >= 0 && domain->regions[nregion]->bboxflag) {
    xlo = MAX(xlo,domain->regions[nregion]->extent_xlo);
    xhi = MIN(xhi,domain->regions[nregion]->extent_xhi);
    ylo = MAX(ylo,domain->regions[nregion]->extent_ylo);
    yhi = MIN(yhi,domain->regions[nregion]->extent_yhi);
    zlo = MAX(zlo,domain->regions[nregion]->extent_zlo);
    zhi = MIN(zhi,domain->regions[nregion]->extent_zhi);
  }

  // generate random positions for each new atom/molecule within bounding box
  // iterate until atom is within region, variable, and triclinic simulation box
  // if final atom position is in my subbox, create it

  if (xlo > xhi || ylo > yhi || zlo > zhi)
    error->all(FLERR,"No overlap of box and region for create_atoms");

  int valid;
  for (int i = 0; i < nrandom; i++) {
    while (1) {
      xone[0] = xlo + random->uniform() * (xhi-xlo);
      xone[1] = ylo + random->uniform() * (yhi-ylo);
      xone[2] = zlo + random->uniform() * (zhi-zlo);
      if (domain->dimension == 2) xone[2] = zmid;

      valid = 1;
      if (nregion >= 0 &&
          domain->regions[nregion]->match(xone[0],xone[1],xone[2]) == 0)
        valid = 0;
      if (varflag && vartest(xone) == 0) valid = 0;
      if (triclinic) {
        domain->x2lamda(xone,lamda);
        coord = lamda;
        if (coord[0] < boxlo[0] || coord[0] >= boxhi[0] ||
            coord[1] < boxlo[1] || coord[1] >= boxhi[1] ||
            coord[2] < boxlo[2] || coord[2] >= boxhi[2]) valid = 0;
      } else coord = xone;

      if (valid) break;
    }

    // if triclinic, coord is now in lamda units

    if (coord[0] >= sublo[0] && coord[0] < subhi[0] &&
        coord[1] >= sublo[1] && coord[1] < subhi[1] &&
        coord[2] >= sublo[2] && coord[2] < subhi[2]) {
      if (mode == ATOM) atom->avec->create_atom(ntype,xone);
      else if (quatone[0] == 0 && quatone[1] == 0 && quatone[2] == 0)
        add_molecule(xone);
      else add_molecule(xone, quatone);
    }
  }

  // clean-up

  delete random;
}

/* ----------------------------------------------------------------------
   add many atoms by looping over lattice
------------------------------------------------------------------------- */

void CreateAtoms::add_lattice()
{
  // convert 8 corners of my subdomain from box coords to lattice coords
  // for orthogonal, use corner pts of my subbox
  // for triclinic, use bounding box of my subbox
  // xyz min to max = bounding box around the domain corners in lattice space

  double bboxlo[3],bboxhi[3];

  if (triclinic == 0) {
    bboxlo[0] = domain->sublo[0]; bboxhi[0] = domain->subhi[0];
    bboxlo[1] = domain->sublo[1]; bboxhi[1] = domain->subhi[1];
    bboxlo[2] = domain->sublo[2]; bboxhi[2] = domain->subhi[2];
  } else domain->bbox(domain->sublo_lamda,domain->subhi_lamda,bboxlo,bboxhi);

  // narrow down the subbox by the bounding box of the given region, if available.
  // for small regions in large boxes, this can result in a significant speedup

  if ((style == REGION) && domain->regions[nregion]->bboxflag) {

    const double rxmin = domain->regions[nregion]->extent_xlo;
    const double rxmax = domain->regions[nregion]->extent_xhi;
    const double rymin = domain->regions[nregion]->extent_ylo;
    const double rymax = domain->regions[nregion]->extent_yhi;
    const double rzmin = domain->regions[nregion]->extent_zlo;
    const double rzmax = domain->regions[nregion]->extent_zhi;

    if (rxmin > bboxlo[0]) bboxlo[0] = (rxmin > bboxhi[0]) ? bboxhi[0] : rxmin;
    if (rxmax < bboxhi[0]) bboxhi[0] = (rxmax < bboxlo[0]) ? bboxlo[0] : rxmax;
    if (rymin > bboxlo[1]) bboxlo[1] = (rymin > bboxhi[1]) ? bboxhi[1] : rymin;
    if (rymax < bboxhi[1]) bboxhi[1] = (rymax < bboxlo[1]) ? bboxlo[1] : rymax;
    if (rzmin > bboxlo[2]) bboxlo[2] = (rzmin > bboxhi[2]) ? bboxhi[2] : rzmin;
    if (rzmax < bboxhi[2]) bboxhi[2] = (rzmax < bboxlo[2]) ? bboxlo[2] : rzmax;
  }

  double xmin,ymin,zmin,xmax,ymax,zmax;
  xmin = ymin = zmin = BIG;
  xmax = ymax = zmax = -BIG;

  // convert to lattice coordinates and set bounding box
  domain->lattice->bbox(1,bboxlo[0],bboxlo[1],bboxlo[2],
                        xmin,ymin,zmin,xmax,ymax,zmax);
  domain->lattice->bbox(1,bboxhi[0],bboxlo[1],bboxlo[2],
                        xmin,ymin,zmin,xmax,ymax,zmax);
  domain->lattice->bbox(1,bboxlo[0],bboxhi[1],bboxlo[2],
                        xmin,ymin,zmin,xmax,ymax,zmax);
  domain->lattice->bbox(1,bboxhi[0],bboxhi[1],bboxlo[2],
                        xmin,ymin,zmin,xmax,ymax,zmax);
  domain->lattice->bbox(1,bboxlo[0],bboxlo[1],bboxhi[2],
                        xmin,ymin,zmin,xmax,ymax,zmax);
  domain->lattice->bbox(1,bboxhi[0],bboxlo[1],bboxhi[2],
                        xmin,ymin,zmin,xmax,ymax,zmax);
  domain->lattice->bbox(1,bboxlo[0],bboxhi[1],bboxhi[2],
                        xmin,ymin,zmin,xmax,ymax,zmax);
  domain->lattice->bbox(1,bboxhi[0],bboxhi[1],bboxhi[2],
                        xmin,ymin,zmin,xmax,ymax,zmax);

  // ilo:ihi,jlo:jhi,klo:khi = loop bounds for lattice overlap of my subbox
  // overlap = any part of a unit cell (face,edge,pt) in common with my subbox
  // in lattice space, subbox is a tilted box
  // but bbox of subbox is aligned with lattice axes
  // so ilo:khi unit cells should completely tile bounding box
  // decrement lo, increment hi to avoid round-off issues in lattice->bbox(),
  //   which can lead to missing atoms in rare cases
  // extra decrement of lo if min < 0, since static_cast(-1.5) = -1

  ilo = static_cast<int> (xmin) - 1;
  jlo = static_cast<int> (ymin) - 1;
  klo = static_cast<int> (zmin) - 1;
  ihi = static_cast<int> (xmax) + 1;
  jhi = static_cast<int> (ymax) + 1;
  khi = static_cast<int> (zmax) + 1;

  if (xmin < 0.0) ilo--;
  if (ymin < 0.0) jlo--;
  if (zmin < 0.0) klo--;

  // count lattice sites on each proc

  nlatt_overflow = 0;
  loop_lattice(COUNT);

  // nadd = # of atoms each proc will insert (estimated if subsetflag)

  int overflow;
  MPI_Allreduce(&nlatt_overflow,&overflow,1,MPI_INT,MPI_SUM,world);
  if (overflow)
    error->all(FLERR,"Create_atoms lattice size overflow on 1 or more procs");

  bigint nadd;

  if (subsetflag == NONE) {
    if (nprocs == 1) nadd = nlatt;
    else nadd = static_cast<bigint> (LB_FACTOR * nlatt);
  } else {
    bigint bnlatt = nlatt;
    bigint bnlattall;
    MPI_Allreduce(&bnlatt,&bnlattall,1,MPI_LMP_BIGINT,MPI_SUM,world);
    if (subsetflag == RATIO)
      nsubset = static_cast<bigint> (subsetfrac * bnlattall);
    if (nsubset > bnlattall)
      error->all(FLERR,"Create_atoms subset size > # of lattice sites");
    if (nprocs == 1) nadd = nsubset;
    else nadd = static_cast<bigint> (LB_FACTOR * nsubset/bnlattall * nlatt);
  }

  // allocate atom arrays to size N, rounded up by AtomVec->DELTA

  bigint nbig = atom->avec->roundup(nadd + atom->nlocal);
  int n = static_cast<int> (nbig);
  atom->avec->grow(n);

  // add atoms or molecules
  // if no subset: add to all lattice sites
  // if subset: count lattice sites, select random subset, then add

  if (subsetflag == NONE) loop_lattice(INSERT);
  else {
    memory->create(flag,nlatt,"create_atoms:flag");
    memory->create(next,nlatt,"create_atoms:next");
    ranlatt->select_subset(nsubset,nlatt,flag,next);
    loop_lattice(INSERT_SELECTED);
    memory->destroy(flag);
    memory->destroy(next);
  }
}

/* ----------------------------------------------------------------------
   iterate on 3d periodic lattice of unit cells using loop bounds
   iterate on nbasis atoms in each unit cell
   convert lattice coords to box coords
   check if lattice point meets all criteria to be added
   perform action on atom or molecule (on each basis point) if meets all criteria
   actions = add, count, add if flagged
------------------------------------------------------------------------- */

void CreateAtoms::loop_lattice(int action)
{
  int i,j,k,m;

  const double * const * const basis = domain->lattice->basis;

  nlatt = 0;

  for (k = klo; k <= khi; k++) {
    for (j = jlo; j <= jhi; j++) {
      for (i = ilo; i <= ihi; i++) {
        for (m = 0; m < nbasis; m++) {
          double *coord;
          double x[3],lamda[3];

          x[0] = i + basis[m][0];
          x[1] = j + basis[m][1];
          x[2] = k + basis[m][2];

          // convert from lattice coords to box coords

          domain->lattice->lattice2box(x[0],x[1],x[2]);

          // if a region was specified, test if atom is in it

          if (style == REGION)
            if (!domain->regions[nregion]->match(x[0],x[1],x[2])) continue;

          // if variable test specified, eval variable

          if (varflag && vartest(x) == 0) continue;

          // test if atom/molecule position is in my subbox

          if (triclinic) {
            domain->x2lamda(x,lamda);
            coord = lamda;
          } else coord = x;

          if (coord[0] < sublo[0] || coord[0] >= subhi[0] ||
              coord[1] < sublo[1] || coord[1] >= subhi[1] ||
              coord[2] < sublo[2] || coord[2] >= subhi[2]) continue;

          // this proc owns the lattice site
          // perform action: add, just count, add if flagged
          // add = add an atom or entire molecule to my list of atoms

          if (action == INSERT) {
            if (mode == ATOM) atom->avec->create_atom(basistype[m],x);
            else if (quatone[0] == 0 && quatone[1] == 0 && quatone[2] == 0)
              add_molecule(x);
            else add_molecule(x,quatone);
          } else if (action == COUNT) {
            if (nlatt == MAXSMALLINT) nlatt_overflow = 1;
          } else if (action == INSERT_SELECTED && flag[nlatt]) {
            if (mode == ATOM) atom->avec->create_atom(basistype[m],x);
            else if (quatone[0] == 0 && quatone[1] == 0 && quatone[2] == 0)
              add_molecule(x);
            else add_molecule(x,quatone);
          }

          nlatt++;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   add a randomly rotated molecule with its center at center
   if quat_user set, perform requested rotation
------------------------------------------------------------------------- */

void CreateAtoms::add_molecule(double *center, double *quat_user)
{
  int n;
  double r[3],rotmat[3][3],quat[4],xnew[3];

  if (quat_user) {
    quat[0] = quat_user[0]; quat[1] = quat_user[1];
    quat[2] = quat_user[2]; quat[3] = quat_user[3];
  } else {
    if (domain->dimension == 3) {
      r[0] = ranmol->uniform() - 0.5;
      r[1] = ranmol->uniform() - 0.5;
      r[2] = ranmol->uniform() - 0.5;
    } else {
      r[0] = r[1] = 0.0;
      r[2] = 1.0;
    }
    MathExtra::norm3(r);
    double theta = ranmol->uniform() * MY_2PI;
    MathExtra::axisangle_to_quat(r,theta,quat);
  }

  MathExtra::quat_to_mat(quat,rotmat);
  onemol->quat_external = quat;

  // create atoms in molecule with atom ID = 0 and mol ID = 0
  // reset in caller after all molecules created by all procs
  // pass add_molecule_atom an offset of 0 since don't know
  //   max tag of atoms in previous molecules at this point

  int natoms = onemol->natoms;
  for (int m = 0; m < natoms; m++) {
    MathExtra::matvec(rotmat,onemol->dx[m],xnew);
    MathExtra::add3(xnew,center,xnew);
    atom->avec->create_atom(ntype+onemol->type[m],xnew);
    n = atom->nlocal - 1;
    atom->add_molecule_atom(onemol,m,n,0);
  }
}

/* ----------------------------------------------------------------------
   test a generated atom position against variable evaluation
   first set x,y,z values in internal variables
------------------------------------------------------------------------- */

int CreateAtoms::vartest(double *x)
{
  if (xstr) input->variable->internal_set(xvar,x[0]);
  if (ystr) input->variable->internal_set(yvar,x[1]);
  if (zstr) input->variable->internal_set(zvar,x[2]);

  double value = input->variable->compute_equal(vvar);

  if (value == 0.0) return 0;
  return 1;
}
