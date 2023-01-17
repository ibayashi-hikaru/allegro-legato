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

#include "fix_adapt.h"

#include "atom.h"
#include "bond.h"
#include "domain.h"
#include "error.h"
#include "fix_store.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{PAIR,KSPACE,ATOM,BOND};
enum{DIAMETER,CHARGE};

/* ---------------------------------------------------------------------- */

FixAdapt::FixAdapt(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg),
nadapt(0), id_fix_diam(nullptr), id_fix_chg(nullptr), adapt(nullptr)
{
  if (narg < 5) error->all(FLERR,"Illegal fix adapt command");
  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery < 0) error->all(FLERR,"Illegal fix adapt command");

  dynamic_group_allow = 1;
  create_attribute = 1;

  // count # of adaptations

  nadapt = 0;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"pair") == 0) {
      if (iarg+6 > narg) error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 6;
    } else if (strcmp(arg[iarg],"kspace") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 2;
    } else if (strcmp(arg[iarg],"atom") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 3;
    } else if (strcmp(arg[iarg],"bond") == 0) {
      if (iarg+5 > narg) error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 5;
    } else break;
  }

  if (nadapt == 0) error->all(FLERR,"Illegal fix adapt command");
  adapt = new Adapt[nadapt];

  // parse keywords

  nadapt = 0;
  diamflag = 0;
  chgflag = 0;

  iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"pair") == 0) {
      if (iarg+6 > narg) error->all(FLERR,"Illegal fix adapt command");
      adapt[nadapt].which = PAIR;
      adapt[nadapt].pair = nullptr;
      adapt[nadapt].pstyle = utils::strdup(arg[iarg+1]);
      adapt[nadapt].pparam = utils::strdup(arg[iarg+2]);
      utils::bounds(FLERR,arg[iarg+3],1,atom->ntypes,
                    adapt[nadapt].ilo,adapt[nadapt].ihi,error);
      utils::bounds(FLERR,arg[iarg+4],1,atom->ntypes,
                    adapt[nadapt].jlo,adapt[nadapt].jhi,error);
      if (utils::strmatch(arg[iarg+5],"^v_")) {
        adapt[nadapt].var = utils::strdup(arg[iarg+5]+2);
      } else error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 6;

    } else if (strcmp(arg[iarg],"bond") == 0) {
      if (iarg+5 > narg) error->all(FLERR, "Illegal fix adapt command");
      adapt[nadapt].which = BOND;
      adapt[nadapt].bond = nullptr;
      adapt[nadapt].bstyle = utils::strdup(arg[iarg+1]);
      adapt[nadapt].bparam = utils::strdup(arg[iarg+2]);
      utils::bounds(FLERR,arg[iarg+3],1,atom->nbondtypes,
                    adapt[nadapt].ilo,adapt[nadapt].ihi,error);
      if (utils::strmatch(arg[iarg+4],"^v_")) {
        adapt[nadapt].var = utils::strdup(arg[iarg+4]+2);
      } else error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 5;

    } else if (strcmp(arg[iarg],"kspace") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix adapt command");
      adapt[nadapt].which = KSPACE;
      if (utils::strmatch(arg[iarg+1],"^v_")) {
        adapt[nadapt].var = utils::strdup(arg[iarg+1]+2);
      } else error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 2;

    } else if (strcmp(arg[iarg],"atom") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix adapt command");
      adapt[nadapt].which = ATOM;
      if (strcmp(arg[iarg+1],"diameter") == 0 ||
          strcmp(arg[iarg+1],"diameter/disc") == 0) {
        adapt[nadapt].aparam = DIAMETER;
        diamflag = 1;
        discflag = 0;
        if (strcmp(arg[iarg+1],"diameter/disc") == 0) discflag = 1;
      } else if (strcmp(arg[iarg+1],"charge") == 0) {
        adapt[nadapt].aparam = CHARGE;
        chgflag = 1;
      } else error->all(FLERR,"Illegal fix adapt command");
      if (utils::strmatch(arg[iarg+2],"^v_")) {
        adapt[nadapt].var = utils::strdup(arg[iarg+2]+2);
      } else error->all(FLERR,"Illegal fix adapt command");
      nadapt++;
      iarg += 3;
    } else break;
  }

  // optional keywords

  resetflag = 0;
  scaleflag = 0;
  massflag = 1;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"reset") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix adapt command");
      if (strcmp(arg[iarg+1],"no") == 0) resetflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) resetflag = 1;
      else error->all(FLERR,"Illegal fix adapt command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"scale") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix adapt command");
      if (strcmp(arg[iarg+1],"no") == 0) scaleflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) scaleflag = 1;
      else error->all(FLERR,"Illegal fix adapt command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"mass") == 0) {
      if (iarg+2 > narg)error->all(FLERR,"Illegal fix adapt command");
      if (strcmp(arg[iarg+1],"no") == 0) massflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) massflag = 1;
      else error->all(FLERR,"Illegal fix adapt command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix adapt command");
  }

  // if scaleflag set with diameter or charge adaptation,
  // then previous step scale factors are written to restart file
  // initialize them here in case one is used and other is never defined

  if (scaleflag && (diamflag || chgflag)) restart_global = 1;
  previous_diam_scale = previous_chg_scale = 1.0;

  // allocate pair style arrays

  int n = atom->ntypes;
  for (int m = 0; m < nadapt; m++)
    if (adapt[m].which == PAIR)
      memory->create(adapt[m].array_orig,n+1,n+1,"adapt:array_orig");

  // allocate bond style arrays:

  n = atom->nbondtypes;
  for (int m = 0; m < nadapt; ++m)
    if (adapt[m].which == BOND)
      memory->create(adapt[m].vector_orig,n+1,"adapt:vector_orig");
}

/* ---------------------------------------------------------------------- */

FixAdapt::~FixAdapt()
{
  for (int m = 0; m < nadapt; m++) {
    delete [] adapt[m].var;
    if (adapt[m].which == PAIR) {
      delete [] adapt[m].pstyle;
      delete [] adapt[m].pparam;
      memory->destroy(adapt[m].array_orig);
    } else if (adapt[m].which == BOND) {
      delete [] adapt[m].bstyle;
      delete [] adapt[m].bparam;
      memory->destroy(adapt[m].vector_orig);
    }
  }
  delete [] adapt;

  // check nfix in case all fixes have already been deleted

  if (id_fix_diam && modify->nfix) modify->delete_fix(id_fix_diam);
  if (id_fix_chg && modify->nfix) modify->delete_fix(id_fix_chg);
  delete [] id_fix_diam;
  delete [] id_fix_chg;
}

/* ---------------------------------------------------------------------- */

int FixAdapt::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= POST_RUN;
  mask |= PRE_FORCE_RESPA;
  return mask;
}

/* ----------------------------------------------------------------------
   if need to restore per-atom quantities, create new fix STORE styles
------------------------------------------------------------------------- */

void FixAdapt::post_constructor()
{
  if (!resetflag) return;
  if (!diamflag && !chgflag) return;

  // new id = fix-ID + FIX_STORE_ATTRIBUTE
  // new fix group = group for this fix

  id_fix_diam = nullptr;
  id_fix_chg = nullptr;

  if (diamflag && atom->radius_flag) {
    id_fix_diam = utils::strdup(id + std::string("_FIX_STORE_DIAM"));
    fix_diam = (FixStore *) modify->add_fix(fmt::format("{} {} STORE peratom 1 1",
                                                        id_fix_diam,group->names[igroup]));
    if (fix_diam->restart_reset) fix_diam->restart_reset = 0;
    else {
      double *vec = fix_diam->vstore;
      double *radius = atom->radius;
      int *mask = atom->mask;
      int nlocal = atom->nlocal;

      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) vec[i] = radius[i];
        else vec[i] = 0.0;
      }
    }
  }

  if (chgflag && atom->q_flag) {
    id_fix_chg = utils::strdup(id + std::string("_FIX_STORE_CHG"));
    fix_chg = (FixStore *) modify->add_fix(fmt::format("{} {} STORE peratom 1 1",
                                                       id_fix_chg,group->names[igroup]));
    if (fix_chg->restart_reset) fix_chg->restart_reset = 0;
    else {
      double *vec = fix_chg->vstore;
      double *q = atom->q;
      int *mask = atom->mask;
      int nlocal = atom->nlocal;

      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) vec[i] = q[i];
        else vec[i] = 0.0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixAdapt::init()
{
  int i,j;

  // allow a dynamic group only if ATOM attribute not used

  if (group->dynamic[igroup])
    for (i = 0; i < nadapt; i++)
      if (adapt[i].which == ATOM)
        error->all(FLERR,"Cannot use dynamic group with fix adapt atom");

  // setup and error checks

  anypair = 0;
  anybond = 0;

  for (int m = 0; m < nadapt; m++) {
    Adapt *ad = &adapt[m];

    ad->ivar = input->variable->find(ad->var);
    if (ad->ivar < 0)
      error->all(FLERR,"Variable name for fix adapt does not exist");
    if (!input->variable->equalstyle(ad->ivar))
      error->all(FLERR,"Variable for fix adapt is invalid style");

    if (ad->which == PAIR) {
      anypair = 1;
      ad->pair = nullptr;

      // if ad->pstyle has trailing sub-style annotation ":N",
      //   strip it for pstyle arg to pair_match() and set nsub = N
      // this should work for appended suffixes as well

      char *pstyle = utils::strdup(ad->pstyle);
      char *cptr;
      int nsub = 0;
      if ((cptr = strchr(pstyle,':'))) {
        *cptr = '\0';
        nsub = utils::inumeric(FLERR,cptr+1,false,lmp);
      }

      if (lmp->suffix_enable)
        ad->pair = force->pair_match(fmt::format("{}/{}",pstyle,lmp->suffix),1,nsub);

      if (ad->pair == nullptr) ad->pair = force->pair_match(pstyle,1,nsub);
      if (ad->pair == nullptr)
        error->all(FLERR,"Fix adapt pair style does not exist");

      void *ptr = ad->pair->extract(ad->pparam,ad->pdim);
      if (ptr == nullptr)
        error->all(FLERR,"Fix adapt pair style param not supported");

      // for pair styles only parameters that are 2-d arrays in atom types or
      // scalars are supported

      if (ad->pdim != 2 && ad->pdim != 0)
        error->all(FLERR,"Fix adapt pair style param is not compatible");

      if (ad->pdim == 2) ad->array = (double **) ptr;
      if (ad->pdim == 0) ad->scalar = (double *) ptr;

      // if pair hybrid, test that ilo,ihi,jlo,jhi are valid for sub-style

      if (utils::strmatch(force->pair_style,"^hybrid")) {
        PairHybrid *pair = (PairHybrid *) force->pair;
        for (i = ad->ilo; i <= ad->ihi; i++)
          for (j = MAX(ad->jlo,i); j <= ad->jhi; j++)
            if (!pair->check_ijtype(i,j,pstyle))
              error->all(FLERR,"Fix adapt type pair range is not valid for "
                         "pair hybrid sub-style");
      }

      delete [] pstyle;
    } else if (ad->which == BOND) {
      ad->bond = nullptr;
      anybond = 1;

      char *bstyle = utils::strdup(ad->bstyle);
      if (lmp->suffix_enable)
        ad->bond = force->bond_match(fmt::format("{}/{}",bstyle,lmp->suffix));

      if (ad->bond == nullptr) ad->bond = force->bond_match(bstyle);
      if (ad->bond == nullptr )
        error->all(FLERR,"Fix adapt bond style does not exist");

      void *ptr = ad->bond->extract(ad->bparam,ad->bdim);

      if (ptr == nullptr)
        error->all(FLERR,"Fix adapt bond style param not supported");

      // for bond styles, use a vector

      if (ad->bdim == 1) ad->vector = (double *) ptr;

      if (utils::strmatch(force->bond_style,"^hybrid"))
        error->all(FLERR,"Fix adapt does not support bond_style hybrid");

      delete [] bstyle;

    } else if (ad->which == KSPACE) {
      if (force->kspace == nullptr)
        error->all(FLERR,"Fix adapt kspace style does not exist");
      kspace_scale = (double *) force->kspace->extract("scale");

    } else if (ad->which == ATOM) {
      if (ad->aparam == DIAMETER) {
        if (!atom->radius_flag)
          error->all(FLERR,"Fix adapt requires atom attribute diameter");
        if (!atom->rmass_flag)
          error->all(FLERR,"Fix adapt requires atom attribute mass");
        if (discflag && domain->dimension != 2)
          error->all(FLERR,"Fix adapt requires 2d simulation");
        if (!restart_reset) previous_diam_scale = 1.0;
      }
      if (ad->aparam == CHARGE) {
        if (!atom->q_flag)
          error->all(FLERR,"Fix adapt requires atom attribute charge");
        if (!restart_reset) previous_chg_scale = 1.0;
      }
    }
  }

  if (restart_reset) restart_reset = 0;

  // make copy of original pair/bond array values

  for (int m = 0; m < nadapt; m++) {
    Adapt *ad = &adapt[m];
    if (ad->which == PAIR && ad->pdim == 2) {
      for (i = ad->ilo; i <= ad->ihi; i++)
        for (j = MAX(ad->jlo,i); j <= ad->jhi; j++)
          ad->array_orig[i][j] = ad->array[i][j];
    } else if (ad->which == PAIR && ad->pdim == 0) {
      ad->scalar_orig = *ad->scalar;

    } else if (ad->which == BOND && ad->bdim == 1) {
      for (i = ad->ilo; i <= ad->ihi; ++i )
        ad->vector_orig[i] = ad->vector[i];
    }

  }

  // fixes that store initial per-atom values

  if (id_fix_diam) {
    int ifix = modify->find_fix(id_fix_diam);
    if (ifix < 0) error->all(FLERR,"Could not find fix adapt storage fix ID");
    fix_diam = (FixStore *) modify->fix[ifix];
  }
  if (id_fix_chg) {
    int ifix = modify->find_fix(id_fix_chg);
    if (ifix < 0) error->all(FLERR,"Could not find fix adapt storage fix ID");
    fix_chg = (FixStore *) modify->fix[ifix];
  }

  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixAdapt::setup_pre_force(int /*vflag*/)
{
  change_settings();
}

/* ---------------------------------------------------------------------- */

void FixAdapt::setup_pre_force_respa(int vflag, int ilevel)
{
  if (ilevel < nlevels_respa-1) return;
  setup_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAdapt::pre_force(int /*vflag*/)
{
  if (nevery == 0) return;
  if (update->ntimestep % nevery) return;
  change_settings();
}

/* ---------------------------------------------------------------------- */

void FixAdapt::pre_force_respa(int vflag, int ilevel, int)
{
  if (ilevel < nlevels_respa-1) return;
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAdapt::post_run()
{
  if (resetflag) restore_settings();
}

/* ----------------------------------------------------------------------
   change pair,kspace,atom parameters based on variable evaluation
------------------------------------------------------------------------- */

void FixAdapt::change_settings()
{
  int i,j;

  // variable evaluation may invoke computes so wrap with clear/add

  modify->clearstep_compute();

  for (int m = 0; m < nadapt; m++) {
    Adapt *ad = &adapt[m];
    double value = input->variable->compute_equal(ad->ivar);

    // set global scalar or type pair array values

    if (ad->which == PAIR) {
      if (ad->pdim == 0) {
        if (scaleflag) *ad->scalar = value * ad->scalar_orig;
        else *ad->scalar = value;
      } else if (ad->pdim == 2) {
        if (scaleflag)
          for (i = ad->ilo; i <= ad->ihi; i++)
            for (j = MAX(ad->jlo,i); j <= ad->jhi; j++)
              ad->array[i][j] = value*ad->array_orig[i][j];
        else
          for (i = ad->ilo; i <= ad->ihi; i++)
            for (j = MAX(ad->jlo,i); j <= ad->jhi; j++)
              ad->array[i][j] = value;
      }

    // set bond type array values:

    } else if (ad->which == BOND) {
      if (ad->bdim == 1) {
        if (scaleflag)
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value*ad->vector_orig[i];
        else
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value;
      }

    // set kspace scale factor

    } else if (ad->which == KSPACE) {
      *kspace_scale = value;

    // set per atom values, also make changes for ghost atoms

    } else if (ad->which == ATOM) {

      // reset radius to new value, for both owned and ghost atoms
      // also reset rmass to new value assuming density remains constant
      // for scaleflag, previous_diam_scale is the scale factor on previous step

      if (ad->aparam == DIAMETER) {
        double scale;
        double *radius = atom->radius;
        double *rmass = atom->rmass;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        int nall = nlocal + atom->nghost;

        if (scaleflag) scale = value / previous_diam_scale;

        for (i = 0; i < nall; i++) {
          if (mask[i] & groupbit) {
            if (massflag) {
              if (!scaleflag) scale = 0.5*value / radius[i];
              if (discflag) rmass[i] *= scale*scale;
              else rmass[i] *= scale*scale*scale;
            }
            if (scaleflag) radius[i] *= scale;
            else radius[i] = 0.5*value;
          }
        }

        if (scaleflag) previous_diam_scale = value;

      // reset charge to new value, for both owned and ghost atoms
      // for scaleflag, previous_chg_scale is the scale factor on previous step

      } else if (ad->aparam == CHARGE) {
        double scale;
        double *q = atom->q;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        int nall = nlocal + atom->nghost;

        if (scaleflag) scale = value / previous_chg_scale;

        for (i = 0; i < nall; i++) {
          if (mask[i] & groupbit) {
            if (scaleflag) q[i] *= scale;
            else q[i] = value;
          }
        }

        if (scaleflag) previous_chg_scale = value;
      }
    }
  }

  modify->addstep_compute(update->ntimestep + nevery);

  // re-initialize pair styles if any PAIR settings were changed
  // ditto for bond styles if any BOND settings were changed
  // this resets other coeffs that may depend on changed values,
  //   and also offset and tail corrections
  // we must call force->pair->reinit() instead of the individual
  // adapted pair styles so that also the top-level
  // tail correction values are updated for hybrid pair styles.
  //  same for bond styles

  if (anypair) force->pair->reinit();
  if (anybond) force->bond->reinit();

  // reset KSpace charges if charges have changed

  if (chgflag && force->kspace) force->kspace->qsum_qsq();
}

/* ----------------------------------------------------------------------
   restore pair,kspace,atom parameters to original values
------------------------------------------------------------------------- */

void FixAdapt::restore_settings()
{
  for (int m = 0; m < nadapt; m++) {
    Adapt *ad = &adapt[m];
    if (ad->which == PAIR) {
      if (ad->pdim == 0) *ad->scalar = ad->scalar_orig;
      else if (ad->pdim == 2) {
        for (int i = ad->ilo; i <= ad->ihi; i++)
          for (int j = MAX(ad->jlo,i); j <= ad->jhi; j++)
            ad->array[i][j] = ad->array_orig[i][j];
      }

    } else if (ad->which == BOND) {
      if (ad->pdim == 1) {
        for (int i = ad->ilo; i <= ad->ihi; i++)
          ad->vector[i] = ad->vector_orig[i];
      }

    } else if (ad->which == KSPACE) {
      *kspace_scale = 1.0;

    } else if (ad->which == ATOM) {
      if (diamflag) {
        double scale;

        double *vec = fix_diam->vstore;
        double *radius = atom->radius;
        double *rmass = atom->rmass;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        if (scaleflag) scale = previous_diam_scale;

        for (int i = 0; i < nlocal; i++)
          if (mask[i] & groupbit) {
            if (massflag) {
              if (!scaleflag) scale = vec[i] / radius[i];
              if (discflag) rmass[i] *= scale*scale;
              else rmass[i] *= scale*scale*scale;
            }
            radius[i] = vec[i];
          }
      }
      if (chgflag) {
        double *vec = fix_chg->vstore;
        double *q = atom->q;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        for (int i = 0; i < nlocal; i++)
          if (mask[i] & groupbit) q[i] = vec[i];
      }
    }
  }

  if (anypair) force->pair->reinit();
  if (anybond) force->bond->reinit();
  if (chgflag && force->kspace) force->kspace->qsum_qsq();
}

/* ----------------------------------------------------------------------
   initialize one atom's storage values, called when atom is created
------------------------------------------------------------------------- */

void FixAdapt::set_arrays(int i)
{
  if (fix_diam) fix_diam->vstore[i] = atom->radius[i];
  if (fix_chg) fix_chg->vstore[i] = atom->q[i];
}

/* ----------------------------------------------------------------------
   write scale factors for diameter and charge to restart file
------------------------------------------------------------------------- */

void FixAdapt::write_restart(FILE *fp)
{
  int size = 2*sizeof(double);

  fwrite(&size,sizeof(int),1,fp);
  fwrite(&previous_diam_scale,sizeof(double),1,fp);
  fwrite(&previous_chg_scale,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
   use scale factors from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixAdapt::restart(char *buf)
{
  double *dbuf = (double *) buf;

  previous_diam_scale = dbuf[0];
  previous_chg_scale = dbuf[1];
}
