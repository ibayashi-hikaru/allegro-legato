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
   Contributing author: German Samolyuk (ORNL)
   Based on PairTersoffZBL by Aidan Thompson (SNL) and David Farrell (NWU)
------------------------------------------------------------------------- */

#include "pair_gw_zbl.h"

#include "comm.h"
#include "error.h"
#include "math_const.h"
#include "memory.h"
#include "potential_file_reader.h"
#include "tokenizer.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairGWZBL::PairGWZBL(LAMMPS *lmp) : PairGW(lmp)
{
  // hard-wired constants in metal or real units
  // a0 = Bohr radius
  // epsilon0 = permittivity of vacuum = q / energy-distance units
  // e = unit charge
  // 1 Kcal/mole = 0.043365121 eV

  if (strcmp(update->unit_style,"metal") == 0) {
    global_a_0 = 0.529;
    global_epsilon_0 = 0.00552635;
    global_e = 1.0;
  } else if (strcmp(update->unit_style,"real") == 0) {
    global_a_0 = 0.529;
    global_epsilon_0 = 0.00552635 * 0.043365121;
    global_e = 1.0;
  } else error->all(FLERR,"Pair gw/zbl requires metal or real units");
}

/* ---------------------------------------------------------------------- */

void PairGWZBL::read_file(char *file)
{
  memory->sfree(params);
  params = nullptr;
  nparams = maxparam = 0;

  // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, file, "gw/zbl", unit_convert_flag);
    char * line;

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    double conversion_factor = utils::get_conversion_factor(utils::ENERGY,
                                                            unit_convert);
    while ((line = reader.next_line(NPARAMS_PER_LINE))) {
      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();
        std::string jname = values.next_string();
        std::string kname = values.next_string();

        // ielement,jelement,kelement = 1st args
        // if all 3 args are in element list, then parse this line
        // else skip to next entry in file
        int ielement, jelement, kelement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements) continue;
        for (jelement = 0; jelement < nelements; jelement++)
          if (jname == elements[jelement]) break;
        if (jelement == nelements) continue;
        for (kelement = 0; kelement < nelements; kelement++)
          if (kname == elements[kelement]) break;
        if (kelement == nelements) continue;

        // load up parameter settings and error check their values

        if (nparams == maxparam) {
          maxparam += DELTA;
          params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
                                              "pair:params");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(params + nparams, 0, DELTA*sizeof(Param));
        }

        params[nparams].ielement = ielement;
        params[nparams].jelement = jelement;
        params[nparams].kelement = kelement;
        params[nparams].powerm      = values.next_double();
        params[nparams].gamma       = values.next_double();
        params[nparams].lam3        = values.next_double();
        params[nparams].c           = values.next_double();
        params[nparams].d           = values.next_double();
        params[nparams].h           = values.next_double();
        params[nparams].powern      = values.next_double();
        params[nparams].beta        = values.next_double();
        params[nparams].lam2        = values.next_double();
        params[nparams].bigb        = values.next_double();
        params[nparams].bigr        = values.next_double();
        params[nparams].bigd        = values.next_double();
        params[nparams].lam1        = values.next_double();
        params[nparams].biga        = values.next_double();
        params[nparams].Z_i         = values.next_double();
        params[nparams].Z_j         = values.next_double();
        params[nparams].ZBLcut      = values.next_double();
        params[nparams].ZBLexpscale = values.next_double();
        params[nparams].powermint = int(params[nparams].powerm);

        if (unit_convert) {
          params[nparams].biga *= conversion_factor;
          params[nparams].bigb *= conversion_factor;
        }
      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      // currently only allow m exponent of 1 or 3
      if (
          params[nparams].lam3 < 0.0 || params[nparams].c < 0.0 ||
          params[nparams].d < 0.0 || params[nparams].powern < 0.0 ||
          params[nparams].beta < 0.0 || params[nparams].lam2 < 0.0 ||
          params[nparams].bigb < 0.0 || params[nparams].bigr < 0.0 ||
          params[nparams].bigd < 0.0 ||
          params[nparams].bigd > params[nparams].bigr ||
          params[nparams].lam3 < 0.0 || params[nparams].biga < 0.0 ||
          params[nparams].powerm - params[nparams].powermint != 0.0 ||
          (params[nparams].powermint != 3 && params[nparams].powermint != 1) ||
          params[nparams].gamma < 0.0 ||
          params[nparams].Z_i < 1.0 || params[nparams].Z_j < 1.0 ||
          params[nparams].ZBLcut < 0.0 || params[nparams].ZBLexpscale < 0.0)
        error->one(FLERR,"Illegal GW parameter");

      nparams++;
    }
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    params = (Param *) memory->srealloc(params,maxparam*sizeof(Param), "pair:params");
  }

  MPI_Bcast(params, maxparam*sizeof(Param), MPI_BYTE, 0, world);
}

/* ---------------------------------------------------------------------- */

void PairGWZBL::repulsive(Param *param, double rsq, double &fforce,
                               int eflag, double &eng)
{
  double r,tmp_fc,tmp_fc_d,tmp_exp;

  // GW repulsive portion

  r = sqrt(rsq);
  tmp_fc = gw_fc(r,param);
  tmp_fc_d = gw_fc_d(r,param);
  tmp_exp = exp(-param->lam1 * r);
  double fforce_gw = param->biga * tmp_exp * (tmp_fc_d - tmp_fc*param->lam1);
  double eng_gw = tmp_fc * param->biga * tmp_exp;

  // ZBL repulsive portion

  double esq = pow(global_e,2.0);
  double a_ij = (0.8854*global_a_0) /
    (pow(param->Z_i,0.23) + pow(param->Z_j,0.23));
  double premult = (param->Z_i * param->Z_j * esq)/(4.0*MY_PI*global_epsilon_0);
  double r_ov_a = r/a_ij;
  double phi = 0.1818*exp(-3.2*r_ov_a) + 0.5099*exp(-0.9423*r_ov_a) +
    0.2802*exp(-0.4029*r_ov_a) + 0.02817*exp(-0.2016*r_ov_a);
  double dphi = (1.0/a_ij) * (-3.2*0.1818*exp(-3.2*r_ov_a) -
                              0.9423*0.5099*exp(-0.9423*r_ov_a) -
                              0.4029*0.2802*exp(-0.4029*r_ov_a) -
                              0.2016*0.02817*exp(-0.2016*r_ov_a));
  double fforce_ZBL = premult*-phi/rsq + premult*dphi/r;
  double eng_ZBL = premult*(1.0/r)*phi;

  // combine two parts with smoothing by Fermi-like function

  fforce = -(-F_fermi_d(r,param) * eng_ZBL +
             (1.0 - F_fermi(r,param))*fforce_ZBL +
             F_fermi_d(r,param)*eng_gw + F_fermi(r,param)*fforce_gw) / r;

  if (eflag)
    eng = (1.0 - F_fermi(r,param))*eng_ZBL + F_fermi(r,param)*eng_gw;
}

/* ---------------------------------------------------------------------- */

double PairGWZBL::gw_fa(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return -param->bigb * exp(-param->lam2 * r) * gw_fc(r,param) *
    F_fermi(r,param);
}

/* ---------------------------------------------------------------------- */

double PairGWZBL::gw_fa_d(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return param->bigb * exp(-param->lam2 * r) *
    (param->lam2 * gw_fc(r,param) * F_fermi(r,param) -
     gw_fc_d(r,param) * F_fermi(r,param) - gw_fc(r,param) *
     F_fermi_d(r,param));
}

/* ----------------------------------------------------------------------
   Fermi-like smoothing function
------------------------------------------------------------------------- */

double PairGWZBL::F_fermi(double r, Param *param)
{
  return 1.0 / (1.0 + exp(-param->ZBLexpscale*(r-param->ZBLcut)));
}

/* ----------------------------------------------------------------------
   Fermi-like smoothing function derivative with respect to r
------------------------------------------------------------------------- */

double PairGWZBL::F_fermi_d(double r, Param *param)
{
  return param->ZBLexpscale*exp(-param->ZBLexpscale*(r-param->ZBLcut)) /
    pow(1.0 + exp(-param->ZBLexpscale*(r-param->ZBLcut)),2.0);
}
