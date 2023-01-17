// clang-format off
/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, hmaktulga@lbl.gov
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
  Ananth Y Grama, ayg@cs.purdue.edu

  Please cite the related publication:
  H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama,
  "Parallel Reactive Molecular Dynamics: Numerical Methods and
  Algorithmic Techniques", Parallel Computing, in press.

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details:
  <https://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

#include "reaxff_api.h"

#include "pair.h"

#include <cmath>

namespace ReaxFF {
  void Compute_Polarization_Energy(reax_system *system, simulation_data *data)
  {
    int  i, type_i;
    double q, en_tmp;

    data->my_en.e_pol = 0.0;
    for (i = 0; i < system->n; i++) {
      type_i = system->my_atoms[i].type;
      if (type_i < 0) continue;
      q = system->my_atoms[i].q;

      en_tmp = KCALpMOL_to_EV * (system->reax_param.sbp[type_i].chi * q +
                                 (system->reax_param.sbp[type_i].eta / 2.) * SQR(q));
      data->my_en.e_pol += en_tmp;

      /* tally energy into global or per-atom energy accumulators */
      if (system->pair_ptr->eflag_either)
        system->pair_ptr->ev_tally(i,i,system->n,1,0.0,en_tmp,0.0,0.0,0.0,0.0);
    }
  }

  void vdW_Coulomb_Energy(reax_system *system, control_params *control,
                          simulation_data *data, storage *workspace,
                          reax_list **lists)
  {
    int i, j, pj, natoms;
    int start_i, end_i, flag;
    rc_tagint orig_i, orig_j;
    double p_vdW1, p_vdW1i;
    double powr_vdW1, powgi_vdW1;
    double tmp, r_ij, fn13, exp1, exp2;
    double Tap, dTap, dfn13, CEvd, CEclmb, de_core;
    double dr3gamij_1, dr3gamij_3;
    double e_ele, e_vdW, e_core, SMALL = 0.0001;
    double e_lg, de_lg, r_ij5, r_ij6, re6;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_list *far_nbrs;

    // Tallying variables:
    double pe_vdw, f_tmp, delij[3];

    natoms = system->n;
    far_nbrs = (*lists) + FAR_NBRS;
    p_vdW1 = system->reax_param.gp.l[28];
    p_vdW1i = 1.0 / p_vdW1;
    e_core = 0;
    e_vdW = 0;
    e_lg = de_lg = 0.0;

    for (i = 0; i < natoms; ++i) {
      if (system->my_atoms[i].type < 0) continue;
      start_i = Start_Index(i, far_nbrs);
      end_i   = End_Index(i, far_nbrs);
      orig_i  = system->my_atoms[i].orig_id;

      for (pj = start_i; pj < end_i; ++pj) {
        nbr_pj = &(far_nbrs->select.far_nbr_list[pj]);
        j = nbr_pj->nbr;
        if (system->my_atoms[j].type < 0) continue;
        orig_j  = system->my_atoms[j].orig_id;

        flag = 0;
        if (nbr_pj->d <= control->nonb_cut) {
          if (j < natoms) flag = 1;
          else if (orig_i < orig_j) flag = 1;
          else if (orig_i == orig_j) {
            if (nbr_pj->dvec[2] > SMALL) flag = 1;
            else if (fabs(nbr_pj->dvec[2]) < SMALL) {
              if (nbr_pj->dvec[1] > SMALL) flag = 1;
              else if (fabs(nbr_pj->dvec[1]) < SMALL && nbr_pj->dvec[0] > SMALL)
                flag = 1;
            }
          }
        }

        if (flag) {

          r_ij = nbr_pj->d;
          twbp = &(system->reax_param.tbp[system->my_atoms[i].type]
                   [system->my_atoms[j].type]);

          Tap = workspace->Tap[7] * r_ij + workspace->Tap[6];
          Tap = Tap * r_ij + workspace->Tap[5];
          Tap = Tap * r_ij + workspace->Tap[4];
          Tap = Tap * r_ij + workspace->Tap[3];
          Tap = Tap * r_ij + workspace->Tap[2];
          Tap = Tap * r_ij + workspace->Tap[1];
          Tap = Tap * r_ij + workspace->Tap[0];

          dTap = 7*workspace->Tap[7] * r_ij + 6*workspace->Tap[6];
          dTap = dTap * r_ij + 5*workspace->Tap[5];
          dTap = dTap * r_ij + 4*workspace->Tap[4];
          dTap = dTap * r_ij + 3*workspace->Tap[3];
          dTap = dTap * r_ij + 2*workspace->Tap[2];
          dTap += workspace->Tap[1]/r_ij;

          /*vdWaals Calculations*/
          if (system->reax_param.gp.vdw_type==1 || system->reax_param.gp.vdw_type==3)
            { // shielding
              powr_vdW1 = pow(r_ij, p_vdW1);
              powgi_vdW1 = pow(1.0 / twbp->gamma_w, p_vdW1);

              fn13 = pow(powr_vdW1 + powgi_vdW1, p_vdW1i);
              exp1 = exp(twbp->alpha * (1.0 - fn13 / twbp->r_vdW));
              exp2 = exp(0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW));

              e_vdW = twbp->D * (exp1 - 2.0 * exp2);
              data->my_en.e_vdW += Tap * e_vdW;

              dfn13 = pow(powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0) *
                pow(r_ij, p_vdW1 - 2.0);

              CEvd = dTap * e_vdW -
                Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) * dfn13;
            } else { // no shielding
            exp1 = exp(twbp->alpha * (1.0 - r_ij / twbp->r_vdW));
            exp2 = exp(0.5 * twbp->alpha * (1.0 - r_ij / twbp->r_vdW));

            e_vdW = twbp->D * (exp1 - 2.0 * exp2);
            data->my_en.e_vdW += Tap * e_vdW;

            CEvd = dTap * e_vdW -
              Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) / r_ij;
          }

          if (system->reax_param.gp.vdw_type==2 || system->reax_param.gp.vdw_type==3)
            { // inner wall
              e_core = twbp->ecore * exp(twbp->acore * (1.0-(r_ij/twbp->rcore)));
              data->my_en.e_vdW += Tap * e_core;

              de_core = -(twbp->acore/twbp->rcore) * e_core;
              CEvd += dTap * e_core + Tap * de_core / r_ij;

              //  lg correction, only if lgvdw is yes
              if (control->lgflag) {
                r_ij5 = pow(r_ij, 5.0);
                r_ij6 = pow(r_ij, 6.0);
                re6 = pow(twbp->lgre, 6.0);
                e_lg = -(twbp->lgcij/(r_ij6 + re6));
                data->my_en.e_vdW += Tap * e_lg;

                de_lg = -6.0 * e_lg *  r_ij5 / (r_ij6 + re6) ;
                CEvd += dTap * e_lg + Tap * de_lg / r_ij;
              }

            }

          /*Coulomb Calculations*/
          dr3gamij_1 = (r_ij * r_ij * r_ij + twbp->gamma);
          dr3gamij_3 = pow(dr3gamij_1 , 0.33333333333333);

          tmp = Tap / dr3gamij_3;
          data->my_en.e_ele += e_ele =
            C_ele * system->my_atoms[i].q * system->my_atoms[j].q * tmp;

          CEclmb = C_ele * system->my_atoms[i].q * system->my_atoms[j].q *
            (dTap -  Tap * r_ij / dr3gamij_1) / dr3gamij_3;

          /* tally into per-atom energy */
          if (system->pair_ptr->evflag) {
            pe_vdw = Tap * (e_vdW + e_core + e_lg);
            rvec_ScaledSum(delij, 1., system->my_atoms[i].x,
                            -1., system->my_atoms[j].x);
            f_tmp = -(CEvd + CEclmb);
            system->pair_ptr->ev_tally(i,j,natoms,1,pe_vdw,e_ele,
                                       f_tmp,delij[0],delij[1],delij[2]);
          }

          rvec_ScaledAdd(workspace->f[i], -(CEvd + CEclmb), nbr_pj->dvec);
          rvec_ScaledAdd(workspace->f[j], +(CEvd + CEclmb), nbr_pj->dvec);
        }
      }
    }

    Compute_Polarization_Energy(system, data);
  }

  void Tabulated_vdW_Coulomb_Energy(reax_system *system, control_params *control,
                                     simulation_data *data, storage *workspace,
                                     reax_list **lists)
  {
    int i, j, pj, r, natoms;
    int type_i, type_j, tmin, tmax;
    int start_i, end_i, flag;
    rc_tagint orig_i, orig_j;
    double r_ij, base, dif;
    double e_vdW, e_ele;
    double CEvd, CEclmb, SMALL = 0.0001;
    double f_tmp, delij[3];

    far_neighbor_data *nbr_pj;
    reax_list *far_nbrs;
    LR_lookup_table *t;
    LR_lookup_table ** & LR = system->LR;

    natoms = system->n;
    far_nbrs = (*lists) + FAR_NBRS;

    e_ele = e_vdW = 0;

    for (i = 0; i < natoms; ++i) {
      type_i  = system->my_atoms[i].type;
      if (type_i < 0) continue;
      start_i = Start_Index(i,far_nbrs);
      end_i   = End_Index(i,far_nbrs);
      orig_i  = system->my_atoms[i].orig_id;

      for (pj = start_i; pj < end_i; ++pj) {
        nbr_pj = &(far_nbrs->select.far_nbr_list[pj]);
        j = nbr_pj->nbr;
        type_j = system->my_atoms[j].type;
        if (type_j < 0) continue;
        orig_j  = system->my_atoms[j].orig_id;

        flag = 0;
        if (nbr_pj->d <= control->nonb_cut) {
          if (j < natoms) flag = 1;
          else if (orig_i < orig_j) flag = 1;
          else if (orig_i == orig_j) {
            if (nbr_pj->dvec[2] > SMALL) flag = 1;
            else if (fabs(nbr_pj->dvec[2]) < SMALL) {
              if (nbr_pj->dvec[1] > SMALL) flag = 1;
              else if (fabs(nbr_pj->dvec[1]) < SMALL && nbr_pj->dvec[0] > SMALL)
                flag = 1;
            }
          }
        }

        if (flag) {

          r_ij   = nbr_pj->d;
          tmin  = MIN(type_i, type_j);
          tmax  = MAX(type_i, type_j);
          t = &(LR[tmin][tmax]);

          /* Cubic Spline Interpolation */
          r = (int)(r_ij * t->inv_dx);
          if (r == 0)  ++r;
          base = (double)(r+1) * t->dx;
          dif = r_ij - base;

          e_vdW = ((t->vdW[r].d*dif + t->vdW[r].c)*dif + t->vdW[r].b)*dif +
            t->vdW[r].a;

          e_ele = ((t->ele[r].d*dif + t->ele[r].c)*dif + t->ele[r].b)*dif +
            t->ele[r].a;
          e_ele *= system->my_atoms[i].q * system->my_atoms[j].q;

          data->my_en.e_vdW += e_vdW;
          data->my_en.e_ele += e_ele;

          CEvd = ((t->CEvd[r].d*dif + t->CEvd[r].c)*dif + t->CEvd[r].b)*dif +
            t->CEvd[r].a;

          CEclmb = ((t->CEclmb[r].d*dif+t->CEclmb[r].c)*dif+t->CEclmb[r].b)*dif +
            t->CEclmb[r].a;
          CEclmb *= system->my_atoms[i].q * system->my_atoms[j].q;

          /* tally into per-atom energy */
          if (system->pair_ptr->evflag) {
            rvec_ScaledSum(delij, 1., system->my_atoms[i].x,
                            -1., system->my_atoms[j].x);
            f_tmp = -(CEvd + CEclmb);
            system->pair_ptr->ev_tally(i,j,natoms,1,e_vdW,e_ele,
                                       f_tmp,delij[0],delij[1],delij[2]);
          }

          rvec_ScaledAdd(workspace->f[i], -(CEvd + CEclmb), nbr_pj->dvec);
          rvec_ScaledAdd(workspace->f[j], +(CEvd + CEclmb), nbr_pj->dvec);
        }
      }
    }

    Compute_Polarization_Energy(system, data);
  }

  void LR_vdW_Coulomb(reax_system *system, storage *workspace,
                      control_params *control, int i, int j,
                      double r_ij, LR_data *lr)
  {
    double p_vdW1 = system->reax_param.gp.l[28];
    double p_vdW1i = 1.0 / p_vdW1;
    double powr_vdW1, powgi_vdW1;
    double tmp, fn13, exp1, exp2;
    double Tap, dTap, dfn13;
    double dr3gamij_1, dr3gamij_3;
    double e_core, de_core;
    double e_lg, de_lg, r_ij5, r_ij6, re6;
    two_body_parameters *twbp;

    twbp = &(system->reax_param.tbp[i][j]);
    e_core = 0;
    de_core = 0;
    e_lg = de_lg = 0.0;

    /* calculate taper and its derivative */
    Tap = workspace->Tap[7] * r_ij + workspace->Tap[6];
    Tap = Tap * r_ij + workspace->Tap[5];
    Tap = Tap * r_ij + workspace->Tap[4];
    Tap = Tap * r_ij + workspace->Tap[3];
    Tap = Tap * r_ij + workspace->Tap[2];
    Tap = Tap * r_ij + workspace->Tap[1];
    Tap = Tap * r_ij + workspace->Tap[0];

    dTap = 7*workspace->Tap[7] * r_ij + 6*workspace->Tap[6];
    dTap = dTap * r_ij + 5*workspace->Tap[5];
    dTap = dTap * r_ij + 4*workspace->Tap[4];
    dTap = dTap * r_ij + 3*workspace->Tap[3];
    dTap = dTap * r_ij + 2*workspace->Tap[2];
    dTap += workspace->Tap[1]/r_ij;

    /*vdWaals Calculations*/
    if (system->reax_param.gp.vdw_type==1 || system->reax_param.gp.vdw_type==3)
      { // shielding
        powr_vdW1 = pow(r_ij, p_vdW1);
        powgi_vdW1 = pow(1.0 / twbp->gamma_w, p_vdW1);

        fn13 = pow(powr_vdW1 + powgi_vdW1, p_vdW1i);
        exp1 = exp(twbp->alpha * (1.0 - fn13 / twbp->r_vdW));
        exp2 = exp(0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW));

        lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);

        dfn13 = pow(powr_vdW1 + powgi_vdW1, p_vdW1i-1.0) * pow(r_ij, p_vdW1-2.0);

        lr->CEvd = dTap * twbp->D * (exp1 - 2.0 * exp2) -
          Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) * dfn13;
      }
    else { // no shielding
      exp1 = exp(twbp->alpha * (1.0 - r_ij / twbp->r_vdW));
      exp2 = exp(0.5 * twbp->alpha * (1.0 - r_ij / twbp->r_vdW));

      lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);
      lr->CEvd = dTap * twbp->D * (exp1 - 2.0 * exp2) -
        Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) / r_ij;
    }

    if (system->reax_param.gp.vdw_type==2 || system->reax_param.gp.vdw_type==3)
      { // inner wall
        e_core = twbp->ecore * exp(twbp->acore * (1.0-(r_ij/twbp->rcore)));
        lr->e_vdW += Tap * e_core;

        de_core = -(twbp->acore/twbp->rcore) * e_core;
        lr->CEvd += dTap * e_core + Tap * de_core / r_ij;

        //  lg correction, only if lgvdw is yes
        if (control->lgflag) {
          r_ij5 = pow(r_ij, 5.0);
          r_ij6 = pow(r_ij, 6.0);
          re6 = pow(twbp->lgre, 6.0);
          e_lg = -(twbp->lgcij/(r_ij6 + re6));
          lr->e_vdW += Tap * e_lg;

          de_lg = -6.0 * e_lg *  r_ij5 / (r_ij6 + re6) ;
          lr->CEvd += dTap * e_lg + Tap * de_lg/r_ij;
        }

      }


    /* Coulomb calculations */
    dr3gamij_1 = (r_ij * r_ij * r_ij + twbp->gamma);
    dr3gamij_3 = pow(dr3gamij_1 , 0.33333333333333);

    tmp = Tap / dr3gamij_3;
    lr->H = EV_to_KCALpMOL * tmp;
    lr->e_ele = C_ele * tmp;

    lr->CEclmb = C_ele * (dTap -  Tap * r_ij / dr3gamij_1) / dr3gamij_3;
  }
}

