/* -*- c++ -*- ----------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the MACHDYN package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * ----------------------------------------------------------------------- */

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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(smd/ulsph/effm,ComputeSMD_Ulsph_Effm);
// clang-format on
#else

#ifndef LMP_COMPUTE_SMD_ULSPH_EFFM_H
#define LMP_COMPUTE_SMD_ULSPH_EFFM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeSMD_Ulsph_Effm : public Compute {
 public:
  ComputeSMD_Ulsph_Effm(class LAMMPS *, int, char **);
  ~ComputeSMD_Ulsph_Effm();
  void init();
  void compute_peratom();
  double memory_usage();

 private:
  int nmax;
  double *dt_vector;
};

}    // namespace LAMMPS_NS

#endif
#endif
