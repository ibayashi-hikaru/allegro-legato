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

#include "atom_vec_full_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm_kokkos.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "memory_kokkos.h"
#include "modify.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecFullKokkos::AtomVecFullKokkos(LAMMPS *lmp) : AtomVecKokkos(lmp)
{
  molecular = Atom::MOLECULAR;
  bonds_allow = angles_allow = dihedrals_allow = impropers_allow = 1;
  mass_type = PER_TYPE;

  comm_x_only = comm_f_only = 1;
  size_forward = 3;
  size_reverse = 3;
  size_border = 8;
  size_velocity = 3;
  size_data_atom = 7;
  size_data_vel = 4;
  xcol_data = 5;

  atom->molecule_flag = atom->q_flag = 1;

  k_count = DAT::tdual_int_1d("atom::k_count",1);
  atomKK = (AtomKokkos *) atom;
  commKK = (CommKokkos *) comm;
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by DELTA
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVecFullKokkos::grow(int n)
{
  auto DELTA = LMP_KOKKOS_AV_DELTA;
  int step = MAX(DELTA,nmax*0.01);
  if (n == 0) nmax += step;
  else nmax = n;
  atomKK->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR,"Per-processor system is too big");

  atomKK->sync(Device,ALL_MASK);
  atomKK->modified(Device,ALL_MASK);

  memoryKK->grow_kokkos(atomKK->k_tag,atomKK->tag,nmax,"atom:tag");
  memoryKK->grow_kokkos(atomKK->k_type,atomKK->type,nmax,"atom:type");
  memoryKK->grow_kokkos(atomKK->k_mask,atomKK->mask,nmax,"atom:mask");
  memoryKK->grow_kokkos(atomKK->k_image,atomKK->image,nmax,"atom:image");

  memoryKK->grow_kokkos(atomKK->k_x,atomKK->x,nmax,"atom:x");
  memoryKK->grow_kokkos(atomKK->k_v,atomKK->v,nmax,"atom:v");
  memoryKK->grow_kokkos(atomKK->k_f,atomKK->f,nmax,"atom:f");

  memoryKK->grow_kokkos(atomKK->k_q,atomKK->q,nmax,"atom:q");
  memoryKK->grow_kokkos(atomKK->k_molecule,atomKK->molecule,nmax,"atom:molecule");

  memoryKK->grow_kokkos(atomKK->k_nspecial,atomKK->nspecial,nmax,3,"atom:nspecial");
  memoryKK->grow_kokkos(atomKK->k_special,atomKK->special,nmax,atomKK->maxspecial,
                      "atom:special");
  memoryKK->grow_kokkos(atomKK->k_num_bond,atomKK->num_bond,nmax,"atom:num_bond");
  memoryKK->grow_kokkos(atomKK->k_bond_type,atomKK->bond_type,nmax,atomKK->bond_per_atom,
                      "atom:bond_type");
  memoryKK->grow_kokkos(atomKK->k_bond_atom,atomKK->bond_atom,nmax,atomKK->bond_per_atom,
                      "atom:bond_atom");

  memoryKK->grow_kokkos(atomKK->k_num_angle,atomKK->num_angle,nmax,"atom:num_angle");
  memoryKK->grow_kokkos(atomKK->k_angle_type,atomKK->angle_type,nmax,atomKK->angle_per_atom,
                      "atom:angle_type");
  memoryKK->grow_kokkos(atomKK->k_angle_atom1,atomKK->angle_atom1,nmax,atomKK->angle_per_atom,
                      "atom:angle_atom1");
  memoryKK->grow_kokkos(atomKK->k_angle_atom2,atomKK->angle_atom2,nmax,atomKK->angle_per_atom,
                      "atom:angle_atom2");
  memoryKK->grow_kokkos(atomKK->k_angle_atom3,atomKK->angle_atom3,nmax,atomKK->angle_per_atom,
                      "atom:angle_atom3");

  memoryKK->grow_kokkos(atomKK->k_num_dihedral,atomKK->num_dihedral,nmax,"atom:num_dihedral");
  memoryKK->grow_kokkos(atomKK->k_dihedral_type,atomKK->dihedral_type,nmax,
                      atomKK->dihedral_per_atom,"atom:dihedral_type");
  memoryKK->grow_kokkos(atomKK->k_dihedral_atom1,atomKK->dihedral_atom1,nmax,
                      atomKK->dihedral_per_atom,"atom:dihedral_atom1");
  memoryKK->grow_kokkos(atomKK->k_dihedral_atom2,atomKK->dihedral_atom2,nmax,
                      atomKK->dihedral_per_atom,"atom:dihedral_atom2");
  memoryKK->grow_kokkos(atomKK->k_dihedral_atom3,atomKK->dihedral_atom3,nmax,
                      atomKK->dihedral_per_atom,"atom:dihedral_atom3");
  memoryKK->grow_kokkos(atomKK->k_dihedral_atom4,atomKK->dihedral_atom4,nmax,
                      atomKK->dihedral_per_atom,"atom:dihedral_atom4");

  memoryKK->grow_kokkos(atomKK->k_num_improper,atomKK->num_improper,nmax,"atom:num_improper");
  memoryKK->grow_kokkos(atomKK->k_improper_type,atomKK->improper_type,nmax,
                      atomKK->improper_per_atom,"atom:improper_type");
  memoryKK->grow_kokkos(atomKK->k_improper_atom1,atomKK->improper_atom1,nmax,
                      atomKK->improper_per_atom,"atom:improper_atom1");
  memoryKK->grow_kokkos(atomKK->k_improper_atom2,atomKK->improper_atom2,nmax,
                      atomKK->improper_per_atom,"atom:improper_atom2");
  memoryKK->grow_kokkos(atomKK->k_improper_atom3,atomKK->improper_atom3,nmax,
                      atomKK->improper_per_atom,"atom:improper_atom3");
  memoryKK->grow_kokkos(atomKK->k_improper_atom4,atomKK->improper_atom4,nmax,
                      atomKK->improper_per_atom,"atom:improper_atom4");

  grow_pointers();
  atomKK->sync(Host,ALL_MASK);

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
------------------------------------------------------------------------- */

void AtomVecFullKokkos::grow_pointers()
{
  tag = atomKK->tag;
  d_tag = atomKK->k_tag.d_view;
  h_tag = atomKK->k_tag.h_view;

  type = atomKK->type;
  d_type = atomKK->k_type.d_view;
  h_type = atomKK->k_type.h_view;
  mask = atomKK->mask;
  d_mask = atomKK->k_mask.d_view;
  h_mask = atomKK->k_mask.h_view;
  image = atomKK->image;
  d_image = atomKK->k_image.d_view;
  h_image = atomKK->k_image.h_view;

  x = atomKK->x;
  d_x = atomKK->k_x.d_view;
  h_x = atomKK->k_x.h_view;
  v = atomKK->v;
  d_v = atomKK->k_v.d_view;
  h_v = atomKK->k_v.h_view;
  f = atomKK->f;
  d_f = atomKK->k_f.d_view;
  h_f = atomKK->k_f.h_view;

  q = atomKK->q;
  d_q = atomKK->k_q.d_view;
  h_q = atomKK->k_q.h_view;

  molecule = atomKK->molecule;
  d_molecule = atomKK->k_molecule.d_view;
  h_molecule = atomKK->k_molecule.h_view;

  nspecial = atomKK->nspecial;
  d_nspecial = atomKK->k_nspecial.d_view;
  h_nspecial = atomKK->k_nspecial.h_view;
  special = atomKK->special;
  d_special = atomKK->k_special.d_view;
  h_special = atomKK->k_special.h_view;

  num_bond = atomKK->num_bond;
  d_num_bond = atomKK->k_num_bond.d_view;
  h_num_bond = atomKK->k_num_bond.h_view;
  bond_type = atomKK->bond_type;
  d_bond_type = atomKK->k_bond_type.d_view;
  h_bond_type = atomKK->k_bond_type.h_view;
  bond_atom = atomKK->bond_atom;
  d_bond_atom = atomKK->k_bond_atom.d_view;
  h_bond_atom = atomKK->k_bond_atom.h_view;

  num_angle = atomKK->num_angle;
  d_num_angle = atomKK->k_num_angle.d_view;
  h_num_angle = atomKK->k_num_angle.h_view;
  angle_type = atomKK->angle_type;
  d_angle_type = atomKK->k_angle_type.d_view;
  h_angle_type = atomKK->k_angle_type.h_view;
  angle_atom1 = atomKK->angle_atom1;
  d_angle_atom1 = atomKK->k_angle_atom1.d_view;
  h_angle_atom1 = atomKK->k_angle_atom1.h_view;
  angle_atom2 = atomKK->angle_atom2;
  d_angle_atom2 = atomKK->k_angle_atom2.d_view;
  h_angle_atom2 = atomKK->k_angle_atom2.h_view;
  angle_atom3 = atomKK->angle_atom3;
  d_angle_atom3 = atomKK->k_angle_atom3.d_view;
  h_angle_atom3 = atomKK->k_angle_atom3.h_view;

  num_dihedral = atomKK->num_dihedral;
  d_num_dihedral = atomKK->k_num_dihedral.d_view;
  h_num_dihedral = atomKK->k_num_dihedral.h_view;
  dihedral_type = atomKK->dihedral_type;
  d_dihedral_type = atomKK->k_dihedral_type.d_view;
  h_dihedral_type = atomKK->k_dihedral_type.h_view;
  dihedral_atom1 = atomKK->dihedral_atom1;
  d_dihedral_atom1 = atomKK->k_dihedral_atom1.d_view;
  h_dihedral_atom1 = atomKK->k_dihedral_atom1.h_view;
  dihedral_atom2 = atomKK->dihedral_atom2;
  d_dihedral_atom2 = atomKK->k_dihedral_atom2.d_view;
  h_dihedral_atom2 = atomKK->k_dihedral_atom2.h_view;
  dihedral_atom3 = atomKK->dihedral_atom3;
  d_dihedral_atom3 = atomKK->k_dihedral_atom3.d_view;
  h_dihedral_atom3 = atomKK->k_dihedral_atom3.h_view;
  dihedral_atom4 = atomKK->dihedral_atom4;
  d_dihedral_atom4 = atomKK->k_dihedral_atom4.d_view;
  h_dihedral_atom4 = atomKK->k_dihedral_atom4.h_view;

  num_improper = atomKK->num_improper;
  d_num_improper = atomKK->k_num_improper.d_view;
  h_num_improper = atomKK->k_num_improper.h_view;
  improper_type = atomKK->improper_type;
  d_improper_type = atomKK->k_improper_type.d_view;
  h_improper_type = atomKK->k_improper_type.h_view;
  improper_atom1 = atomKK->improper_atom1;
  d_improper_atom1 = atomKK->k_improper_atom1.d_view;
  h_improper_atom1 = atomKK->k_improper_atom1.h_view;
  improper_atom2 = atomKK->improper_atom2;
  d_improper_atom2 = atomKK->k_improper_atom2.d_view;
  h_improper_atom2 = atomKK->k_improper_atom2.h_view;
  improper_atom3 = atomKK->improper_atom3;
  d_improper_atom3 = atomKK->k_improper_atom3.d_view;
  h_improper_atom3 = atomKK->k_improper_atom3.h_view;
  improper_atom4 = atomKK->improper_atom4;
  d_improper_atom4 = atomKK->k_improper_atom4.d_view;
  h_improper_atom4 = atomKK->k_improper_atom4.h_view;
}

/* ----------------------------------------------------------------------
   copy atom I info to atom J
------------------------------------------------------------------------- */

void AtomVecFullKokkos::copy(int i, int j, int delflag)
{
  int k;

  h_tag[j] = h_tag[i];
  h_type[j] = h_type[i];
  mask[j] = mask[i];
  h_image[j] = h_image[i];
  h_x(j,0) = h_x(i,0);
  h_x(j,1) = h_x(i,1);
  h_x(j,2) = h_x(i,2);
  h_v(j,0) = h_v(i,0);
  h_v(j,1) = h_v(i,1);
  h_v(j,2) = h_v(i,2);

  h_q[j] = h_q[i];
  h_molecule(j) = h_molecule(i);

  h_num_bond(j) = h_num_bond(i);
  for (k = 0; k < h_num_bond(j); k++) {
    h_bond_type(j,k) = h_bond_type(i,k);
    h_bond_atom(j,k) = h_bond_atom(i,k);
  }

  h_nspecial(j,0) = h_nspecial(i,0);
  h_nspecial(j,1) = h_nspecial(i,1);
  h_nspecial(j,2) = h_nspecial(i,2);
  for (k = 0; k < h_nspecial(j,2); k++)
    h_special(j,k) = h_special(i,k);

  h_num_angle(j) = h_num_angle(i);
  for (k = 0; k < h_num_angle(j); k++) {
    h_angle_type(j,k) = h_angle_type(i,k);
    h_angle_atom1(j,k) = h_angle_atom1(i,k);
    h_angle_atom2(j,k) = h_angle_atom2(i,k);
    h_angle_atom3(j,k) = h_angle_atom3(i,k);
  }

  h_num_dihedral(j) = h_num_dihedral(i);
  for (k = 0; k < h_num_dihedral(j); k++) {
    h_dihedral_type(j,k) = h_dihedral_type(i,k);
    h_dihedral_atom1(j,k) = h_dihedral_atom1(i,k);
    h_dihedral_atom2(j,k) = h_dihedral_atom2(i,k);
    h_dihedral_atom3(j,k) = h_dihedral_atom3(i,k);
    h_dihedral_atom4(j,k) = h_dihedral_atom4(i,k);
  }

  h_num_improper(j) = h_num_improper(i);
  for (k = 0; k < h_num_improper(j); k++) {
    h_improper_type(j,k) = h_improper_type(i,k);
    h_improper_atom1(j,k) = h_improper_atom1(i,k);
    h_improper_atom2(j,k) = h_improper_atom2(i,k);
    h_improper_atom3(j,k) = h_improper_atom3(i,k);
    h_improper_atom4(j,k) = h_improper_atom4(i,k);
  }

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->copy_arrays(i,j,delflag);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG>
struct AtomVecFullKokkos_PackBorder {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_xfloat_2d _buf;
  const typename AT::t_int_2d_const _list;
  const int _iswap;
  const typename AT::t_x_array_randomread _x;
  const typename AT::t_tagint_1d _tag;
  const typename AT::t_int_1d _type;
  const typename AT::t_int_1d _mask;
  const typename AT::t_float_1d _q;
  const typename AT::t_tagint_1d _molecule;
  X_FLOAT _dx,_dy,_dz;

  AtomVecFullKokkos_PackBorder(
      const typename AT::t_xfloat_2d &buf,
      const typename AT::t_int_2d_const &list,
      const int & iswap,
      const typename AT::t_x_array &x,
      const typename AT::t_tagint_1d &tag,
      const typename AT::t_int_1d &type,
      const typename AT::t_int_1d &mask,
      const typename AT::t_float_1d &q,
      const typename AT::t_tagint_1d &molecule,
      const X_FLOAT &dx, const X_FLOAT &dy, const X_FLOAT &dz):
      _buf(buf),_list(list),_iswap(iswap),
      _x(x),_tag(tag),_type(type),_mask(mask),_q(q),_molecule(molecule),
      _dx(dx),_dy(dy),_dz(dz) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
      const int j = _list(_iswap,i);
      if (PBC_FLAG == 0) {
          _buf(i,0) = _x(j,0);
          _buf(i,1) = _x(j,1);
          _buf(i,2) = _x(j,2);
          _buf(i,3) = d_ubuf(_tag(j)).d;
          _buf(i,4) = d_ubuf(_type(j)).d;
          _buf(i,5) = d_ubuf(_mask(j)).d;
          _buf(i,6) = _q(j);
          _buf(i,7) = d_ubuf(_molecule(j)).d;
      } else {
          _buf(i,0) = _x(j,0) + _dx;
          _buf(i,1) = _x(j,1) + _dy;
          _buf(i,2) = _x(j,2) + _dz;
          _buf(i,3) = d_ubuf(_tag(j)).d;
          _buf(i,4) = d_ubuf(_type(j)).d;
          _buf(i,5) = d_ubuf(_mask(j)).d;
          _buf(i,6) = _q(j);
          _buf(i,7) = d_ubuf(_molecule(j)).d;
      }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecFullKokkos::pack_border_kokkos(int n, DAT::tdual_int_2d k_sendlist,
                                               DAT::tdual_xfloat_2d buf,int iswap,
                                               int pbc_flag, int *pbc, ExecutionSpace space)
{
  X_FLOAT dx,dy,dz;

  if (pbc_flag != 0) {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (space==Host) {
      AtomVecFullKokkos_PackBorder<LMPHostType,1> f(
        buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
        iswap,h_x,h_tag,h_type,h_mask,h_q,h_molecule,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecFullKokkos_PackBorder<LMPDeviceType,1> f(
        buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
        iswap,d_x,d_tag,d_type,d_mask,d_q,d_molecule,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    }

  } else {
    dx = dy = dz = 0;
    if (space==Host) {
      AtomVecFullKokkos_PackBorder<LMPHostType,0> f(
        buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
        iswap,h_x,h_tag,h_type,h_mask,h_q,h_molecule,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecFullKokkos_PackBorder<LMPDeviceType,0> f(
        buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
        iswap,d_x,d_tag,d_type,d_mask,d_q,d_molecule,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    }
  }
  return n*size_border;
}

/* ---------------------------------------------------------------------- */

int AtomVecFullKokkos::pack_border(int n, int *list, double *buf,
                                        int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = h_x(j,0);
      buf[m++] = h_x(j,1);
      buf[m++] = h_x(j,2);
      buf[m++] = ubuf(h_tag(j)).d;
      buf[m++] = ubuf(h_type(j)).d;
      buf[m++] = ubuf(h_mask(j)).d;
      buf[m++] = h_q(j);
      buf[m++] = ubuf(h_molecule(j)).d;
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = h_x(j,0) + dx;
      buf[m++] = h_x(j,1) + dy;
      buf[m++] = h_x(j,2) + dz;
      buf[m++] = ubuf(h_tag(j)).d;
      buf[m++] = ubuf(h_type(j)).d;
      buf[m++] = ubuf(h_mask(j)).d;
      buf[m++] = h_q(j);
      buf[m++] = ubuf(h_molecule(j)).d;
    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecFullKokkos::pack_border_vel(int n, int *list, double *buf,
                                            int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz,dvx,dvy,dvz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = h_x(j,0);
      buf[m++] = h_x(j,1);
      buf[m++] = h_x(j,2);
      buf[m++] = ubuf(h_tag(j)).d;
      buf[m++] = ubuf(h_type(j)).d;
      buf[m++] = ubuf(h_mask(j)).d;
      buf[m++] = h_q(j);
      buf[m++] = ubuf(h_molecule(j)).d;
      buf[m++] = h_v(j,0);
      buf[m++] = h_v(j,1);
      buf[m++] = h_v(j,2);
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = h_x(j,0) + dx;
        buf[m++] = h_x(j,1) + dy;
        buf[m++] = h_x(j,2) + dz;
        buf[m++] = ubuf(h_tag(j)).d;
        buf[m++] = ubuf(h_type(j)).d;
        buf[m++] = ubuf(h_mask(j)).d;
        buf[m++] = h_q(j);
        buf[m++] = ubuf(h_molecule(j)).d;
        buf[m++] = h_v(j,0);
        buf[m++] = h_v(j,1);
        buf[m++] = h_v(j,2);
      }
    } else {
      dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
      dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
      dvz = pbc[2]*h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = h_x(j,0) + dx;
        buf[m++] = h_x(j,1) + dy;
        buf[m++] = h_x(j,2) + dz;
        buf[m++] = ubuf(h_tag(j)).d;
        buf[m++] = ubuf(h_type(j)).d;
        buf[m++] = ubuf(h_mask(j)).d;
        buf[m++] = h_q(j);
        buf[m++] = ubuf(h_molecule(j)).d;
        if (mask[i] & deform_groupbit) {
          buf[m++] = h_v(j,0) + dvx;
          buf[m++] = h_v(j,1) + dvy;
          buf[m++] = h_v(j,2) + dvz;
        } else {
          buf[m++] = h_v(j,0);
          buf[m++] = h_v(j,1);
          buf[m++] = h_v(j,2);
        }
      }
    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecFullKokkos::pack_border_hybrid(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = h_q(j);
    buf[m++] = ubuf(h_molecule(j)).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecFullKokkos_UnpackBorder {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  const typename AT::t_xfloat_2d_const _buf;
  typename AT::t_x_array _x;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_float_1d _q;
  typename AT::t_tagint_1d _molecule;
  int _first;


  AtomVecFullKokkos_UnpackBorder(
      const typename AT::t_xfloat_2d_const &buf,
      typename AT::t_x_array &x,
      typename AT::t_tagint_1d &tag,
      typename AT::t_int_1d &type,
      typename AT::t_int_1d &mask,
      typename AT::t_float_1d &q,
      typename AT::t_tagint_1d &molecule,
      const int& first):
    _buf(buf),_x(x),_tag(tag),_type(type),_mask(mask),_q(q),_molecule(molecule),
    _first(first) {
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
      _x(i+_first,0) = _buf(i,0);
      _x(i+_first,1) = _buf(i,1);
      _x(i+_first,2) = _buf(i,2);
      _tag(i+_first) = (tagint) d_ubuf(_buf(i,3)).i;
      _type(i+_first) = (int) d_ubuf(_buf(i,4)).i;
      _mask(i+_first) = (int) d_ubuf(_buf(i,5)).i;
      _q(i+_first) = _buf(i,6);
      _molecule(i+_first) = (tagint) d_ubuf(_buf(i,7)).i;

  }
};

/* ---------------------------------------------------------------------- */

void AtomVecFullKokkos::unpack_border_kokkos(const int &n, const int &first,
                                                  const DAT::tdual_xfloat_2d &buf,
                                                  ExecutionSpace space) {
  atomKK->modified(space,X_MASK|TAG_MASK|TYPE_MASK|MASK_MASK|Q_MASK|MOLECULE_MASK);

  while (first+n >= nmax) grow(0);

  if (space==Host) {
    struct AtomVecFullKokkos_UnpackBorder<LMPHostType>
      f(buf.view<LMPHostType>(),h_x,h_tag,h_type,h_mask,h_q,h_molecule,first);
    Kokkos::parallel_for(n,f);
  } else {
    struct AtomVecFullKokkos_UnpackBorder<LMPDeviceType>
      f(buf.view<LMPDeviceType>(),d_x,d_tag,d_type,d_mask,d_q,d_molecule,first);
    Kokkos::parallel_for(n,f);
  }

  atomKK->modified(space,X_MASK|TAG_MASK|TYPE_MASK|MASK_MASK|Q_MASK|MOLECULE_MASK);
}

/* ---------------------------------------------------------------------- */

void AtomVecFullKokkos::unpack_border(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  while (last > nmax) grow(0);

  for (i = first; i < last; i++) {
    h_x(i,0) = buf[m++];
    h_x(i,1) = buf[m++];
    h_x(i,2) = buf[m++];
    h_tag(i) =  (tagint)  ubuf(buf[m++]).i;
    h_type(i) = (int) ubuf(buf[m++]).i;
    h_mask(i) = (int) ubuf(buf[m++]).i;
    h_q(i) = buf[m++];
    h_molecule(i) = (tagint) ubuf(buf[m++]).i;
  }

  atomKK->modified(Host,X_MASK|TAG_MASK|TYPE_MASK|MASK_MASK|Q_MASK|MOLECULE_MASK);

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ---------------------------------------------------------------------- */

void AtomVecFullKokkos::unpack_border_vel(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  while (last > nmax) grow(0);

  for (i = first; i < last; i++) {
    h_x(i,0) = buf[m++];
    h_x(i,1) = buf[m++];
    h_x(i,2) = buf[m++];
    h_tag(i) =  (tagint)  ubuf(buf[m++]).i;
    h_type(i) = (int) ubuf(buf[m++]).i;
    h_mask(i) = (int) ubuf(buf[m++]).i;
    h_q(i) = buf[m++];
    h_molecule(i) = (tagint) ubuf(buf[m++]).i;
    h_v(i,0) = buf[m++];
    h_v(i,1) = buf[m++];
    h_v(i,2) = buf[m++];
  }

  atomKK->modified(Host,X_MASK|TAG_MASK|TYPE_MASK|MASK_MASK|Q_MASK|MOLECULE_MASK|V_MASK);

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ---------------------------------------------------------------------- */

int AtomVecFullKokkos::unpack_border_hybrid(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    h_q(i) = buf[m++];
    h_molecule(i) = (tagint) ubuf(buf[m++]).i;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecFullKokkos_PackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_x_array_randomread _x;
  typename AT::t_v_array_randomread _v;
  typename AT::t_tagint_1d_randomread _tag;
  typename AT::t_int_1d_randomread _type;
  typename AT::t_int_1d_randomread _mask;
  typename AT::t_imageint_1d_randomread _image;
  typename AT::t_float_1d_randomread _q;
  typename AT::t_tagint_1d_randomread _molecule;
  typename AT::t_int_2d_randomread _nspecial;
  typename AT::t_tagint_2d_randomread _special;
  typename AT::t_int_1d_randomread _num_bond;
  typename AT::t_int_2d_randomread _bond_type;
  typename AT::t_tagint_2d_randomread _bond_atom;
  typename AT::t_int_1d_randomread _num_angle;
  typename AT::t_int_2d_randomread _angle_type;
  typename AT::t_tagint_2d_randomread _angle_atom1,_angle_atom2,_angle_atom3;
  typename AT::t_int_1d_randomread _num_dihedral;
  typename AT::t_int_2d_randomread _dihedral_type;
  typename AT::t_tagint_2d_randomread _dihedral_atom1,_dihedral_atom2,
    _dihedral_atom3,_dihedral_atom4;
  typename AT::t_int_1d_randomread _num_improper;
  typename AT::t_int_2d_randomread _improper_type;
  typename AT::t_tagint_2d_randomread _improper_atom1,_improper_atom2,
    _improper_atom3,_improper_atom4;
  typename AT::t_x_array _xw;
  typename AT::t_v_array _vw;
  typename AT::t_tagint_1d _tagw;
  typename AT::t_int_1d _typew;
  typename AT::t_int_1d _maskw;
  typename AT::t_imageint_1d _imagew;
  typename AT::t_float_1d _qw;
  typename AT::t_tagint_1d _moleculew;
  typename AT::t_int_2d _nspecialw;
  typename AT::t_tagint_2d _specialw;
  typename AT::t_int_1d _num_bondw;
  typename AT::t_int_2d _bond_typew;
  typename AT::t_tagint_2d _bond_atomw;
  typename AT::t_int_1d _num_anglew;
  typename AT::t_int_2d _angle_typew;
  typename AT::t_tagint_2d _angle_atom1w,_angle_atom2w,_angle_atom3w;
  typename AT::t_int_1d _num_dihedralw;
  typename AT::t_int_2d _dihedral_typew;
  typename AT::t_tagint_2d _dihedral_atom1w,_dihedral_atom2w,
    _dihedral_atom3w,_dihedral_atom4w;
  typename AT::t_int_1d _num_improperw;
  typename AT::t_int_2d _improper_typew;
  typename AT::t_tagint_2d _improper_atom1w,_improper_atom2w,
    _improper_atom3w,_improper_atom4w;
  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d_const _sendlist;
  typename AT::t_int_1d_const _copylist;
  int _nlocal,_dim;
  X_FLOAT _lo,_hi;
  size_t elements;

  AtomVecFullKokkos_PackExchangeFunctor(
      const AtomKokkos* atom,
      const typename AT::tdual_xfloat_2d buf,
      typename AT::tdual_int_1d sendlist,
      typename AT::tdual_int_1d copylist,int nlocal, int dim,
                X_FLOAT lo, X_FLOAT hi):
    _x(atom->k_x.view<DeviceType>()),
    _v(atom->k_v.view<DeviceType>()),
    _tag(atom->k_tag.view<DeviceType>()),
    _type(atom->k_type.view<DeviceType>()),
    _mask(atom->k_mask.view<DeviceType>()),
    _image(atom->k_image.view<DeviceType>()),
    _q(atom->k_q.view<DeviceType>()),
    _molecule(atom->k_molecule.view<DeviceType>()),
    _nspecial(atom->k_nspecial.view<DeviceType>()),
    _special(atom->k_special.view<DeviceType>()),
    _num_bond(atom->k_num_bond.view<DeviceType>()),
    _bond_type(atom->k_bond_type.view<DeviceType>()),
    _bond_atom(atom->k_bond_atom.view<DeviceType>()),
    _num_angle(atom->k_num_angle.view<DeviceType>()),
    _angle_type(atom->k_angle_type.view<DeviceType>()),
    _angle_atom1(atom->k_angle_atom1.view<DeviceType>()),
    _angle_atom2(atom->k_angle_atom2.view<DeviceType>()),
    _angle_atom3(atom->k_angle_atom3.view<DeviceType>()),
    _num_dihedral(atom->k_num_dihedral.view<DeviceType>()),
    _dihedral_type(atom->k_dihedral_type.view<DeviceType>()),
    _dihedral_atom1(atom->k_dihedral_atom1.view<DeviceType>()),
    _dihedral_atom2(atom->k_dihedral_atom2.view<DeviceType>()),
    _dihedral_atom3(atom->k_dihedral_atom3.view<DeviceType>()),
    _dihedral_atom4(atom->k_dihedral_atom4.view<DeviceType>()),
    _num_improper(atom->k_num_improper.view<DeviceType>()),
    _improper_type(atom->k_improper_type.view<DeviceType>()),
    _improper_atom1(atom->k_improper_atom1.view<DeviceType>()),
    _improper_atom2(atom->k_improper_atom2.view<DeviceType>()),
    _improper_atom3(atom->k_improper_atom3.view<DeviceType>()),
    _improper_atom4(atom->k_improper_atom4.view<DeviceType>()),
    _xw(atom->k_x.view<DeviceType>()),
    _vw(atom->k_v.view<DeviceType>()),
    _tagw(atom->k_tag.view<DeviceType>()),
    _typew(atom->k_type.view<DeviceType>()),
    _maskw(atom->k_mask.view<DeviceType>()),
    _imagew(atom->k_image.view<DeviceType>()),
    _qw(atom->k_q.view<DeviceType>()),
    _moleculew(atom->k_molecule.view<DeviceType>()),
    _nspecialw(atom->k_nspecial.view<DeviceType>()),
    _specialw(atom->k_special.view<DeviceType>()),
    _num_bondw(atom->k_num_bond.view<DeviceType>()),
    _bond_typew(atom->k_bond_type.view<DeviceType>()),
    _bond_atomw(atom->k_bond_atom.view<DeviceType>()),
    _num_anglew(atom->k_num_angle.view<DeviceType>()),
    _angle_typew(atom->k_angle_type.view<DeviceType>()),
    _angle_atom1w(atom->k_angle_atom1.view<DeviceType>()),
    _angle_atom2w(atom->k_angle_atom2.view<DeviceType>()),
    _angle_atom3w(atom->k_angle_atom3.view<DeviceType>()),
    _num_dihedralw(atom->k_num_dihedral.view<DeviceType>()),
    _dihedral_typew(atom->k_dihedral_type.view<DeviceType>()),
    _dihedral_atom1w(atom->k_dihedral_atom1.view<DeviceType>()),
    _dihedral_atom2w(atom->k_dihedral_atom2.view<DeviceType>()),
    _dihedral_atom3w(atom->k_dihedral_atom3.view<DeviceType>()),
    _dihedral_atom4w(atom->k_dihedral_atom4.view<DeviceType>()),
    _num_improperw(atom->k_num_improper.view<DeviceType>()),
    _improper_typew(atom->k_improper_type.view<DeviceType>()),
    _improper_atom1w(atom->k_improper_atom1.view<DeviceType>()),
    _improper_atom2w(atom->k_improper_atom2.view<DeviceType>()),
    _improper_atom3w(atom->k_improper_atom3.view<DeviceType>()),
    _improper_atom4w(atom->k_improper_atom4.view<DeviceType>()),
    _sendlist(sendlist.template view<DeviceType>()),
    _copylist(copylist.template view<DeviceType>()),
    _nlocal(nlocal),_dim(dim),
    _lo(lo),_hi(hi) {
    // 3 comp of x, 3 comp of v, 1 tag, 1 type, 1 mask, 1 image, 1 molecule, 3 nspecial,
    // maxspecial special, 1 num_bond, bond_per_atom bond_type, bond_per_atom bond_atom,
    // 1 num_angle, angle_per_atom angle_type, angle_per_atom angle_atom1, angle_atom2,
    // and angle_atom3
    // 1 num_dihedral, dihedral_per_atom dihedral_type, 4*dihedral_per_atom
    // 1 num_improper, 5*improper_per_atom
    // 1 charge
    // 1 to store buffer length
    elements = 20+atom->maxspecial+2*atom->bond_per_atom+4*atom->angle_per_atom+
      5*atom->dihedral_per_atom + 5*atom->improper_per_atom;
    const int maxsendlist = (buf.template view<DeviceType>().extent(0)*
                             buf.template view<DeviceType>().extent(1))/elements;
    buffer_view<DeviceType>(_buf,buf,maxsendlist,elements);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int &mysend) const {
    int k;
    const int i = _sendlist(mysend);
    _buf(mysend,0) = elements;
    int m = 1;
    _buf(mysend,m++) = _x(i,0);
    _buf(mysend,m++) = _x(i,1);
    _buf(mysend,m++) = _x(i,2);
    _buf(mysend,m++) = _v(i,0);
    _buf(mysend,m++) = _v(i,1);
    _buf(mysend,m++) = _v(i,2);
    _buf(mysend,m++) = d_ubuf(_tag(i)).d;
    _buf(mysend,m++) = d_ubuf(_type(i)).d;
    _buf(mysend,m++) = d_ubuf(_mask(i)).d;
    _buf(mysend,m++) = d_ubuf(_image(i)).d;
    _buf(mysend,m++) = _q(i);
    _buf(mysend,m++) = d_ubuf(_molecule(i)).d;
    _buf(mysend,m++) = d_ubuf(_num_bond(i)).d;
    for (k = 0; k < _num_bond(i); k++) {
      _buf(mysend,m++) = d_ubuf(_bond_type(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_bond_atom(i,k)).d;
    }
    _buf(mysend,m++) = d_ubuf(_num_angle(i)).d;
    for (k = 0; k < _num_angle(i); k++) {
      _buf(mysend,m++) = d_ubuf(_angle_type(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_angle_atom1(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_angle_atom2(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_angle_atom3(i,k)).d;
    }
    _buf(mysend,m++) = d_ubuf(_num_dihedral(i)).d;
    for (k = 0; k < _num_dihedral(i); k++) {
      _buf(mysend,m++) = d_ubuf(_dihedral_type(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_dihedral_atom1(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_dihedral_atom2(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_dihedral_atom3(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_dihedral_atom4(i,k)).d;
    }
    _buf(mysend,m++) = d_ubuf(_num_improper(i)).d;
    for (k = 0; k < _num_improper(i); k++) {
      _buf(mysend,m++) = d_ubuf(_improper_type(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_improper_atom1(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_improper_atom2(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_improper_atom3(i,k)).d;
      _buf(mysend,m++) = d_ubuf(_improper_atom4(i,k)).d;
    }

    _buf(mysend,m++) = d_ubuf(_nspecial(i,0)).d;
    _buf(mysend,m++) = d_ubuf(_nspecial(i,1)).d;
    _buf(mysend,m++) = d_ubuf(_nspecial(i,2)).d;
    for (k = 0; k < _nspecial(i,2); k++)
      _buf(mysend,m++) = d_ubuf(_special(i,k)).d;

    const int j = _copylist(mysend);

    if (j>-1) {
    _xw(i,0) = _x(j,0);
    _xw(i,1) = _x(j,1);
    _xw(i,2) = _x(j,2);
    _vw(i,0) = _v(j,0);
    _vw(i,1) = _v(j,1);
    _vw(i,2) = _v(j,2);
    _tagw(i) = _tag(j);
    _typew(i) = _type(j);
    _maskw(i) = _mask(j);
    _imagew(i) = _image(j);
    _qw(i) = _q(j);
    _moleculew(i) = _molecule(j);
    _num_bondw(i) = _num_bond(j);
    for (k = 0; k < _num_bond(j); k++) {
      _bond_typew(i,k) = _bond_type(j,k);
      _bond_atomw(i,k) = _bond_atom(j,k);
    }
    _num_anglew(i) = _num_angle(j);
    for (k = 0; k < _num_angle(j); k++) {
      _angle_typew(i,k) = _angle_type(j,k);
      _angle_atom1w(i,k) = _angle_atom1(j,k);
      _angle_atom2w(i,k) = _angle_atom2(j,k);
      _angle_atom3w(i,k) = _angle_atom3(j,k);
    }
    _num_dihedralw(i) = _num_dihedral(j);
    for (k = 0; k < _num_dihedral(j); k++) {
      _dihedral_typew(i,k) = _dihedral_type(j,k);
      _dihedral_atom1w(i,k) = _dihedral_atom1(j,k);
      _dihedral_atom2w(i,k) = _dihedral_atom2(j,k);
      _dihedral_atom3w(i,k) = _dihedral_atom3(j,k);
      _dihedral_atom4w(i,k) = _dihedral_atom4(j,k);
    }
    _num_improperw(i) = _num_improper(j);
    for (k = 0; k < _num_improper(j); k++) {
      _improper_typew(i,k) = _improper_type(j,k);
      _improper_atom1w(i,k) = _improper_atom1(j,k);
      _improper_atom2w(i,k) = _improper_atom2(j,k);
      _improper_atom3w(i,k) = _improper_atom3(j,k);
      _improper_atom4w(i,k) = _improper_atom4(j,k);
    }
    _nspecialw(i,0) = _nspecial(j,0);
    _nspecialw(i,1) = _nspecial(j,1);
    _nspecialw(i,2) = _nspecial(j,2);
    for (k = 0; k < _nspecial(j,2); k++)
      _specialw(i,k) = _special(j,k);
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecFullKokkos::pack_exchange_kokkos(const int &nsend,DAT::tdual_xfloat_2d &k_buf,
                                                 DAT::tdual_int_1d k_sendlist,
                                                 DAT::tdual_int_1d k_copylist,
                                                 ExecutionSpace space,int dim,X_FLOAT lo,
                                                 X_FLOAT hi )
{
  const int elements = 20+atom->maxspecial+2*atom->bond_per_atom+4*atom->angle_per_atom+
      5*atom->dihedral_per_atom + 5*atom->improper_per_atom;
  if (nsend > (int) (k_buf.view<LMPHostType>().extent(0)*
              k_buf.view<LMPHostType>().extent(1))/elements) {
    int newsize = nsend*elements/k_buf.view<LMPHostType>().extent(1)+1;
    k_buf.resize(newsize,k_buf.view<LMPHostType>().extent(1));
  }
  if (space == Host) {
    AtomVecFullKokkos_PackExchangeFunctor<LMPHostType>
      f(atomKK,k_buf,k_sendlist,k_copylist,atom->nlocal,dim,lo,hi);
    Kokkos::parallel_for(nsend,f);
    return nsend*elements;
  } else {
    AtomVecFullKokkos_PackExchangeFunctor<LMPDeviceType>
      f(atomKK,k_buf,k_sendlist,k_copylist,atom->nlocal,dim,lo,hi);
    Kokkos::parallel_for(nsend,f);
    return nsend*elements;
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecFullKokkos::pack_exchange(int i, double *buf)
{
  int k;
  int m = 1;
  buf[m++] = h_x(i,0);
  buf[m++] = h_x(i,1);
  buf[m++] = h_x(i,2);
  buf[m++] = h_v(i,0);
  buf[m++] = h_v(i,1);
  buf[m++] = h_v(i,2);
  buf[m++] = ubuf(h_tag(i)).d;
  buf[m++] = ubuf(h_type(i)).d;
  buf[m++] = ubuf(h_mask(i)).d;
  buf[m++] = ubuf(h_image(i)).d;
  buf[m++] = h_q(i);
  buf[m++] = ubuf(h_molecule(i)).d;
  buf[m++] = ubuf(h_num_bond(i)).d;
  for (k = 0; k < h_num_bond(i); k++) {
    buf[m++] = ubuf(h_bond_type(i,k)).d;
    buf[m++] = ubuf(h_bond_atom(i,k)).d;
  }
  buf[m++] = ubuf(h_num_angle(i)).d;
  for (k = 0; k < h_num_angle(i); k++) {
    buf[m++] = ubuf(h_angle_type(i,k)).d;
    buf[m++] = ubuf(h_angle_atom1(i,k)).d;
    buf[m++] = ubuf(h_angle_atom2(i,k)).d;
    buf[m++] = ubuf(h_angle_atom3(i,k)).d;
  }
  buf[m++] = ubuf(h_num_dihedral(i)).d;
  for (k = 0; k < h_num_dihedral(i); k++) {
    buf[m++] = ubuf(h_dihedral_type(i,k)).d;
    buf[m++] = ubuf(h_dihedral_atom1(i,k)).d;
    buf[m++] = ubuf(h_dihedral_atom2(i,k)).d;
    buf[m++] = ubuf(h_dihedral_atom3(i,k)).d;
    buf[m++] = ubuf(h_dihedral_atom4(i,k)).d;
  }
  buf[m++] = ubuf(h_num_improper(i)).d;
  for (k = 0; k < h_num_improper(i); k++) {
    buf[m++] = ubuf(h_improper_type(i,k)).d;
    buf[m++] = ubuf(h_improper_atom1(i,k)).d;
    buf[m++] = ubuf(h_improper_atom2(i,k)).d;
    buf[m++] = ubuf(h_improper_atom3(i,k)).d;
    buf[m++] = ubuf(h_improper_atom4(i,k)).d;
  }
  buf[m++] = ubuf(h_nspecial(i,0)).d;
  buf[m++] = ubuf(h_nspecial(i,1)).d;
  buf[m++] = ubuf(h_nspecial(i,2)).d;
  for (k = 0; k < h_nspecial(i,2); k++)
    buf[m++] = ubuf(h_special(i,k)).d;

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->pack_exchange(i,&buf[m]);

  buf[0] = m;
  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecFullKokkos_UnpackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_x_array _x;
  typename AT::t_v_array _v;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_imageint_1d _image;
  typename AT::t_float_1d _q;
  typename AT::t_tagint_1d _molecule;
  typename AT::t_int_2d _nspecial;
  typename AT::t_tagint_2d _special;
  typename AT::t_int_1d _num_bond;
  typename AT::t_int_2d _bond_type;
  typename AT::t_tagint_2d _bond_atom;
  typename AT::t_int_1d _num_angle;
  typename AT::t_int_2d _angle_type;
  typename AT::t_tagint_2d _angle_atom1,_angle_atom2,_angle_atom3;
  typename AT::t_int_1d _num_dihedral;
  typename AT::t_int_2d _dihedral_type;
  typename AT::t_tagint_2d _dihedral_atom1,_dihedral_atom2,
    _dihedral_atom3,_dihedral_atom4;
  typename AT::t_int_1d _num_improper;
  typename AT::t_int_2d _improper_type;
  typename AT::t_tagint_2d _improper_atom1,_improper_atom2,
    _improper_atom3,_improper_atom4;

  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d _nlocal;
  int _dim;
  X_FLOAT _lo,_hi;
  size_t elements;

  AtomVecFullKokkos_UnpackExchangeFunctor(
      const AtomKokkos* atom,
      const typename AT::tdual_xfloat_2d buf,
      typename AT::tdual_int_1d nlocal,
      int dim, X_FLOAT lo, X_FLOAT hi):
    _x(atom->k_x.view<DeviceType>()),
    _v(atom->k_v.view<DeviceType>()),
    _tag(atom->k_tag.view<DeviceType>()),
    _type(atom->k_type.view<DeviceType>()),
    _mask(atom->k_mask.view<DeviceType>()),
    _image(atom->k_image.view<DeviceType>()),
    _q(atom->k_q.view<DeviceType>()),
    _molecule(atom->k_molecule.view<DeviceType>()),
    _nspecial(atom->k_nspecial.view<DeviceType>()),
    _special(atom->k_special.view<DeviceType>()),
    _num_bond(atom->k_num_bond.view<DeviceType>()),
    _bond_type(atom->k_bond_type.view<DeviceType>()),
    _bond_atom(atom->k_bond_atom.view<DeviceType>()),
    _num_angle(atom->k_num_angle.view<DeviceType>()),
    _angle_type(atom->k_angle_type.view<DeviceType>()),
    _angle_atom1(atom->k_angle_atom1.view<DeviceType>()),
    _angle_atom2(atom->k_angle_atom2.view<DeviceType>()),
    _angle_atom3(atom->k_angle_atom3.view<DeviceType>()),
    _num_dihedral(atom->k_num_dihedral.view<DeviceType>()),
    _dihedral_type(atom->k_dihedral_type.view<DeviceType>()),
    _dihedral_atom1(atom->k_dihedral_atom1.view<DeviceType>()),
    _dihedral_atom2(atom->k_dihedral_atom2.view<DeviceType>()),
    _dihedral_atom3(atom->k_dihedral_atom3.view<DeviceType>()),
    _dihedral_atom4(atom->k_dihedral_atom4.view<DeviceType>()),
    _num_improper(atom->k_num_improper.view<DeviceType>()),
    _improper_type(atom->k_improper_type.view<DeviceType>()),
    _improper_atom1(atom->k_improper_atom1.view<DeviceType>()),
    _improper_atom2(atom->k_improper_atom2.view<DeviceType>()),
    _improper_atom3(atom->k_improper_atom3.view<DeviceType>()),
    _improper_atom4(atom->k_improper_atom4.view<DeviceType>()),
    _nlocal(nlocal.template view<DeviceType>()),_dim(dim),
    _lo(lo),_hi(hi) {

    elements = 20+atom->maxspecial+2*atom->bond_per_atom+4*atom->angle_per_atom+
      5*atom->dihedral_per_atom + 5*atom->improper_per_atom;
    const int maxsendlist = (buf.template view<DeviceType>().extent(0)*
                             buf.template view<DeviceType>().extent(1))/elements;
    buffer_view<DeviceType>(_buf,buf,maxsendlist,elements);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int &myrecv) const {
    X_FLOAT x = _buf(myrecv,_dim+1);
    if (x >= _lo && x < _hi) {
      int i = Kokkos::atomic_fetch_add(&_nlocal(0),1);
      int m = 1;
      _x(i,0) = _buf(myrecv,m++);
      _x(i,1) = _buf(myrecv,m++);
      _x(i,2) = _buf(myrecv,m++);
      _v(i,0) = _buf(myrecv,m++);
      _v(i,1) = _buf(myrecv,m++);
      _v(i,2) = _buf(myrecv,m++);
      _tag(i) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
      _type(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
      _mask(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
      _image(i) = (imageint) d_ubuf(_buf(myrecv,m++)).i;
      _q(i) = _buf(myrecv,m++);
      _molecule(i) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
      _num_bond(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
      int k;
      for (k = 0; k < _num_bond(i); k++) {
        _bond_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
        _bond_atom(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
      }
      _num_angle(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
      for (k = 0; k < _num_angle(i); k++) {
        _angle_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
        _angle_atom1(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        _angle_atom2(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        _angle_atom3(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
      }
      _num_dihedral(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
      for (k = 0; k < _num_dihedral(i); k++) {
        _dihedral_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
        _dihedral_atom1(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        _dihedral_atom2(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        _dihedral_atom3(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        _dihedral_atom4(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
      }
      _num_improper(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
      for (k = 0; k < _num_improper(i); k++) {
        _improper_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
        _improper_atom1(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        _improper_atom2(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        _improper_atom3(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        _improper_atom4(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
      }
      _nspecial(i,0) = (int) d_ubuf(_buf(myrecv,m++)).i;
      _nspecial(i,1) = (int) d_ubuf(_buf(myrecv,m++)).i;
      _nspecial(i,2) = (int) d_ubuf(_buf(myrecv,m++)).i;
      for (k = 0; k < _nspecial(i,2); k++)
        _special(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecFullKokkos::unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf,int nrecv,
                                              int nlocal,int dim,X_FLOAT lo,X_FLOAT hi,
                                              ExecutionSpace space) {
  const size_t elements = 20+atom->maxspecial+2*atom->bond_per_atom+4*atom->angle_per_atom+
    5*atom->dihedral_per_atom + 5*atom->improper_per_atom;
  if (space == Host) {
    k_count.h_view(0) = nlocal;
    AtomVecFullKokkos_UnpackExchangeFunctor<LMPHostType>
      f(atomKK,k_buf,k_count,dim,lo,hi);
    Kokkos::parallel_for(nrecv/elements,f);
    return k_count.h_view(0);
  } else {
    k_count.h_view(0) = nlocal;
    k_count.modify<LMPHostType>();
    k_count.sync<LMPDeviceType>();
    AtomVecFullKokkos_UnpackExchangeFunctor<LMPDeviceType>
      f(atomKK,k_buf,k_count,dim,lo,hi);
    Kokkos::parallel_for(nrecv/elements,f);
    k_count.modify<LMPDeviceType>();
    k_count.sync<LMPHostType>();

    return k_count.h_view(0);
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecFullKokkos::unpack_exchange(double *buf)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);
  atomKK->modified(Host,X_MASK | V_MASK | TAG_MASK | TYPE_MASK |
           MASK_MASK | IMAGE_MASK | Q_MASK | MOLECULE_MASK | BOND_MASK |
           ANGLE_MASK | DIHEDRAL_MASK | IMPROPER_MASK | SPECIAL_MASK);

  int k;
  int m = 1;
  h_x(nlocal,0) = buf[m++];
  h_x(nlocal,1) = buf[m++];
  h_x(nlocal,2) = buf[m++];
  h_v(nlocal,0) = buf[m++];
  h_v(nlocal,1) = buf[m++];
  h_v(nlocal,2) = buf[m++];
  h_tag(nlocal) = (tagint) ubuf(buf[m++]).i;
  h_type(nlocal) = (int) ubuf(buf[m++]).i;
  h_mask(nlocal) = (int) ubuf(buf[m++]).i;
  h_image(nlocal) = (imageint) ubuf(buf[m++]).i;
  h_q(nlocal) = buf[m++];
  h_molecule(nlocal) = (tagint) ubuf(buf[m++]).i;

  h_num_bond(nlocal) = (int) ubuf(buf[m++]).i;
  for (k = 0; k < h_num_bond(nlocal); k++) {
    h_bond_type(nlocal,k) = (int) ubuf(buf[m++]).i;
    h_bond_atom(nlocal,k) = (tagint) ubuf(buf[m++]).i;
  }
  h_num_angle(nlocal) = (int) ubuf(buf[m++]).i;
  for (k = 0; k < h_num_angle(nlocal); k++) {
    h_angle_type(nlocal,k) = (int) ubuf(buf[m++]).i;
    h_angle_atom1(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_angle_atom2(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_angle_atom3(nlocal,k) = (tagint) ubuf(buf[m++]).i;
  }
  h_num_dihedral(nlocal) = (int) ubuf(buf[m++]).i;
  for (k = 0; k < h_num_dihedral(nlocal); k++) {
    h_dihedral_type(nlocal,k) = (int) ubuf(buf[m++]).i;
    h_dihedral_atom1(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_dihedral_atom2(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_dihedral_atom3(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_dihedral_atom4(nlocal,k) = (tagint) ubuf(buf[m++]).i;
  }
  h_num_improper(nlocal) = (int) ubuf(buf[m++]).i;
  for (k = 0; k < h_num_improper(nlocal); k++) {
    h_improper_type(nlocal,k) = (int) ubuf(buf[m++]).i;
    h_improper_atom1(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_improper_atom2(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_improper_atom3(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_improper_atom4(nlocal,k) = (tagint) ubuf(buf[m++]).i;
  }
  h_nspecial(nlocal,0) = (int) ubuf(buf[m++]).i;
  h_nspecial(nlocal,1) = (int) ubuf(buf[m++]).i;
  h_nspecial(nlocal,2) = (int) ubuf(buf[m++]).i;
  for (k = 0; k < h_nspecial(nlocal,2); k++)
         h_special(nlocal,k) = (tagint) ubuf(buf[m++]).i;

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->
        unpack_exchange(nlocal,&buf[m]);

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   size of restart data for all atoms owned by this proc
   include extra data stored by fixes
------------------------------------------------------------------------- */

int AtomVecFullKokkos::size_restart()
{
  int i;

  int nlocal = atom->nlocal;
  int n = 0;
  for (i = 0; i < nlocal; i++)
    n += 17 + 2*num_bond[i] + 4*num_angle[i] +
      5*num_dihedral[i] + 5*num_improper[i];

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      for (i = 0; i < nlocal; i++)
        n += modify->fix[atom->extra_restart[iextra]]->size_restart(i);

  return n;
}

/* ----------------------------------------------------------------------
   pack atom I's data for restart file including extra quantities
   xyz must be 1st 3 values, so that read_restart can test on them
   molecular types may be negative, but write as positive
------------------------------------------------------------------------- */

int AtomVecFullKokkos::pack_restart(int i, double *buf)
{
  atomKK->sync(Host,X_MASK | V_MASK | TAG_MASK | TYPE_MASK |
       MASK_MASK | IMAGE_MASK | Q_MASK | MOLECULE_MASK | BOND_MASK |
       ANGLE_MASK | DIHEDRAL_MASK | IMPROPER_MASK | SPECIAL_MASK);

  int m = 1;
  buf[m++] = h_x(i,0);
  buf[m++] = h_x(i,1);
  buf[m++] = h_x(i,2);
  buf[m++] = ubuf(h_tag(i)).d;
  buf[m++] = ubuf(h_type(i)).d;
  buf[m++] = ubuf(h_mask(i)).d;
  buf[m++] = ubuf(h_image(i)).d;
  buf[m++] = h_v(i,0);
  buf[m++] = h_v(i,1);
  buf[m++] = h_v(i,2);

  buf[m++] = h_q(i);
  buf[m++] = ubuf(h_molecule(i)).d;

  buf[m++] = ubuf(h_num_bond(i)).d;
  for (int k = 0; k < h_num_bond(i); k++) {
    buf[m++] = ubuf(MAX(h_bond_type(i,k),-h_bond_type(i,k))).d;
    buf[m++] = ubuf(h_bond_atom(i,k)).d;
  }

  buf[m++] = ubuf(h_num_angle(i)).d;
  for (int k = 0; k < h_num_angle(i); k++) {
    buf[m++] = ubuf(MAX(h_angle_type(i,k),-h_angle_type(i,k))).d;
    buf[m++] = ubuf(h_angle_atom1(i,k)).d;
    buf[m++] = ubuf(h_angle_atom2(i,k)).d;
    buf[m++] = ubuf(h_angle_atom3(i,k)).d;
  }

  buf[m++] = ubuf(h_num_dihedral(i)).d;
  for (int k = 0; k < h_num_dihedral(i); k++) {
    buf[m++] = ubuf(MAX(h_dihedral_type(i,k),-h_dihedral_type(i,k))).d;
    buf[m++] = ubuf(h_dihedral_atom1(i,k)).d;
    buf[m++] = ubuf(h_dihedral_atom2(i,k)).d;
    buf[m++] = ubuf(h_dihedral_atom3(i,k)).d;
    buf[m++] = ubuf(h_dihedral_atom4(i,k)).d;
  }

  buf[m++] = ubuf(h_num_improper(i)).d;
  for (int k = 0; k < h_num_improper(i); k++) {
    buf[m++] = ubuf(MAX(h_improper_type(i,k),-h_improper_type(i,k))).d;
    buf[m++] = ubuf(h_improper_atom1(i,k)).d;
    buf[m++] = ubuf(h_improper_atom2(i,k)).d;
    buf[m++] = ubuf(h_improper_atom3(i,k)).d;
    buf[m++] = ubuf(h_improper_atom4(i,k)).d;
  }

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      m += modify->fix[atom->extra_restart[iextra]]->pack_restart(i,&buf[m]);

  buf[0] = m;
  return m;
}

/* ----------------------------------------------------------------------
   unpack data for one atom from restart file including extra quantities
------------------------------------------------------------------------- */

int AtomVecFullKokkos::unpack_restart(double *buf)
{
  int k;

  int nlocal = atom->nlocal;
  if (nlocal == nmax) {
    grow(0);
    if (atom->nextra_store)
      memory->grow(atom->extra,nmax,atom->nextra_store,"atom:extra");
  }
  atomKK->sync(Host,X_MASK | V_MASK | TAG_MASK | TYPE_MASK |
       MASK_MASK | IMAGE_MASK | Q_MASK | MOLECULE_MASK | BOND_MASK |
       ANGLE_MASK | DIHEDRAL_MASK | IMPROPER_MASK | SPECIAL_MASK);
  atomKK->modified(Host,X_MASK | V_MASK | TAG_MASK | TYPE_MASK |
           MASK_MASK | IMAGE_MASK | Q_MASK | MOLECULE_MASK | BOND_MASK |
           ANGLE_MASK | DIHEDRAL_MASK | IMPROPER_MASK | SPECIAL_MASK);

  int m = 1;
  h_x(nlocal,0) = buf[m++];
  h_x(nlocal,1) = buf[m++];
  h_x(nlocal,2) = buf[m++];
  h_tag(nlocal) = (tagint) ubuf(buf[m++]).i;
  h_type(nlocal) = (int) ubuf(buf[m++]).i;
  h_mask(nlocal) = (int) ubuf(buf[m++]).i;
  h_image(nlocal) = (imageint) ubuf(buf[m++]).i;
  h_v(nlocal,0) = buf[m++];
  h_v(nlocal,1) = buf[m++];
  h_v(nlocal,2) = buf[m++];

  h_q(nlocal) = buf[m++];
  h_molecule(nlocal) = (tagint) ubuf(buf[m++]).i;

  h_num_bond(nlocal) = (int) ubuf(buf[m++]).i;
  for (k = 0; k < h_num_bond(nlocal); k++) {
    h_bond_type(nlocal,k) = (int) ubuf(buf[m++]).i;
    h_bond_atom(nlocal,k) = (tagint) ubuf(buf[m++]).i;
  }

  h_num_angle(nlocal) = (int) ubuf(buf[m++]).i;
  for (k = 0; k < h_num_angle(nlocal); k++) {
    h_angle_type(nlocal,k) = (int) ubuf(buf[m++]).i;
    h_angle_atom1(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_angle_atom2(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_angle_atom3(nlocal,k) = (tagint) ubuf(buf[m++]).i;
  }

  h_num_dihedral(nlocal) = (int) ubuf(buf[m++]).i;
  for (k = 0; k < h_num_dihedral(nlocal); k++) {
    h_dihedral_type(nlocal,k) = (int) ubuf(buf[m++]).i;
    h_dihedral_atom1(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_dihedral_atom2(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_dihedral_atom3(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_dihedral_atom4(nlocal,k) = (tagint) ubuf(buf[m++]).i;
  }

  h_num_improper(nlocal) = (int) ubuf(buf[m++]).i;
  for (k = 0; k < h_num_improper(nlocal); k++) {
    h_improper_type(nlocal,k) = (int) ubuf(buf[m++]).i;
    h_improper_atom1(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_improper_atom2(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_improper_atom3(nlocal,k) = (tagint) ubuf(buf[m++]).i;
    h_improper_atom4(nlocal,k) = (tagint) ubuf(buf[m++]).i;
  }

  h_nspecial(nlocal,0) = h_nspecial(nlocal,1) = h_nspecial(nlocal,2) = 0;

  double **extra = atom->extra;
  if (atom->nextra_store) {
    int size = static_cast<int> (buf[0]) - m;
    for (int i = 0; i < size; i++) extra[nlocal][i] = buf[m++];
  }

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   create one atom of itype at coord
   set other values to defaults
------------------------------------------------------------------------- */

void AtomVecFullKokkos::create_atom(int itype, double *coord)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) {
    atomKK->modified(Host,ALL_MASK);
    grow(0);
  }
  atomKK->sync(Host,ALL_MASK);
  atomKK->modified(Host,ALL_MASK);

  tag[nlocal] = 0;
  type[nlocal] = itype;
  h_x(nlocal,0) = coord[0];
  h_x(nlocal,1) = coord[1];
  h_x(nlocal,2) = coord[2];
  h_mask(nlocal) = 1;
  h_image(nlocal) = ((imageint) IMGMAX << IMG2BITS) |
    ((imageint) IMGMAX << IMGBITS) | IMGMAX;
  h_v(nlocal,0) = 0.0;
  h_v(nlocal,1) = 0.0;
  h_v(nlocal,2) = 0.0;

  h_q(nlocal) = 0.0;
  h_molecule(nlocal) = 0;
  h_num_bond(nlocal) = 0;
  h_num_angle(nlocal) = 0;
  h_num_dihedral(nlocal) = 0;
  h_num_improper(nlocal) = 0;
  h_nspecial(nlocal,0) = h_nspecial(nlocal,1) = h_nspecial(nlocal,2) = 0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
   initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecFullKokkos::data_atom(double *coord, imageint imagetmp,
                                       char **values)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);
  atomKK->modified(Host,ALL_MASK);

  h_tag(nlocal) = utils::inumeric(FLERR,values[0],true,lmp);
  h_molecule(nlocal) = utils::inumeric(FLERR,values[1],true,lmp);
  h_type(nlocal) = utils::inumeric(FLERR,values[2],true,lmp);
  if (h_type(nlocal) <= 0 || h_type(nlocal) > atom->ntypes)
    error->one(FLERR,"Invalid atom type in Atoms section of data file");

  h_q(nlocal) = utils::numeric(FLERR,values[3],true,lmp);

  h_x(nlocal,0) = coord[0];
  h_x(nlocal,1) = coord[1];
  h_x(nlocal,2) = coord[2];

  h_image(nlocal) = imagetmp;

  h_mask(nlocal) = 1;
  h_v(nlocal,0) = 0.0;
  h_v(nlocal,1) = 0.0;
  h_v(nlocal,2) = 0.0;
  h_num_bond(nlocal) = 0;
  h_num_angle(nlocal) = 0;
  h_num_dihedral(nlocal) = 0;
  h_num_improper(nlocal) = 0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack hybrid quantities from one line in Atoms section of data file
   initialize other atom quantities for this sub-style
------------------------------------------------------------------------- */

int AtomVecFullKokkos::data_atom_hybrid(int nlocal, char **values)
{
  h_molecule(nlocal) = utils::inumeric(FLERR,values[0],true,lmp);
  h_q(nlocal) = utils::numeric(FLERR,values[1],true,lmp);
  h_num_bond(nlocal) = 0;
  h_num_angle(nlocal) = 0;
  h_num_dihedral(nlocal) = 0;
  h_num_improper(nlocal) = 0;
  return 2;
}

/* ----------------------------------------------------------------------
   pack atom info for data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecFullKokkos::pack_data(double **buf)
{
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    buf[i][0] = h_tag(i);
    buf[i][1] = h_molecule(i);
    buf[i][2] = h_type(i);
    buf[i][3] = h_q(i);
    buf[i][4] = h_x(i,0);
    buf[i][5] = h_x(i,1);
    buf[i][6] = h_x(i,2);
    buf[i][7] = (h_image[i] & IMGMASK) - IMGMAX;
    buf[i][8] = (h_image[i] >> IMGBITS & IMGMASK) - IMGMAX;
    buf[i][9] = (h_image[i] >> IMG2BITS) - IMGMAX;
  }
}

/* ----------------------------------------------------------------------
   pack hybrid atom info for data file
------------------------------------------------------------------------- */

int AtomVecFullKokkos::pack_data_hybrid(int i, double *buf)
{
  buf[0] = h_molecule(i);
  buf[1] = h_q(i);
  return 2;
}

/* ----------------------------------------------------------------------
   write atom info to data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecFullKokkos::write_data(FILE *fp, int n, double **buf)
{
  for (int i = 0; i < n; i++)
    fprintf(fp,"%d %d %d %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d\n",
            (int) buf[i][0],(int) buf[i][1], (int) buf[i][2], buf[i][3],
            buf[i][4],buf[i][5],buf[i][6],
            (int) buf[i][7],(int) buf[i][8],(int) buf[i][9]);
}

/* ----------------------------------------------------------------------
   write hybrid atom info to data file
------------------------------------------------------------------------- */

int AtomVecFullKokkos::write_data_hybrid(FILE *fp, double *buf)
{
  fprintf(fp," " TAGINT_FORMAT " %-1.16e",(tagint) ubuf(buf[0]).i,buf[1]);
  return 2;
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
------------------------------------------------------------------------- */

double AtomVecFullKokkos::memory_usage()
{
  double bytes = 0;

  if (atom->memcheck("tag")) bytes += memory->usage(tag,nmax);
  if (atom->memcheck("type")) bytes += memory->usage(type,nmax);
  if (atom->memcheck("mask")) bytes += memory->usage(mask,nmax);
  if (atom->memcheck("image")) bytes += memory->usage(image,nmax);
  if (atom->memcheck("x")) bytes += memory->usage(x,nmax,3);
  if (atom->memcheck("v")) bytes += memory->usage(v,nmax,3);
  if (atom->memcheck("f")) bytes += memory->usage(f,nmax*commKK->nthreads,3);

  if (atom->memcheck("q")) bytes += memory->usage(q,nmax);
  if (atom->memcheck("molecule")) bytes += memory->usage(molecule,nmax);
  if (atom->memcheck("nspecial")) bytes += memory->usage(nspecial,nmax,3);
  if (atom->memcheck("special"))
    bytes += memory->usage(special,nmax,atom->maxspecial);

  if (atom->memcheck("num_bond")) bytes += memory->usage(num_bond,nmax);
  if (atom->memcheck("bond_type"))
    bytes += memory->usage(bond_type,nmax,atom->bond_per_atom);
  if (atom->memcheck("bond_atom"))
    bytes += memory->usage(bond_atom,nmax,atom->bond_per_atom);

  if (atom->memcheck("num_angle")) bytes += memory->usage(num_angle,nmax);
  if (atom->memcheck("angle_type"))
    bytes += memory->usage(angle_type,nmax,atom->angle_per_atom);
  if (atom->memcheck("angle_atom1"))
    bytes += memory->usage(angle_atom1,nmax,atom->angle_per_atom);
  if (atom->memcheck("angle_atom2"))
    bytes += memory->usage(angle_atom2,nmax,atom->angle_per_atom);
  if (atom->memcheck("angle_atom3"))
    bytes += memory->usage(angle_atom3,nmax,atom->angle_per_atom);

  if (atom->memcheck("num_dihedral")) bytes += memory->usage(num_dihedral,nmax);
  if (atom->memcheck("dihedral_type"))
    bytes += memory->usage(dihedral_type,nmax,atom->dihedral_per_atom);
  if (atom->memcheck("dihedral_atom1"))
    bytes += memory->usage(dihedral_atom1,nmax,atom->dihedral_per_atom);
  if (atom->memcheck("dihedral_atom2"))
    bytes += memory->usage(dihedral_atom2,nmax,atom->dihedral_per_atom);
  if (atom->memcheck("dihedral_atom3"))
    bytes += memory->usage(dihedral_atom3,nmax,atom->dihedral_per_atom);
  if (atom->memcheck("dihedral_atom4"))
    bytes += memory->usage(dihedral_atom4,nmax,atom->dihedral_per_atom);
  if (atom->memcheck("num_improper")) bytes += memory->usage(num_improper,nmax);
  if (atom->memcheck("improper_type"))
    bytes += memory->usage(improper_type,nmax,atom->improper_per_atom);
  if (atom->memcheck("improper_atom1"))
    bytes += memory->usage(improper_atom1,nmax,atom->improper_per_atom);
  if (atom->memcheck("improper_atom2"))
    bytes += memory->usage(improper_atom2,nmax,atom->improper_per_atom);
  if (atom->memcheck("improper_atom3"))
    bytes += memory->usage(improper_atom3,nmax,atom->improper_per_atom);
  if (atom->memcheck("improper_atom4"))
    bytes += memory->usage(improper_atom4,nmax,atom->improper_per_atom);

  return bytes;
}

/* ---------------------------------------------------------------------- */

void AtomVecFullKokkos::sync(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.sync<LMPDeviceType>();
    if (mask & V_MASK) atomKK->k_v.sync<LMPDeviceType>();
    if (mask & F_MASK) atomKK->k_f.sync<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK->k_tag.sync<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK->k_type.sync<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK->k_mask.sync<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK->k_image.sync<LMPDeviceType>();
    if (mask & Q_MASK) atomKK->k_q.sync<LMPDeviceType>();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.sync<LMPDeviceType>();
    if (mask & SPECIAL_MASK) {
      atomKK->k_nspecial.sync<LMPDeviceType>();
      atomKK->k_special.sync<LMPDeviceType>();
    }
    if (mask & BOND_MASK) {
      atomKK->k_num_bond.sync<LMPDeviceType>();
      atomKK->k_bond_type.sync<LMPDeviceType>();
      atomKK->k_bond_atom.sync<LMPDeviceType>();
    }
    if (mask & ANGLE_MASK) {
      atomKK->k_num_angle.sync<LMPDeviceType>();
      atomKK->k_angle_type.sync<LMPDeviceType>();
      atomKK->k_angle_atom1.sync<LMPDeviceType>();
      atomKK->k_angle_atom2.sync<LMPDeviceType>();
      atomKK->k_angle_atom3.sync<LMPDeviceType>();
    }
    if (mask & DIHEDRAL_MASK) {
      atomKK->k_num_dihedral.sync<LMPDeviceType>();
      atomKK->k_dihedral_type.sync<LMPDeviceType>();
      atomKK->k_dihedral_atom1.sync<LMPDeviceType>();
      atomKK->k_dihedral_atom2.sync<LMPDeviceType>();
      atomKK->k_dihedral_atom3.sync<LMPDeviceType>();
      atomKK->k_dihedral_atom4.sync<LMPDeviceType>();
    }
    if (mask & IMPROPER_MASK) {
      atomKK->k_num_improper.sync<LMPDeviceType>();
      atomKK->k_improper_type.sync<LMPDeviceType>();
      atomKK->k_improper_atom1.sync<LMPDeviceType>();
      atomKK->k_improper_atom2.sync<LMPDeviceType>();
      atomKK->k_improper_atom3.sync<LMPDeviceType>();
      atomKK->k_improper_atom4.sync<LMPDeviceType>();
    }
  } else {
    if (mask & X_MASK) atomKK->k_x.sync<LMPHostType>();
    if (mask & V_MASK) atomKK->k_v.sync<LMPHostType>();
    if (mask & F_MASK) atomKK->k_f.sync<LMPHostType>();
    if (mask & TAG_MASK) atomKK->k_tag.sync<LMPHostType>();
    if (mask & TYPE_MASK) atomKK->k_type.sync<LMPHostType>();
    if (mask & MASK_MASK) atomKK->k_mask.sync<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK->k_image.sync<LMPHostType>();
    if (mask & Q_MASK) atomKK->k_q.sync<LMPHostType>();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.sync<LMPHostType>();
    if (mask & SPECIAL_MASK) {
      atomKK->k_nspecial.sync<LMPHostType>();
      atomKK->k_special.sync<LMPHostType>();
    }
    if (mask & BOND_MASK) {
      atomKK->k_num_bond.sync<LMPHostType>();
      atomKK->k_bond_type.sync<LMPHostType>();
      atomKK->k_bond_atom.sync<LMPHostType>();
    }
    if (mask & ANGLE_MASK) {
      atomKK->k_num_angle.sync<LMPHostType>();
      atomKK->k_angle_type.sync<LMPHostType>();
      atomKK->k_angle_atom1.sync<LMPHostType>();
      atomKK->k_angle_atom2.sync<LMPHostType>();
      atomKK->k_angle_atom3.sync<LMPHostType>();
    }
    if (mask & DIHEDRAL_MASK) {
      atomKK->k_num_dihedral.sync<LMPHostType>();
      atomKK->k_dihedral_type.sync<LMPHostType>();
      atomKK->k_dihedral_atom1.sync<LMPHostType>();
      atomKK->k_dihedral_atom2.sync<LMPHostType>();
      atomKK->k_dihedral_atom3.sync<LMPHostType>();
      atomKK->k_dihedral_atom4.sync<LMPHostType>();
    }
    if (mask & IMPROPER_MASK) {
      atomKK->k_num_improper.sync<LMPHostType>();
      atomKK->k_improper_type.sync<LMPHostType>();
      atomKK->k_improper_atom1.sync<LMPHostType>();
      atomKK->k_improper_atom2.sync<LMPHostType>();
      atomKK->k_improper_atom3.sync<LMPHostType>();
      atomKK->k_improper_atom4.sync<LMPHostType>();
    }
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecFullKokkos::sync_overlapping_device(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if ((mask & X_MASK) && atomKK->k_x.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_x_array>(atomKK->k_x,space);
    if ((mask & V_MASK) && atomKK->k_v.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_v,space);
    if ((mask & F_MASK) && atomKK->k_f.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_f,space);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_type,space);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_mask,space);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space);
    if ((mask & Q_MASK) && atomKK->k_q.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_q,space);
    if ((mask & MOLECULE_MASK) && atomKK->k_molecule.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_molecule,space);
    if (mask & SPECIAL_MASK) {
      if (atomKK->k_nspecial.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_int_2d>(atomKK->k_nspecial,space);
      if (atomKK->k_special.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_special,space);
    }
    if (mask & BOND_MASK) {
      if (atomKK->k_num_bond.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_int_1d>(atomKK->k_num_bond,space);
      if (atomKK->k_bond_type.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_int_2d>(atomKK->k_bond_type,space);
      if (atomKK->k_bond_atom.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_bond_atom,space);
    }
    if (mask & ANGLE_MASK) {
      if (atomKK->k_num_angle.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_int_1d>(atomKK->k_num_angle,space);
      if (atomKK->k_angle_type.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_int_2d>(atomKK->k_angle_type,space);
      if (atomKK->k_angle_atom1.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_angle_atom1,space);
      if (atomKK->k_angle_atom2.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_angle_atom2,space);
      if (atomKK->k_angle_atom3.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_angle_atom3,space);
    }
    if (mask & DIHEDRAL_MASK) {
      if (atomKK->k_num_dihedral.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_int_1d>(atomKK->k_num_dihedral,space);
      if (atomKK->k_dihedral_type.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_int_2d>(atomKK->k_dihedral_type,space);
      if (atomKK->k_dihedral_atom1.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_dihedral_atom1,space);
      if (atomKK->k_dihedral_atom2.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_dihedral_atom2,space);
      if (atomKK->k_dihedral_atom3.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_dihedral_atom3,space);
    }
    if (mask & IMPROPER_MASK) {
      if (atomKK->k_num_improper.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_int_1d>(atomKK->k_num_improper,space);
      if (atomKK->k_improper_type.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_int_2d>(atomKK->k_improper_type,space);
      if (atomKK->k_improper_atom1.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_improper_atom1,space);
      if (atomKK->k_improper_atom2.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_improper_atom2,space);
      if (atomKK->k_improper_atom3.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_improper_atom3,space);
      if (atomKK->k_improper_atom4.need_sync<LMPDeviceType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_improper_atom4,space);
    }
  } else {
    if ((mask & X_MASK) && atomKK->k_x.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_x_array>(atomKK->k_x,space);
    if ((mask & V_MASK) && atomKK->k_v.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_v,space);
    if ((mask & F_MASK) && atomKK->k_f.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_f,space);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_type,space);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_mask,space);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space);
    if ((mask & Q_MASK) && atomKK->k_q.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_q,space);
    if ((mask & MOLECULE_MASK) && atomKK->k_molecule.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_molecule,space);
    if (mask & SPECIAL_MASK) {
      if (atomKK->k_nspecial.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_int_2d>(atomKK->k_nspecial,space);
      if (atomKK->k_special.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_special,space);
    }
    if (mask & BOND_MASK) {
      if (atomKK->k_num_bond.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_int_1d>(atomKK->k_num_bond,space);
      if (atomKK->k_bond_type.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_int_2d>(atomKK->k_bond_type,space);
      if (atomKK->k_bond_atom.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_bond_atom,space);
    }
    if (mask & ANGLE_MASK) {
      if (atomKK->k_num_angle.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_int_1d>(atomKK->k_num_angle,space);
      if (atomKK->k_angle_type.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_int_2d>(atomKK->k_angle_type,space);
      if (atomKK->k_angle_atom1.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_angle_atom1,space);
      if (atomKK->k_angle_atom2.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_angle_atom2,space);
      if (atomKK->k_angle_atom3.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_angle_atom3,space);
    }
    if (mask & DIHEDRAL_MASK) {
      if (atomKK->k_num_dihedral.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_int_1d>(atomKK->k_num_dihedral,space);
      if (atomKK->k_dihedral_type.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_int_2d>(atomKK->k_dihedral_type,space);
      if (atomKK->k_dihedral_atom1.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_dihedral_atom1,space);
      if (atomKK->k_dihedral_atom2.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_dihedral_atom2,space);
      if (atomKK->k_dihedral_atom3.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_dihedral_atom3,space);
      if (atomKK->k_dihedral_atom4.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_dihedral_atom4,space);
    }
    if (mask & IMPROPER_MASK) {
      if (atomKK->k_num_improper.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_int_1d>(atomKK->k_num_improper,space);
      if (atomKK->k_improper_type.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_int_2d>(atomKK->k_improper_type,space);
      if (atomKK->k_improper_atom1.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_improper_atom1,space);
      if (atomKK->k_improper_atom2.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_improper_atom2,space);
      if (atomKK->k_improper_atom3.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_improper_atom3,space);
      if (atomKK->k_improper_atom4.need_sync<LMPHostType>())
        perform_async_copy<DAT::tdual_tagint_2d>(atomKK->k_improper_atom4,space);
    }
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecFullKokkos::modified(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.modify<LMPDeviceType>();
    if (mask & V_MASK) atomKK->k_v.modify<LMPDeviceType>();
    if (mask & F_MASK) atomKK->k_f.modify<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK->k_tag.modify<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK->k_type.modify<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK->k_mask.modify<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK->k_image.modify<LMPDeviceType>();
    if (mask & Q_MASK) atomKK->k_q.modify<LMPDeviceType>();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.modify<LMPDeviceType>();
    if (mask & SPECIAL_MASK) {
      atomKK->k_nspecial.modify<LMPDeviceType>();
      atomKK->k_special.modify<LMPDeviceType>();
    }
    if (mask & BOND_MASK) {
      atomKK->k_num_bond.modify<LMPDeviceType>();
      atomKK->k_bond_type.modify<LMPDeviceType>();
      atomKK->k_bond_atom.modify<LMPDeviceType>();
    }
    if (mask & ANGLE_MASK) {
      atomKK->k_num_angle.modify<LMPDeviceType>();
      atomKK->k_angle_type.modify<LMPDeviceType>();
      atomKK->k_angle_atom1.modify<LMPDeviceType>();
      atomKK->k_angle_atom2.modify<LMPDeviceType>();
      atomKK->k_angle_atom3.modify<LMPDeviceType>();
    }
    if (mask & DIHEDRAL_MASK) {
      atomKK->k_num_dihedral.modify<LMPDeviceType>();
      atomKK->k_dihedral_type.modify<LMPDeviceType>();
      atomKK->k_dihedral_atom1.modify<LMPDeviceType>();
      atomKK->k_dihedral_atom2.modify<LMPDeviceType>();
      atomKK->k_dihedral_atom3.modify<LMPDeviceType>();
      atomKK->k_dihedral_atom4.modify<LMPDeviceType>();
    }
    if (mask & IMPROPER_MASK) {
      atomKK->k_num_improper.modify<LMPDeviceType>();
      atomKK->k_improper_type.modify<LMPDeviceType>();
      atomKK->k_improper_atom1.modify<LMPDeviceType>();
      atomKK->k_improper_atom2.modify<LMPDeviceType>();
      atomKK->k_improper_atom3.modify<LMPDeviceType>();
      atomKK->k_improper_atom4.modify<LMPDeviceType>();
    }
  } else {
    if (mask & X_MASK) atomKK->k_x.modify<LMPHostType>();
    if (mask & V_MASK) atomKK->k_v.modify<LMPHostType>();
    if (mask & F_MASK) atomKK->k_f.modify<LMPHostType>();
    if (mask & TAG_MASK) atomKK->k_tag.modify<LMPHostType>();
    if (mask & TYPE_MASK) atomKK->k_type.modify<LMPHostType>();
    if (mask & MASK_MASK) atomKK->k_mask.modify<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK->k_image.modify<LMPHostType>();
    if (mask & Q_MASK) atomKK->k_q.modify<LMPHostType>();
    if (mask & MOLECULE_MASK) atomKK->k_molecule.modify<LMPHostType>();
    if (mask & SPECIAL_MASK) {
      atomKK->k_nspecial.modify<LMPHostType>();
      atomKK->k_special.modify<LMPHostType>();
    }
    if (mask & BOND_MASK) {
      atomKK->k_num_bond.modify<LMPHostType>();
      atomKK->k_bond_type.modify<LMPHostType>();
      atomKK->k_bond_atom.modify<LMPHostType>();
    }
    if (mask & ANGLE_MASK) {
      atomKK->k_num_angle.modify<LMPHostType>();
      atomKK->k_angle_type.modify<LMPHostType>();
      atomKK->k_angle_atom1.modify<LMPHostType>();
      atomKK->k_angle_atom2.modify<LMPHostType>();
      atomKK->k_angle_atom3.modify<LMPHostType>();
    }
    if (mask & DIHEDRAL_MASK) {
      atomKK->k_num_dihedral.modify<LMPHostType>();
      atomKK->k_dihedral_type.modify<LMPHostType>();
      atomKK->k_dihedral_atom1.modify<LMPHostType>();
      atomKK->k_dihedral_atom2.modify<LMPHostType>();
      atomKK->k_dihedral_atom3.modify<LMPHostType>();
      atomKK->k_dihedral_atom4.modify<LMPHostType>();
    }
    if (mask & IMPROPER_MASK) {
      atomKK->k_num_improper.modify<LMPHostType>();
      atomKK->k_improper_type.modify<LMPHostType>();
      atomKK->k_improper_atom1.modify<LMPHostType>();
      atomKK->k_improper_atom2.modify<LMPHostType>();
      atomKK->k_improper_atom3.modify<LMPHostType>();
      atomKK->k_improper_atom4.modify<LMPHostType>();
    }
  }
}

