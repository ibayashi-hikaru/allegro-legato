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

#include "gridcomm_kokkos.h"

#include "comm.h"
#include "irregular.h"
#include "kokkos.h"
#include "kokkos_base_fft.h"
#include "kspace.h"
#include "memory_kokkos.h"

using namespace LAMMPS_NS;

enum{REGULAR,TILED};

#define DELTA 16

/* ----------------------------------------------------------------------
   NOTES
   tiled implementation only currently works for RCB, not general tiled
   b/c RCB tree is used to find neighboring tiles
   if o indices for ghosts are < 0 or hi indices are >= N,
     then grid is treated as periodic in that dimension,
     communication is done across the periodic boundaries
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   constructor called by all classes except MSM
   gcomm = world communicator
   gn xyz = size of global grid
   i xyz lohi = portion of global grid this proc owns, 0 <= index < N
   o xyz lohi = owned grid portion + ghost grid cells needed in all directions
   if o indices are < 0 or hi indices are >= N,
     then grid is treated as periodic in that dimension,
     communication is done across the periodic boundaries
------------------------------------------------------------------------- */

template<class DeviceType>
GridCommKokkos<DeviceType>::GridCommKokkos(LAMMPS *lmp, MPI_Comm gcomm,
                   int gnx, int gny, int gnz,
                   int ixlo, int ixhi, int iylo, int iyhi, int izlo, int izhi,
                   int oxlo, int oxhi, int oylo, int oyhi, int ozlo, int ozhi)
  : GridComm(lmp, gcomm,
             gnx, gny, gnz,
             ixlo,ixhi, iylo, iyhi, izlo, izhi,
             oxlo, oxhi, oylo, oyhi, ozlo, ozhi)
{
}

/* ----------------------------------------------------------------------
   constructor called by MSM
   gcomm = world communicator or sub-communicator for a hierarchical grid
   flag = 1 if e xyz lohi values = larger grid stored by caller in gcomm = world
   flag = 2 if e xyz lohi values = 6 neighbor procs in gcomm
   gn xyz = size of global grid
   i xyz lohi = portion of global grid this proc owns, 0 <= index < N
   o xyz lohi = owned grid portion + ghost grid cells needed in all directions
   e xyz lohi for flag = 1: extent of larger grid stored by caller
   e xyz lohi for flag = 2: 6 neighbor procs
------------------------------------------------------------------------- */

template<class DeviceType>
GridCommKokkos<DeviceType>::GridCommKokkos(LAMMPS *lmp, MPI_Comm gcomm, int /*flag*/,
                   int gnx, int gny, int gnz,
                   int ixlo, int ixhi, int iylo, int iyhi, int izlo, int izhi,
                   int oxlo, int oxhi, int oylo, int oyhi, int ozlo, int ozhi,
           int /*exlo*/, int /*exhi*/, int /*eylo*/, int /*eyhi*/, int /*ezlo*/, int /*ezhi*/)
  : GridComm(lmp, gcomm,
             gnx, gny, gnz,
             ixlo,ixhi, iylo, iyhi, izlo, izhi,
             oxlo, oxhi, oylo, oyhi, ozlo, ozhi)
{
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
GridCommKokkos<DeviceType>::~GridCommKokkos()
{
  // regular comm data struct

  for (int i = 0; i < nswap; i++) {
    swap[i].packlist = nullptr;
    swap[i].unpacklist = nullptr;
  }

  // tiled comm data structs

  for (int i = 0; i < nsend; i++)
    send[i].packlist = nullptr;

  for (int i = 0; i < nrecv; i++)
    recv[i].unpacklist = nullptr;

  for (int i = 0; i < ncopy; i++) {
    copy[i].packlist = nullptr;
    copy[i].unpacklist = nullptr;
  }

}

/* ----------------------------------------------------------------------
   setup comm for a regular grid of procs
   each proc has 6 neighbors
   comm pattern = series of swaps with one of those 6 procs
   can be multiple swaps with same proc if ghost extent is large
   swap may not be symmetric if both procs do not need same layers of ghosts
   all procs perform same # of swaps in a direction, even if some don't need it
------------------------------------------------------------------------- */

template<class DeviceType>
void GridCommKokkos<DeviceType>::setup_regular(int &nbuf1, int &nbuf2)
{
  int nsent,sendfirst,sendlast,recvfirst,recvlast;
  int sendplanes,recvplanes;
  int notdoneme,notdone;

  // notify 6 neighbor procs how many ghost grid planes I need from them
  // ghost xyz lo = # of my lower grid planes that proc xyz lo needs as its ghosts
  // ghost xyz hi = # of my upper grid planes that proc xyz hi needs as its ghosts
  // if this proc is its own neighbor across periodic bounary, value is from self

  int nplanes = inxlo - outxlo;
  if (procxlo != me)
      MPI_Sendrecv(&nplanes,1,MPI_INT,procxlo,0,
                   &ghostxhi,1,MPI_INT,procxhi,0,gridcomm,MPI_STATUS_IGNORE);
  else ghostxhi = nplanes;

  nplanes = outxhi - inxhi;
  if (procxhi != me)
      MPI_Sendrecv(&nplanes,1,MPI_INT,procxhi,0,
                   &ghostxlo,1,MPI_INT,procxlo,0,gridcomm,MPI_STATUS_IGNORE);
  else ghostxlo = nplanes;

  nplanes = inylo - outylo;
  if (procylo != me)
    MPI_Sendrecv(&nplanes,1,MPI_INT,procylo,0,
                 &ghostyhi,1,MPI_INT,procyhi,0,gridcomm,MPI_STATUS_IGNORE);
  else ghostyhi = nplanes;

  nplanes = outyhi - inyhi;
  if (procyhi != me)
    MPI_Sendrecv(&nplanes,1,MPI_INT,procyhi,0,
                 &ghostylo,1,MPI_INT,procylo,0,gridcomm,MPI_STATUS_IGNORE);
  else ghostylo = nplanes;

  nplanes = inzlo - outzlo;
  if (proczlo != me)
    MPI_Sendrecv(&nplanes,1,MPI_INT,proczlo,0,
                 &ghostzhi,1,MPI_INT,proczhi,0,gridcomm,MPI_STATUS_IGNORE);
  else ghostzhi = nplanes;

  nplanes = outzhi - inzhi;
  if (proczhi != me)
    MPI_Sendrecv(&nplanes,1,MPI_INT,proczhi,0,
                 &ghostzlo,1,MPI_INT,proczlo,0,gridcomm,MPI_STATUS_IGNORE);
  else ghostzlo = nplanes;

  // setup swaps = exchange of grid data with one of 6 neighobr procs
  // can be more than one in a direction if ghost region extends beyond neigh proc
  // all procs have same swap count, but swapsize npack/nunpack can be empty

  nswap = 0;

  // send own grid pts to -x processor, recv ghost grid pts from +x processor

  nsent = 0;
  sendfirst = inxlo;
  sendlast = inxhi;
  recvfirst = inxhi+1;
  notdone = 1;

  while (notdone) {
    if (nswap == maxswap) grow_swap();

    swap[nswap].sendproc = procxlo;
    swap[nswap].recvproc = procxhi;
    sendplanes = MIN(sendlast-sendfirst+1,ghostxlo-nsent);
    swap[nswap].npack =
      indices(k_swap_packlist,nswap,
              sendfirst,sendfirst+sendplanes-1,inylo,inyhi,inzlo,inzhi);

    if (procxlo != me)
      MPI_Sendrecv(&sendplanes,1,MPI_INT,procxlo,0,
                   &recvplanes,1,MPI_INT,procxhi,0,gridcomm,MPI_STATUS_IGNORE);
    else recvplanes = sendplanes;

    swap[nswap].nunpack =
      indices(k_swap_unpacklist,nswap,
              recvfirst,recvfirst+recvplanes-1,inylo,inyhi,inzlo,inzhi);

    nsent += sendplanes;
    sendfirst += sendplanes;
    sendlast += recvplanes;
    recvfirst += recvplanes;
    nswap++;

    if (nsent < ghostxlo) notdoneme = 1;
    else notdoneme = 0;
    MPI_Allreduce(&notdoneme,&notdone,1,MPI_INT,MPI_SUM,gridcomm);
  }

  // send own grid pts to +x processor, recv ghost grid pts from -x processor

  nsent = 0;
  sendfirst = inxlo;
  sendlast = inxhi;
  recvlast = inxlo-1;
  notdone = 1;

  while (notdone) {
    if (nswap == maxswap) grow_swap();

    swap[nswap].sendproc = procxhi;
    swap[nswap].recvproc = procxlo;
    sendplanes = MIN(sendlast-sendfirst+1,ghostxhi-nsent);
    swap[nswap].npack =
      indices(k_swap_packlist,nswap,
              sendlast-sendplanes+1,sendlast,inylo,inyhi,inzlo,inzhi);

    if (procxhi != me)
      MPI_Sendrecv(&sendplanes,1,MPI_INT,procxhi,0,
                   &recvplanes,1,MPI_INT,procxlo,0,gridcomm,MPI_STATUS_IGNORE);
    else recvplanes = sendplanes;

    swap[nswap].nunpack =
      indices(k_swap_unpacklist,nswap,
              recvlast-recvplanes+1,recvlast,inylo,inyhi,inzlo,inzhi);

    nsent += sendplanes;
    sendfirst -= recvplanes;
    sendlast -= sendplanes;
    recvlast -= recvplanes;
    nswap++;

    if (nsent < ghostxhi) notdoneme = 1;
    else notdoneme = 0;
    MPI_Allreduce(&notdoneme,&notdone,1,MPI_INT,MPI_SUM,gridcomm);
  }

  // send own grid pts to -y processor, recv ghost grid pts from +y processor

  nsent = 0;
  sendfirst = inylo;
  sendlast = inyhi;
  recvfirst = inyhi+1;
  notdone = 1;

  while (notdone) {
    if (nswap == maxswap) grow_swap();

    swap[nswap].sendproc = procylo;
    swap[nswap].recvproc = procyhi;
    sendplanes = MIN(sendlast-sendfirst+1,ghostylo-nsent);
    swap[nswap].npack =
      indices(k_swap_packlist,nswap,
              outxlo,outxhi,sendfirst,sendfirst+sendplanes-1,inzlo,inzhi);

    if (procylo != me)
      MPI_Sendrecv(&sendplanes,1,MPI_INT,procylo,0,
                   &recvplanes,1,MPI_INT,procyhi,0,gridcomm,MPI_STATUS_IGNORE);
    else recvplanes = sendplanes;

    swap[nswap].nunpack =
      indices(k_swap_unpacklist,nswap,
              outxlo,outxhi,recvfirst,recvfirst+recvplanes-1,inzlo,inzhi);

    nsent += sendplanes;
    sendfirst += sendplanes;
    sendlast += recvplanes;
    recvfirst += recvplanes;
    nswap++;

    if (nsent < ghostylo) notdoneme = 1;
    else notdoneme = 0;
    MPI_Allreduce(&notdoneme,&notdone,1,MPI_INT,MPI_SUM,gridcomm);
  }

  // send own grid pts to +y processor, recv ghost grid pts from -y processor

  nsent = 0;
  sendfirst = inylo;
  sendlast = inyhi;
  recvlast = inylo-1;
  notdone = 1;

  while (notdone) {
    if (nswap == maxswap) grow_swap();

    swap[nswap].sendproc = procyhi;
    swap[nswap].recvproc = procylo;
    sendplanes = MIN(sendlast-sendfirst+1,ghostyhi-nsent);
    swap[nswap].npack =
      indices(k_swap_packlist,nswap,
              outxlo,outxhi,sendlast-sendplanes+1,sendlast,inzlo,inzhi);

    if (procyhi != me)
      MPI_Sendrecv(&sendplanes,1,MPI_INT,procyhi,0,
                   &recvplanes,1,MPI_INT,procylo,0,gridcomm,MPI_STATUS_IGNORE);
    else recvplanes = sendplanes;

    swap[nswap].nunpack =
      indices(k_swap_unpacklist,nswap,
              outxlo,outxhi,recvlast-recvplanes+1,recvlast,inzlo,inzhi);

    nsent += sendplanes;
    sendfirst -= recvplanes;
    sendlast -= sendplanes;
    recvlast -= recvplanes;
    nswap++;

    if (nsent < ghostyhi) notdoneme = 1;
    else notdoneme = 0;
    MPI_Allreduce(&notdoneme,&notdone,1,MPI_INT,MPI_SUM,gridcomm);
  }

  // send own grid pts to -z processor, recv ghost grid pts from +z processor

  nsent = 0;
  sendfirst = inzlo;
  sendlast = inzhi;
  recvfirst = inzhi+1;
  notdone = 1;

  while (notdone) {
    if (nswap == maxswap) grow_swap();

    swap[nswap].sendproc = proczlo;
    swap[nswap].recvproc = proczhi;
    sendplanes = MIN(sendlast-sendfirst+1,ghostzlo-nsent);
    swap[nswap].npack =
      indices(k_swap_packlist,nswap,
              outxlo,outxhi,outylo,outyhi,sendfirst,sendfirst+sendplanes-1);

    if (proczlo != me)
      MPI_Sendrecv(&sendplanes,1,MPI_INT,proczlo,0,
                   &recvplanes,1,MPI_INT,proczhi,0,gridcomm,MPI_STATUS_IGNORE);
    else recvplanes = sendplanes;

    swap[nswap].nunpack =
      indices(k_swap_unpacklist,nswap,
              outxlo,outxhi,outylo,outyhi,recvfirst,recvfirst+recvplanes-1);

    nsent += sendplanes;
    sendfirst += sendplanes;
    sendlast += recvplanes;
    recvfirst += recvplanes;
    nswap++;

    if (nsent < ghostzlo) notdoneme = 1;
    else notdoneme = 0;
    MPI_Allreduce(&notdoneme,&notdone,1,MPI_INT,MPI_SUM,gridcomm);
  }

  // send own grid pts to +z processor, recv ghost grid pts from -z processor

  nsent = 0;
  sendfirst = inzlo;
  sendlast = inzhi;
  recvlast = inzlo-1;
  notdone = 1;

  while (notdone) {
    if (nswap == maxswap) grow_swap();

    swap[nswap].sendproc = proczhi;
    swap[nswap].recvproc = proczlo;
    sendplanes = MIN(sendlast-sendfirst+1,ghostzhi-nsent);
    swap[nswap].npack =
      indices(k_swap_packlist,nswap,
              outxlo,outxhi,outylo,outyhi,sendlast-sendplanes+1,sendlast);

    if (proczhi != me)
      MPI_Sendrecv(&sendplanes,1,MPI_INT,proczhi,0,
                   &recvplanes,1,MPI_INT,proczlo,0,gridcomm,MPI_STATUS_IGNORE);
    else recvplanes = sendplanes;

    swap[nswap].nunpack =
      indices(k_swap_unpacklist,nswap,
              outxlo,outxhi,outylo,outyhi,recvlast-recvplanes+1,recvlast);

    nsent += sendplanes;
    sendfirst -= recvplanes;
    sendlast -= sendplanes;
    recvlast -= recvplanes;
    nswap++;

    if (nsent < ghostzhi) notdoneme = 1;
    else notdoneme = 0;
    MPI_Allreduce(&notdoneme,&notdone,1,MPI_INT,MPI_SUM,gridcomm);
  }

  // ngrid = max of any forward/reverse pack/unpack grid points

  int ngrid = 0;
  for (int i = 0; i < nswap; i++) {
    ngrid = MAX(ngrid,swap[i].npack);
    ngrid = MAX(ngrid,swap[i].nunpack);
  }

  nbuf1 = nbuf2 = ngrid;
}

/* ----------------------------------------------------------------------
   setup comm for RCB tiled proc domains
   each proc has arbitrary # of neighbors that overlap its ghost extent
   identify which procs will send me ghost cells, and vice versa
   may not be symmetric if both procs do not need same layers of ghosts
   comm pattern = post recvs for all my ghosts, send my owned, wait on recvs
     no exchanges by dimension, unlike CommTiled forward/reverse comm of particles
------------------------------------------------------------------------- */

template<class DeviceType>
void GridCommKokkos<DeviceType>::setup_tiled(int &nbuf1, int &nbuf2)
{
  int i,m;
  double xlo,xhi,ylo,yhi,zlo,zhi;
  int ghostbox[6],pbc[3];

  // setup RCB tree of cut info for grid
  // access CommTiled to get cut dimension
  // cut = this proc's inlo in that dim
  // dim is -1 for proc 0, but never accessed

  rcbinfo = (RCBinfo *)
    memory->smalloc(nprocs*sizeof(RCBinfo),"GridComm:rcbinfo");
  RCBinfo rcbone;
  rcbone.dim = comm->rcbcutdim;
  if (rcbone.dim <= 0) rcbone.cut = inxlo;
  else if (rcbone.dim == 1) rcbone.cut = inylo;
  else if (rcbone.dim == 2) rcbone.cut = inzlo;
  MPI_Allgather(&rcbone,sizeof(RCBinfo),MPI_CHAR,
                rcbinfo,sizeof(RCBinfo),MPI_CHAR,gridcomm);

  // find overlaps of my extended ghost box with all other procs
  // accounts for crossings of periodic boundaries
  // noverlap = # of overlaps, including self
  // overlap = vector of overlap info using Overlap data struct

  ghostbox[0] = outxlo;
  ghostbox[1] = outxhi;
  ghostbox[2] = outylo;
  ghostbox[3] = outyhi;
  ghostbox[4] = outzlo;
  ghostbox[5] = outzhi;

  pbc[0] = pbc[1] = pbc[2] = 0;

  memory->create(overlap_procs,nprocs,"GridComm:overlap_procs");
  noverlap = maxoverlap = 0;
  overlap = nullptr;

  ghost_box_drop(ghostbox,pbc);

  // send each proc an overlap message
  // content: me, index of my overlap, box that overlaps with its owned cells
  // ncopy = # of overlaps with myself, across a periodic boundary

  int *proclist;
  memory->create(proclist,noverlap,"GridComm:proclist");
  srequest = (Request *)
    memory->smalloc(noverlap*sizeof(Request),"GridComm:srequest");

  int nsend_request = 0;
  ncopy = 0;

  for (m = 0; m < noverlap; m++) {
    if (overlap[m].proc == me) ncopy++;
    else {
      proclist[nsend_request] = overlap[m].proc;
      srequest[nsend_request].sender = me;
      srequest[nsend_request].index = m;
      for (i = 0; i < 6; i++)
        srequest[nsend_request].box[i] = overlap[m].box[i];
      nsend_request++;
    }
  }

  Irregular *irregular = new Irregular(lmp);
  int nrecv_request = irregular->create_data(nsend_request,proclist,1);
  Request *rrequest =
    (Request *) memory->smalloc(nrecv_request*sizeof(Request),"GridComm:rrequest");
  irregular->exchange_data((char *) srequest,sizeof(Request),(char *) rrequest);
  irregular->destroy_data();

  // compute overlaps between received ghost boxes and my owned box
  // overlap box used to setup my Send data struct and respond to requests

  send = (Send *) memory->smalloc(nrecv_request*sizeof(Send),"GridComm:send");

  k_send_packlist = DAT::tdual_int_2d("GridComm:send_packlist",nrecv_request,k_send_packlist.extent(1));

  sresponse = (Response *)
    memory->smalloc(nrecv_request*sizeof(Response),"GridComm:sresponse");
  memory->destroy(proclist);
  memory->create(proclist,nrecv_request,"GridComm:proclist");

  for (m = 0; m < nrecv_request; m++) {
    send[m].proc = rrequest[m].sender;
    xlo = MAX(rrequest[m].box[0],inxlo);
    xhi = MIN(rrequest[m].box[1],inxhi);
    ylo = MAX(rrequest[m].box[2],inylo);
    yhi = MIN(rrequest[m].box[3],inyhi);
    zlo = MAX(rrequest[m].box[4],inzlo);
    zhi = MIN(rrequest[m].box[5],inzhi);
    send[m].npack = indices(k_send_packlist,m,xlo,xhi,ylo,yhi,zlo,zhi);

    proclist[m] = rrequest[m].sender;
    sresponse[m].index = rrequest[m].index;
    sresponse[m].box[0] = xlo;
    sresponse[m].box[1] = xhi;
    sresponse[m].box[2] = ylo;
    sresponse[m].box[3] = yhi;
    sresponse[m].box[4] = zlo;
    sresponse[m].box[5] = zhi;
  }

  nsend = nrecv_request;

  // reply to each Request message with a Response message
  // content: index for the overlap on requestor, overlap box on my owned grid

  int nsend_response = nrecv_request;
  int nrecv_response = irregular->create_data(nsend_response,proclist,1);
  Response *rresponse =
    (Response *) memory->smalloc(nrecv_response*sizeof(Response),"GridComm:rresponse");
  irregular->exchange_data((char *) sresponse,sizeof(Response),(char *) rresponse);
  irregular->destroy_data();
  delete irregular;

  // process received responses
  // box used to setup my Recv data struct after unwrapping via PBC
  // adjacent = 0 if any box of ghost cells does not adjoin my owned cells

  recv = (Recv *) memory->smalloc(nrecv_response*sizeof(Recv),"GridComm:recv");

  k_recv_unpacklist = DAT::tdual_int_2d("GridComm:recv_unpacklist",nrecv_response,k_recv_unpacklist.extent(1));

  adjacent = 1;

  for (i = 0; i < nrecv_response; i++) {
    m = rresponse[i].index;
    recv[i].proc = overlap[m].proc;
    xlo = rresponse[i].box[0] + overlap[m].pbc[0] * nx;
    xhi = rresponse[i].box[1] + overlap[m].pbc[0] * nx;
    ylo = rresponse[i].box[2] + overlap[m].pbc[1] * ny;
    yhi = rresponse[i].box[3] + overlap[m].pbc[1] * ny;
    zlo = rresponse[i].box[4] + overlap[m].pbc[2] * nz;
    zhi = rresponse[i].box[5] + overlap[m].pbc[2] * nz;
    recv[i].nunpack = indices(k_recv_unpacklist,i,xlo,xhi,ylo,yhi,zlo,zhi);

    if (xlo != inxhi+1 && xhi != inxlo-1 &&
        ylo != inyhi+1 && yhi != inylo-1 &&
        zlo != inzhi+1 && zhi != inzlo-1) adjacent = 0;
  }

  nrecv = nrecv_response;

  // create Copy data struct from overlaps with self

  copy = (Copy *) memory->smalloc(ncopy*sizeof(Copy),"GridComm:copy");

  k_copy_packlist = DAT::tdual_int_2d("GridComm:copy_packlist",ncopy,k_copy_packlist.extent(1));
  k_copy_unpacklist = DAT::tdual_int_2d("GridComm:copy_unpacklist",ncopy,k_copy_unpacklist.extent(1));

  ncopy = 0;
  for (m = 0; m < noverlap; m++) {
    if (overlap[m].proc != me) continue;
    xlo = overlap[m].box[0];
    xhi = overlap[m].box[1];
    ylo = overlap[m].box[2];
    yhi = overlap[m].box[3];
    zlo = overlap[m].box[4];
    zhi = overlap[m].box[5];
    copy[ncopy].npack = indices(k_copy_packlist,ncopy,xlo,xhi,ylo,yhi,zlo,zhi);
    xlo = overlap[m].box[0] + overlap[m].pbc[0] * nx;
    xhi = overlap[m].box[1] + overlap[m].pbc[0] * nx;
    ylo = overlap[m].box[2] + overlap[m].pbc[1] * ny;
    yhi = overlap[m].box[3] + overlap[m].pbc[1] * ny;
    zlo = overlap[m].box[4] + overlap[m].pbc[2] * nz;
    zhi = overlap[m].box[5] + overlap[m].pbc[2] * nz;
    copy[ncopy].nunpack = indices(k_copy_unpacklist,ncopy,xlo,xhi,ylo,yhi,zlo,zhi);
    ncopy++;
  }

  // set offsets for received data

  int offset = 0;
  for (m = 0; m < nsend; m++) {
    send[m].offset = offset;
    offset += send[m].npack;
  }

  offset = 0;
  for (m = 0; m < nrecv; m++) {
    recv[m].offset = offset;
    offset += recv[m].nunpack;
  }

  // length of MPI requests vector is max of nsend, nrecv

  int nrequest = MAX(nsend,nrecv);
  requests = new MPI_Request[nrequest];

  // clean-up

  memory->sfree(rcbinfo);
  memory->destroy(proclist);
  memory->destroy(overlap_procs);
  memory->sfree(overlap);
  memory->sfree(srequest);
  memory->sfree(rrequest);
  memory->sfree(sresponse);
  memory->sfree(rresponse);

  // nbuf1 = largest pack or unpack in any Send or Recv or Copy
  // nbuf2 = larget of sum of all packs or unpacks in Send or Recv

  nbuf1 = 0;

  for (m = 0; m < ncopy; m++) {
    nbuf1 = MAX(nbuf1,copy[m].npack);
    nbuf1 = MAX(nbuf1,copy[m].nunpack);
  }

  int nbufs = 0;
  for (m = 0; m < nsend; m++) {
    nbuf1 = MAX(nbuf1,send[m].npack);
    nbufs += send[m].npack;
  }

  int nbufr = 0;
  for (m = 0; m < nrecv; m++) {
    nbuf1 = MAX(nbuf1,recv[m].nunpack);
    nbufr += recv[m].nunpack;
  }

  nbuf2 = MAX(nbufs,nbufr);
}

/* ----------------------------------------------------------------------
   forward comm of my owned cells to other's ghost cells
------------------------------------------------------------------------- */

template<class DeviceType>
void GridCommKokkos<DeviceType>::forward_comm_kspace(KSpace *kspace, int nper, int which,
                                   FFT_DAT::tdual_FFT_SCALAR_1d &k_buf1, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf2, MPI_Datatype datatype)
{
  if (layout == REGULAR)
    forward_comm_kspace_regular(kspace,nper,which,k_buf1,k_buf2,datatype);
  else
    forward_comm_kspace_tiled(kspace,nper,which,k_buf1,k_buf2,datatype);
}

/* ----------------------------------------------------------------------
   forward comm on regular grid of procs via list of swaps with 6 neighbor procs
------------------------------------------------------------------------- */

template<class DeviceType>
void GridCommKokkos<DeviceType>::
forward_comm_kspace_regular(KSpace *kspace, int nper, int which,
                            FFT_DAT::tdual_FFT_SCALAR_1d &k_buf1, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf2, MPI_Datatype datatype)
{
  int m;
  MPI_Request request;

  KokkosBaseFFT* kspaceKKBase = dynamic_cast<KokkosBaseFFT*>(kspace);
  FFT_SCALAR* buf1;
  FFT_SCALAR* buf2;
  if (lmp->kokkos->gpu_aware_flag) {
    buf1 = k_buf1.view<DeviceType>().data();
    buf2 = k_buf2.view<DeviceType>().data();
  } else {
    buf1 = k_buf1.h_view.data();
    buf2 = k_buf2.h_view.data();
  }

  for (m = 0; m < nswap; m++) {
    if (swap[m].sendproc == me)
      kspaceKKBase->pack_forward_grid_kokkos(which,k_buf2,swap[m].npack,k_swap_packlist,m);
    else
      kspaceKKBase->pack_forward_grid_kokkos(which,k_buf1,swap[m].npack,k_swap_packlist,m);
    DeviceType().fence();

    if (swap[m].sendproc != me) {

      if (!lmp->kokkos->gpu_aware_flag) {
        k_buf1.modify<DeviceType>();
        k_buf1.sync<LMPHostType>();
      }

      if (swap[m].nunpack) MPI_Irecv(buf2,nper*swap[m].nunpack,datatype,
                                     swap[m].recvproc,0,gridcomm,&request);
      if (swap[m].npack) MPI_Send(buf1,nper*swap[m].npack,datatype,
                                  swap[m].sendproc,0,gridcomm);
      if (swap[m].nunpack) MPI_Wait(&request,MPI_STATUS_IGNORE);

      if (!lmp->kokkos->gpu_aware_flag) {
        k_buf2.modify<LMPHostType>();
        k_buf2.sync<DeviceType>();
      }
    }

    kspaceKKBase->unpack_forward_grid_kokkos(which,k_buf2,0,swap[m].nunpack,k_swap_unpacklist,m);
    DeviceType().fence();
  }
}

/* ----------------------------------------------------------------------
   forward comm on tiled grid decomp via Send/Recv lists of each neighbor proc
------------------------------------------------------------------------- */

template<class DeviceType>
void GridCommKokkos<DeviceType>::
forward_comm_kspace_tiled(KSpace *kspace, int nper, int which,
                          FFT_DAT::tdual_FFT_SCALAR_1d &k_buf1, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf2, MPI_Datatype datatype)
{
  int i,m,offset;

  KokkosBaseFFT* kspaceKKBase = dynamic_cast<KokkosBaseFFT*>(kspace);
  FFT_SCALAR* buf1;
  FFT_SCALAR* buf2;
  if (lmp->kokkos->gpu_aware_flag) {
    buf1 = k_buf1.view<DeviceType>().data();
    buf2 = k_buf2.view<DeviceType>().data();
  } else {
    buf1 = k_buf1.h_view.data();
    buf2 = k_buf2.h_view.data();
  }

  // post all receives

  for (m = 0; m < nrecv; m++) {
    offset = nper * recv[m].offset;
    MPI_Irecv(&buf2[offset],nper*recv[m].nunpack,datatype,
              recv[m].proc,0,gridcomm,&requests[m]);
  }

  // perform all sends to other procs

  for (m = 0; m < nsend; m++) {
    kspaceKKBase->pack_forward_grid_kokkos(which,k_buf1,send[m].npack,k_send_packlist,m);
    DeviceType().fence();

    if (!lmp->kokkos->gpu_aware_flag) {
      k_buf1.modify<DeviceType>();
      k_buf1.sync<LMPHostType>();
    }

    MPI_Send(buf1,nper*send[m].npack,datatype,send[m].proc,0,gridcomm);
  }

  // perform all copies to self

  for (m = 0; m < ncopy; m++) {
    kspaceKKBase->pack_forward_grid_kokkos(which,k_buf1,copy[m].npack,k_copy_packlist,m);
    kspaceKKBase->unpack_forward_grid_kokkos(which,k_buf1,0,copy[m].nunpack,k_copy_unpacklist,m);
  }

  // unpack all received data

  for (i = 0; i < nrecv; i++) {
    MPI_Waitany(nrecv,requests,&m,MPI_STATUS_IGNORE);

    if (!lmp->kokkos->gpu_aware_flag) {
      k_buf2.modify<LMPHostType>();
      k_buf2.sync<DeviceType>();
    }

    offset = nper * recv[m].offset;
    kspaceKKBase->unpack_forward_grid_kokkos(which,k_buf2,offset,
                                recv[m].nunpack,k_recv_unpacklist,m);
    DeviceType().fence();
  }
}

/* ----------------------------------------------------------------------
   reverse comm of my ghost cells to sum to owner cells
------------------------------------------------------------------------- */

template<class DeviceType>
void GridCommKokkos<DeviceType>::reverse_comm_kspace(KSpace *kspace, int nper, int which,
                                    FFT_DAT::tdual_FFT_SCALAR_1d &k_buf1, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf2, MPI_Datatype datatype)
{
  if (layout == REGULAR)
    reverse_comm_kspace_regular(kspace,nper,which,k_buf1,k_buf2,datatype);
  else
    reverse_comm_kspace_tiled(kspace,nper,which,k_buf1,k_buf2,datatype);
}

/* ----------------------------------------------------------------------
   reverse comm on regular grid of procs via list of swaps with 6 neighbor procs
------------------------------------------------------------------------- */

template<class DeviceType>
void GridCommKokkos<DeviceType>::
reverse_comm_kspace_regular(KSpace *kspace, int nper, int which,
                            FFT_DAT::tdual_FFT_SCALAR_1d &k_buf1, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf2, MPI_Datatype datatype)
{
  int m;
  MPI_Request request;

  KokkosBaseFFT* kspaceKKBase = dynamic_cast<KokkosBaseFFT*>(kspace);
  FFT_SCALAR* buf1;
  FFT_SCALAR* buf2;
  if (lmp->kokkos->gpu_aware_flag) {
    buf1 = k_buf1.view<DeviceType>().data();
    buf2 = k_buf2.view<DeviceType>().data();
  } else {
    buf1 = k_buf1.h_view.data();
    buf2 = k_buf2.h_view.data();
  }

  for (m = nswap-1; m >= 0; m--) {
    if (swap[m].recvproc == me)
      kspaceKKBase->pack_reverse_grid_kokkos(which,k_buf2,swap[m].nunpack,k_swap_unpacklist,m);
    else
      kspaceKKBase->pack_reverse_grid_kokkos(which,k_buf1,swap[m].nunpack,k_swap_unpacklist,m);
    DeviceType().fence();

    if (swap[m].recvproc != me) {

      if (!lmp->kokkos->gpu_aware_flag) {
        k_buf1.modify<DeviceType>();
        k_buf1.sync<LMPHostType>();
      }

      if (swap[m].npack) MPI_Irecv(buf2,nper*swap[m].npack,datatype,
                                   swap[m].sendproc,0,gridcomm,&request);
      if (swap[m].nunpack) MPI_Send(buf1,nper*swap[m].nunpack,datatype,
                                     swap[m].recvproc,0,gridcomm);
      if (swap[m].npack) MPI_Wait(&request,MPI_STATUS_IGNORE);


      if (!lmp->kokkos->gpu_aware_flag) {
        k_buf2.modify<LMPHostType>();
        k_buf2.sync<DeviceType>();
      }
    }

    kspaceKKBase->unpack_reverse_grid_kokkos(which,k_buf2,0,swap[m].npack,k_swap_packlist,m);
    DeviceType().fence();
  }
}

/* ----------------------------------------------------------------------
   reverse comm on tiled grid decomp via Send/Recv lists of each neighbor proc
------------------------------------------------------------------------- */

template<class DeviceType>
void GridCommKokkos<DeviceType>::
reverse_comm_kspace_tiled(KSpace *kspace, int nper, int which,
                          FFT_DAT::tdual_FFT_SCALAR_1d &k_buf1, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf2, MPI_Datatype datatype)
{
  int i,m,offset;

  KokkosBaseFFT* kspaceKKBase = dynamic_cast<KokkosBaseFFT*>(kspace);

  FFT_SCALAR* buf1;
  FFT_SCALAR* buf2;
  if (lmp->kokkos->gpu_aware_flag) {
    buf1 = k_buf1.view<DeviceType>().data();
    buf2 = k_buf2.view<DeviceType>().data();
  } else {
    buf1 = k_buf1.h_view.data();
    buf2 = k_buf2.h_view.data();
  }

  // post all receives

  for (m = 0; m < nsend; m++) {
    offset = nper * send[m].offset;
    MPI_Irecv(&buf2[offset],nper*send[m].npack,datatype,
              send[m].proc,0,gridcomm,&requests[m]);
  }

  // perform all sends to other procs

  for (m = 0; m < nrecv; m++) {
    kspaceKKBase->pack_reverse_grid_kokkos(which,k_buf1,recv[m].nunpack,k_recv_unpacklist,m);
    DeviceType().fence();

    if (!lmp->kokkos->gpu_aware_flag) {
      k_buf1.modify<DeviceType>();
      k_buf1.sync<LMPHostType>();
    }

    MPI_Send(buf1,nper*recv[m].nunpack,datatype,recv[m].proc,0,gridcomm);
  }

  // perform all copies to self

  for (m = 0; m < ncopy; m++) {
    kspaceKKBase->pack_reverse_grid_kokkos(which,k_buf1,copy[m].nunpack,k_copy_unpacklist,m);
    kspaceKKBase->unpack_reverse_grid_kokkos(which,k_buf1,0,copy[m].npack,k_copy_packlist,m);
  }

  // unpack all received data

  for (i = 0; i < nsend; i++) {
    MPI_Waitany(nsend,requests,&m,MPI_STATUS_IGNORE);

    if (!lmp->kokkos->gpu_aware_flag) {
      k_buf2.modify<LMPHostType>();
      k_buf2.sync<DeviceType>();
    }

    offset = nper * send[m].offset;
    kspaceKKBase->unpack_reverse_grid_kokkos(which,k_buf2,offset,
                                send[m].npack,k_send_packlist,m);
    DeviceType().fence();
  }
}

/* ----------------------------------------------------------------------
   create swap stencil for grid own/ghost communication
   swaps covers all 3 dimensions and both directions
   swaps cover multiple iterations in a direction if need grid pts
     from further away than nearest-neighbor proc
   same swap list used by forward and reverse communication
------------------------------------------------------------------------- */

template<class DeviceType>
void GridCommKokkos<DeviceType>::grow_swap()
{
  maxswap += DELTA;
  swap = (Swap *)
    memory->srealloc(swap,maxswap*sizeof(Swap),"GridComm:swap");

  if (!k_swap_packlist.d_view.data()) {
    k_swap_packlist = DAT::tdual_int_2d("GridComm:swap_packlist",maxswap,k_swap_packlist.extent(1));
    k_swap_unpacklist = DAT::tdual_int_2d("GridComm:swap_unpacklist",maxswap,k_swap_unpacklist.extent(1));
  } else {
    k_swap_packlist.resize(maxswap,k_swap_packlist.extent(1));
    k_swap_unpacklist.resize(maxswap,k_swap_unpacklist.extent(1));
  }
}

/* ----------------------------------------------------------------------
   create 1d list of offsets into 3d array section (xlo:xhi,ylo:yhi,zlo:zhi)
   assume 3d array is allocated as
     (fullxlo:fullxhi,fullylo:fullyhi,fullzlo:fullzhi)
------------------------------------------------------------------------- */

template<class DeviceType>
int GridCommKokkos<DeviceType>::indices(DAT::tdual_int_2d &k_list, int index,
                       int xlo, int xhi, int ylo, int yhi, int zlo, int zhi)
{
  int nmax = (xhi-xlo+1) * (yhi-ylo+1) * (zhi-zlo+1);
  if (k_list.extent(1) < nmax)
    k_list.resize(k_list.extent(0),nmax);

  if (nmax == 0) return 0;

  int nx = (fullxhi-fullxlo+1);
  int ny = (fullyhi-fullylo+1);

  k_list.sync<LMPHostType>();

  int n = 0;
  int ix,iy,iz;
  for (iz = zlo; iz <= zhi; iz++)
    for (iy = ylo; iy <= yhi; iy++)
      for (ix = xlo; ix <= xhi; ix++)
        k_list.h_view(index,n++) = (iz-fullzlo)*ny*nx + (iy-fullylo)*nx + (ix-fullxlo);

  k_list.modify<LMPHostType>();
  k_list.sync<DeviceType>();

  return nmax;
}

namespace LAMMPS_NS {
template class GridCommKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class GridCommKokkos<LMPHostType>;
#endif
}

