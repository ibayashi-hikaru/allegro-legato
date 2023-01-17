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

#include "compute_property_chunk.h"

#include <cstring>
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "compute_chunk_atom.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputePropertyChunk::ComputePropertyChunk(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  idchunk(nullptr), count_one(nullptr), count_all(nullptr)
{
  if (narg < 5) error->all(FLERR,"Illegal compute property/chunk command");

  // ID of compute chunk/atom

  idchunk = utils::strdup(arg[3]);

  ComputePropertyChunk::init();

  // parse values

  nvalues = narg - 4;
  pack_choice = new FnPtrPack[nvalues];
  countflag = 0;

  int i;
  for (int iarg = 4; iarg < narg; iarg++) {
    i = iarg-4;

    if (strcmp(arg[iarg],"count") == 0) {
      pack_choice[i] = &ComputePropertyChunk::pack_count;
      countflag = 1;
    } else if (strcmp(arg[iarg],"id") == 0) {
      if (!cchunk->compress)
        error->all(FLERR,"Compute chunk/atom stores no IDs for "
                   "compute property/chunk");
      pack_choice[i] = &ComputePropertyChunk::pack_id;
    } else if (strcmp(arg[iarg],"coord1") == 0) {
      if (cchunk->ncoord < 1)
        error->all(FLERR,"Compute chunk/atom stores no coord1 for "
                   "compute property/chunk");
      pack_choice[i] = &ComputePropertyChunk::pack_coord1;
    } else if (strcmp(arg[iarg],"coord2") == 0) {
      if (cchunk->ncoord < 2)
        error->all(FLERR,"Compute chunk/atom stores no coord2 for "
                   "compute property/chunk");
      pack_choice[i] = &ComputePropertyChunk::pack_coord2;
    } else if (strcmp(arg[iarg],"coord3") == 0) {
      if (cchunk->ncoord < 3)
        error->all(FLERR,"Compute chunk/atom stores no coord3 for "
                   "compute property/chunk");
      pack_choice[i] = &ComputePropertyChunk::pack_coord3;
    } else error->all(FLERR,
                    "Invalid keyword in compute property/chunk command");
  }

  // initialization

  nchunk = 1;
  maxchunk = 0;
  allocate();

  if (nvalues == 1) {
    vector_flag = 1;
    size_vector = 0;
    size_vector_variable = 1;
    extvector = 0;
  } else {
    array_flag = 1;
    size_array_cols = nvalues;
    size_array_rows = 0;
    size_array_rows_variable = 1;
    extarray = 0;
  }
}

/* ---------------------------------------------------------------------- */

ComputePropertyChunk::~ComputePropertyChunk()
{
  delete [] idchunk;
  delete [] pack_choice;
  memory->destroy(vector);
  memory->destroy(array);
  memory->destroy(count_one);
  memory->destroy(count_all);
}

/* ---------------------------------------------------------------------- */

void ComputePropertyChunk::init()
{
  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for "
               "compute property/chunk");
  cchunk = (ComputeChunkAtom *) modify->compute[icompute];
  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"Compute property/chunk does not use chunk/atom compute");
}

/* ---------------------------------------------------------------------- */

void ComputePropertyChunk::compute_vector()
{
  invoked_vector = update->ntimestep;

  // compute chunk/atom assigns atoms to chunk IDs
  // if need count, extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms

  nchunk = cchunk->setup_chunks();
  if (nchunk > maxchunk) allocate();
  if (nvalues == 1) size_vector = nchunk;
  else size_array_rows = nchunk;

  if (countflag) {
    cchunk->compute_ichunk();
    ichunk = cchunk->ichunk;
  }

  // fill vector

  buf = vector;
  (this->*pack_choice[0])(0);
}

/* ---------------------------------------------------------------------- */

void ComputePropertyChunk::compute_array()
{
  invoked_array = update->ntimestep;

  // compute chunk/atom assigns atoms to chunk IDs
  // if need count, extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms

  nchunk = cchunk->setup_chunks();
  if (nchunk > maxchunk) allocate();
  if (nvalues == 1) size_vector = nchunk;
  else size_array_rows = nchunk;

  if (countflag) {
    cchunk->compute_ichunk();
    ichunk = cchunk->ichunk;
  }

  // fill array

  if (array) buf = &array[0][0];
  for (int n = 0; n < nvalues; n++)
    (this->*pack_choice[n])(n);
}

/* ----------------------------------------------------------------------
   lock methods: called by fix ave/time
   these methods insure vector/array size is locked for Nfreq epoch
     by passing lock info along to compute chunk/atom
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   increment lock counter
------------------------------------------------------------------------- */

void ComputePropertyChunk::lock_enable()
{
  cchunk->lockcount++;
}

/* ----------------------------------------------------------------------
   decrement lock counter in compute chunk/atom, it if still exists
------------------------------------------------------------------------- */

void ComputePropertyChunk::lock_disable()
{
  int icompute = modify->find_compute(idchunk);
  if (icompute >= 0) {
    cchunk = (ComputeChunkAtom *) modify->compute[icompute];
    cchunk->lockcount--;
  }
}

/* ----------------------------------------------------------------------
   calculate and return # of chunks = length of vector/array
------------------------------------------------------------------------- */

int ComputePropertyChunk::lock_length()
{
  nchunk = cchunk->setup_chunks();
  return nchunk;
}

/* ----------------------------------------------------------------------
   set the lock from startstep to stopstep
------------------------------------------------------------------------- */

void ComputePropertyChunk::lock(Fix *fixptr, bigint startstep, bigint stopstep)
{
  cchunk->lock(fixptr,startstep,stopstep);
}

/* ----------------------------------------------------------------------
   unset the lock
------------------------------------------------------------------------- */

void ComputePropertyChunk::unlock(Fix *fixptr)
{
  cchunk->unlock(fixptr);
}

/* ----------------------------------------------------------------------
   free and reallocate per-chunk arrays
------------------------------------------------------------------------- */

void ComputePropertyChunk::allocate()
{
  memory->destroy(vector);
  memory->destroy(array);
  memory->destroy(count_one);
  memory->destroy(count_all);
  maxchunk = nchunk;
  if (nvalues == 1) memory->create(vector,maxchunk,"property/chunk:vector");
  else memory->create(array,maxchunk,nvalues,"property/chunk:array");
  if (countflag) {
    memory->create(count_one,maxchunk,"property/chunk:count_one");
    memory->create(count_all,maxchunk,"property/chunk:count_all");
  }
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputePropertyChunk::memory_usage()
{
  double bytes = (bigint) nchunk * nvalues * sizeof(double);
  if (countflag) bytes += (double) nchunk * 2 * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   one method for every keyword compute property/chunk can output
   the property is packed into buf starting at n with stride nvalues
   customize a new keyword by adding a method
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void ComputePropertyChunk::pack_count(int n)
{
  int index;

  for (int m = 0; m < nchunk; m++) count_one[m] = 0;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      count_one[index]++;
    }
  }

  MPI_Allreduce(count_one,count_all,nchunk,MPI_INT,MPI_SUM,world);

  for (int m = 0; m < nchunk; m++) {
    buf[n] = count_all[m];
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyChunk::pack_id(int n)
{
  int *origID = cchunk->chunkID;
  for (int m = 0; m < nchunk; m++) {
    buf[n] = origID[m];
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyChunk::pack_coord1(int n)
{
  double **coord = cchunk->coord;
  for (int m = 0; m < nchunk; m++) {
    buf[n] = coord[m][0];
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyChunk::pack_coord2(int n)
{
  double **coord = cchunk->coord;
  for (int m = 0; m < nchunk; m++) {
    buf[n] = coord[m][1];
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyChunk::pack_coord3(int n)
{
  double **coord = cchunk->coord;
  for (int m = 0; m < nchunk; m++) {
    buf[n] = coord[m][2];
    n += nvalues;
  }
}
