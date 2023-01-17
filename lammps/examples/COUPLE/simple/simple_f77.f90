!  LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
!  www.cs.sandia.gov/~sjplimp/lammps.html
!  Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories
!
!  Copyright (2003) Sandia Corporation.  Under the terms of Contract
!  DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
!  certain rights in this software.  This software is distributed under 
!  the GNU General Public License.
!
!  See the README file in the top-level LAMMPS directory.

! f_driver = simple example of how an umbrella program
!            can invoke LAMMPS as a library on some subset of procs
! Syntax: simpleF P in.lammps
!         P = # of procs to run LAMMPS on
!             must be <= # of procs the driver code itself runs on
!         in.lammps = LAMMPS input script
!   See README for compilation instructions

PROGRAM f_driver
  IMPLICIT NONE
  INCLUDE 'mpif.h'

  INTEGER, PARAMETER :: fp=20
  INTEGER :: n, narg, ierr, me, nprocs, natoms
  INTEGER :: lammps, nprocs_lammps, comm_lammps
  INTEGER (kind=8) :: ptr

  REAL (kind=8), ALLOCATABLE :: x(:)
  REAL (kind=8), PARAMETER   :: epsilon=0.1

  CHARACTER (len=64)   :: arg
  CHARACTER (len=1024) :: line

  ! setup MPI and various communicators
  ! driver runs on all procs in MPI_COMM_WORLD
  ! comm_lammps only has 1st P procs (could be all or any subset)

  CALL mpi_init(ierr)

  narg = command_argument_count()

  IF (narg /= 2) THEN
     PRINT *, 'Syntax: simpleF P in.lammps'
     CALL mpi_abort(MPI_COMM_WORLD,1,ierr)
  END IF

  CALL mpi_comm_rank(MPI_COMM_WORLD,me,ierr);
  CALL mpi_comm_size(MPI_COMM_WORLD,nprocs,ierr);

  CALL get_command_argument(1,arg)
  READ (arg,'(I10)') nprocs_lammps

  IF (nprocs_lammps > nprocs) THEN
     IF (me == 0) THEN
        PRINT *, 'ERROR: LAMMPS cannot use more procs than available'
        CALL mpi_abort(MPI_COMM_WORLD,2,ierr)
     END IF
  END IF

  lammps = 0
  IF (me < nprocs_lammps) THEN
     lammps = 1
  ELSE
     lammps = MPI_UNDEFINED
  END IF

  CALL mpi_comm_split(MPI_COMM_WORLD,lammps,0,comm_lammps,ierr)

  ! open LAMMPS input script on rank zero

  CALL get_command_argument(2,arg)
  OPEN(UNIT=fp, FILE=arg, ACTION='READ', STATUS='OLD', IOSTAT=ierr)
  IF (ierr /= 0) THEN
     PRINT *, 'ERROR: Could not open LAMMPS input script'
     CALL mpi_abort(MPI_COMM_WORLD,3,ierr);
  END IF

  ! run the input script thru LAMMPS one line at a time until end-of-file
  ! driver proc 0 reads a line, Bcasts it to all procs
  ! (could just send it to proc 0 of comm_lammps and let it Bcast)
  ! all LAMMPS procs call lammps_command() on the line */

  IF (lammps == 1) CALL lammps_open(comm_lammps,ptr)

  n = 0
  DO
     IF (me == 0) THEN
        READ (UNIT=fp, FMT='(A)', IOSTAT=ierr) line
        n = 0
        IF (ierr == 0) THEN
           n = LEN(TRIM(line))
           IF (n == 0 ) THEN
              line = ' '
              n = 1
           END IF
        END IF
     END IF
     CALL mpi_bcast(n,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
     IF (n == 0) EXIT
     CALL mpi_bcast(line,n,MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
     IF (lammps == 1) CALL lammps_command(ptr,line,n)
  END DO
  CLOSE(UNIT=fp)

  ! run 10 more steps followed by a single step */

  IF (lammps == 1) THEN
     CALL lammps_command(ptr,'run 10',6)

     CALL lammps_get_natoms(ptr,natoms)
     print*,'natoms=',natoms

     CALL lammps_command(ptr,'run 1',5);
  END IF

  ! free LAMMPS object

  IF (lammps == 1) CALL lammps_close(ptr);

  ! close down MPI

  CALL mpi_finalize(ierr)

END PROGRAM f_driver
