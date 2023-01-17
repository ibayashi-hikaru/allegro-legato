#!/bin/bash

tempi=0.0
tempf=20.0

rm res_*.dat

# compute Lammps 
N=20
for (( i=0; i<$N; i++ ))
do
  temp="$(echo "$tempi+$i*($tempf-$tempi)/$N" | bc -l)"
  sed s/temperature/${temp}/g test-prec-spin.template > \
    test-prec-spin.in
  
  # test standard Lammps 
  ./../../../../src/lmp_serial -in test-prec-spin.in 

  # test spin/kk with Kokkos Lammps
  # mpirun -np 1 ../../../../src/lmp_kokkos_mpi_only \
  #   -k on -sf kk -in test-prec-spin.in
  
  Hz="$(tail -n 1 average_spin | awk -F " " '{print $3}')"
  sz="$(tail -n 1 average_spin | awk -F " " '{print $5}')"
  en="$(tail -n 1 average_spin | awk -F " " '{print $6}')"
  echo $temp $Hz $sz $en >> res_lammps.dat
done

# compute Langevin
python3 langevin.py > res_langevin.dat

# plot results
python3 plot_precession.py res_lammps.dat res_langevin.dat
