#!/bin/bash
#$ -N Mandelbrot
#$ -q class8i
#$ -pe mpi 48 
#$ -R y

# pe one-node-mpi 1
# Grid Engine Notes:
# -----------------
# 1) Use "-R y" to request job reservation otherwise single 1-core jobs
#    may prevent this multicore MPI job from running.   This is called
#    job starvation.

# Module load boost
module load boost/1.57.0

# Module load OpenMPI
module load openmpi-1.8.3/gcc-4.9.2

####### Run the program 48 Cores ############
for trial in 1 2 3 4 5 ; do
  echo "*** Trial Serial ${trial} ***"
  mpirun -np 1  ./mandelbrot_serial 1000 1000
done

for trial in 1 2 3 4 5 ; do
  echo "*** Trial (48 Cores) ${trial} ***"
  mpirun -np 6 ./mandelbrot_ms 1000 1000
done

for trial in 1 2 3 4 5 ; do
  echo "*** Trial Susie (48 Cores) ${trial} ***"
  mpirun -np 6 ./mandelbrot_susie 1000 1000
done

for trial in 1 2 3 4 5 ; do
  echo "*** Trial Joe (48 Cores) ${trial} ***"
  mpirun -np 6 ./mandelbrot_joe 1000 1000
done

####### Run the program 56 Cores ############
#for trial in 1 2 3 4 5 ; do
#  echo "*** Trial Serial ${trial} ***"
#  mpirun -np 1  ./mandelbrot_serial 1000 1000
#done

#for trial in 1 2 3 4 5 ; do
#  echo "*** Trial (56 Cores) ${trial} ***"
#  mpirun -np 2 ./mandelbrot_ms 1000 1000
#done

#for trial in 1 2 3 4 5 ; do
#  echo "*** Trial Susie (56 Cores) ${trial} ***"
#  mpirun -np 2 ./mandelbrot_susie 1000 1000
#done

#for trial in 1 2 3 4 5 ; do
#  echo "*** Trial Joe (56 Cores) ${trial} ***"
#  mpirun -np 2 ./mandelbrot_joe 1000 1000
#done

####### Run the program 64 Cores ############
#for trial in 1 2 3 4 5 ; do
#  echo "*** Trial Serial ${trial} ***"
#  mpirun -np 1  ./mandelbrot_serial 1000 1000
#done

#for trial in 1 2 3 4 5 ; do
#  echo "*** Trial (64 Cores) ${trial} ***"
#  mpirun -np 8 ./mandelbrot_ms 1000 1000
#done

#for trial in 1 2 3 4 5 ; do
#  echo "*** Trial Susie (64 Cores) ${trial} ***"
#  mpirun -np 8 ./mandelbrot_susie 1000 1000
#done

#for trial in 1 2 3 4 5 ; do
#  echo "*** Trial Joe (64 Cores) ${trial} ***"
#  mpirun -np 8 ./mandelbrot_joe 1000 1000
#done


