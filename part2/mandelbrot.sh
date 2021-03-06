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

#Added for Block Commenting
[ -z $BASH ] || shopt -s expand_aliases
alias BEGINCOMMENT="if [ ]; then"
alias ENDCOMMENT="fi"


# Module load boost
module load boost/1.57.0

# Module load OpenMPI
module load openmpi-1.8.3/gcc-4.9.2

echo "*** Trial Serial  ***"
####### Serial ############
for trial in 1 2 3 4 5 ; do
  ./mandelbrot_serial 500 500
done


echo "*** MS  ***"
######### MS ############
echo "*** MS 2  ***"
for trial in 1 2 3 4 5 ; do
  mpirun -np 2 ./mandelbrot_ms 500 500
done
echo "*** MS 8  ***"
for trial in 1 2 3 4 5 ; do
  mpirun -np 8 ./mandelbrot_ms 500 500
done
echo "*** MS 32  ***"
for trial in 1 2 3 4 5 ; do
  mpirun -np 32 ./mandelbrot_ms 500 500
done

echo "*** Joe  ***"
######### Joe ############
echo "*** Joe 2 ***"
for trial in 1 2 3 4 5 ; do
  mpirun -np 2 ./mandelbrot_joe 500 500
done
echo "*** Joe 8 ***"
for trial in 1 2 3 4 5 ; do
  mpirun -np 8 ./mandelbrot_joe 500 500
done
echo "*** Joe 32 ***"
for trial in 1 2 3 4 5 ; do
  mpirun -np 32 ./mandelbrot_joe 500 500
done

echo "*** Susie  ***"
######### Susie ############
echo "*** Susie  2 ***"
for trial in 1 2 3 4 5 ; do
  mpirun -np 2 ./mandelbrot_susie 500 500
done
echo "*** Susie 8 ***"
for trial in 1 2 3 4 5 ; do
  mpirun -np 8 ./mandelbrot_susie 500 500
done
echo "*** Susie  32 ***"
for trial in 1 2 3 4 5 ; do
  mpirun -np 32 ./mandelbrot_susie 500 500
done

