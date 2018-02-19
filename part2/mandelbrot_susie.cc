/**
 *  \file mandelbrot_susie.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <cassert>
#include "render.hh"
#include "timer.c"
#define ROOT 0

int
mandelbrot(double x, double y) {
  int maxit = 511;
  double cx = x;
  double cy = y;
  double newx, newy;

  int it = 0;
  for (it = 0; it < maxit && (x*x + y*y) < 4; ++it) {
    newx = x*x - y*y + cx;
    newy = 2*x*y + cy;
    x = newx;
    y = newy;
  }
  return it;
}

int
main (int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, & world_rank);	

	double minX = -2.1;
  double maxX = 0.7;
  double minY = -1.25;
  double maxY = 1.25;
  
  int height, width;
  if (argc == 3) {
    height = atoi (argv[1]);
    width = atoi (argv[2]);
    assert (height > 0 && width > 0);
  } else {
    fprintf (stderr, "usage: %s <height> <width>\n", argv[0]);
    fprintf (stderr, "where <height> and <width> are the dimensions of the image.\n");
    return -1;
  }

  double it = (maxY - minY)/height;
  double jt = (maxX - minX)/width;
  double x, y;

  gil::rgb8_image_t img(height, width);
  auto img_view = gil::view(img);

	float * localMandelbrot = (float *)malloc(sizeof(float) * width);
	float * finalMandelbrot = NULL;	
	

	if(world_rank == ROOT)
		finalMandelbrot = (float *)malloc(sizeof(float) * height * width);			
		
	//account rows not being perfectly divisible by processors		
	int dif = height % world_size;
	height -= dif;				
	
	y = minY;
	int nextWidth = 0;
	
	stopwatch_init ();
  struct stopwatch_t* timer = stopwatch_create (); assert (timer);
	stopwatch_start (timer);
	
	for(int i = world_rank; i < height; i += world_size){
		x = minX;
		nextWidth = i * width;
		for(int j = 0; j < width; ++j) {
			localMandelbrot[j] = mandelbrot(x,  y + it*i) / 512.0;
			//printf("First : for loop = %i, [%i] = %f\n",
			//			i, world_rank, localMandelbrot[j]);
			x += jt;
		}	
		MPI_Barrier(MPI_COMM_WORLD);	
		MPI_Gather(localMandelbrot, width, MPI_FLOAT, 
						finalMandelbrot + nextWidth, width,
				 		MPI_FLOAT, ROOT, MPI_COMM_WORLD);		
	} 
	//render results so far
		int otherIter = 0;
		if(world_rank == ROOT){
			for(int i = 0; i < height; ++i) {
				for(int j = 0; j < width; ++j) {
					img_view(j, i) = render(finalMandelbrot[otherIter++]);
				}
			}	
		}	
	
	//compute and render leftover rows if necessary
	if(world_rank == ROOT && dif > 0) {	
		int leftover = height;
		for (int i = height; i < height + dif; ++i) {
			y = minY + it * leftover;			
		  x = minX;
		  for (int j = 0; j < width; ++j) {
		    img_view(j, i) = render(mandelbrot(x, y)/512.0);
				//printf("%f\n",(mandelbrot(x, y)/512.0));
		    x += jt;
		  }
			leftover++;
  	}
	}
  gil::png_write_view("mandelbrot.png", const_view(img));

	long double tFinal = stopwatch_stop (timer);

	MPI_Barrier(MPI_COMM_WORLD);
	if(world_rank == 0){
		printf("Susie Time = %Lg\n", tFinal);
		free(finalMandelbrot);
	}
	free(localMandelbrot);
	stopwatch_destroy (timer);
	MPI_Finalize();
}

/* eof */
