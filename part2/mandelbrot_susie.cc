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
	
	if(world_rank == 0)
		finalMandelbrot = (float *)malloc(sizeof(float) * height * width);			

	//account rows not being perfectly divisible by processors		
	y = minY;
	int dif = height % world_size;
	height -= dif;				
	printf("dif = %i new height = %i \n", dif, height);
	
	for(int i = world_rank; i < height; i += world_size){
		x = minX;
		for(int j = 0; j < width; ++j) {
			localMandelbrot[j] = mandelbrot(x,  y + it*i) / 512.0;
	//		printf("for loop = %i, [%i] = %f\n",i, world_rank, localMandelbrot[j]);
			x += jt;
		}
		MPI_Barrier(MPI_COMM_WORLD);	
		MPI_Gather(	localMandelbrot, width, MPI_FLOAT, finalMandelbrot,
								width, MPI_FLOAT, 0, MPI_COMM_WORLD);
			
	//	if(world_rank == 0){
	//		for(int  j = 0; j < width; ++j)
		//		printf("%f\n",finalMandelbrot[j]);
	//	}
	//	printf("my rank %i\n",world_rank);
	//	printf("%i\n",world_size);
	}
	/*if(world_rank == 1){
		for(int j = 0; j < width; ++j) 
		printf("1 better spit! : %f\n",localMandelbrot[j]); 
	}*/

	//get leftovers if necessary	
	if(world_rank < dif){
		for(int i = world_rank; i < dif; i++){
			x = minX;
			for(int j = 0; j < width; ++j) {
				localMandelbrot[j] = mandelbrot(x,y + it *(world_rank + world_size)) / 512.0;
		//		printf("for loop = %i, [%i] = %f\n",i+world_size, world_rank, localMandelbrot[j]);
				x += jt;	
			}
			//MPI_Barrier(MPI_COMM_WORLD);	
			//MPI_Send(localMandelbrot, width, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
			//MPI_Recv(finalMandelbrot, width, MPI_FLOAT, world_rank, 1, MPI_COMM_WORLD, 						MPI_STATUS_IGNORE);
		//MPI_Barrier(MPI_COMM_WORLD);	
		MPI_Gather(localMandelbrot, width, MPI_FLOAT, finalMandelbrot,
									width, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		height += dif;
	}

	if(world_rank == 0){
		int iter = 0;
		for(int i= 0; i < height; i++){
			for(int j = 0; j < width; j++){
				printf("%f\n",(float)finalMandelbrot[iter]);
				img_view(i, j) = render((float)finalMandelbrot[iter++]);				
			}
		}	
	}		

  gil::png_write_view("mandelbrot.png", const_view(img));

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

/* eof */
