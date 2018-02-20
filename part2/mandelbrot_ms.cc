/**
 *  \file mandelbrot_serial.cc
 *  \brief Lab 2: Mandelbrot master-slave code
 */

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include "render.hh"

using namespace std;

#define MASTER 0

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

void 
compute_single_row(int row, int width, float * in, double minX, double minY,
									double it, double jt){
	double y = minY + row * it;
	double x = minX;
	for(int j = 0; j < width; j++){
		in[j] = mandelbrot(x, y) / 512.0;
		x += jt;	
	} 
}

int
main(int argc, char* argv[]) 
{
	MPI_Init(&argc, &argv);
	
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, & world_rank);	
	MPI_Status status;
	  
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
	
	int localIndx = 0;
	int picIndx = 0;
	int nextIndx = 0;
	float * localMandelbrot = (float *)malloc(sizeof(float) * width);
	bool trueLast = true;
	
	double startTime = MPI_Wtime();

	if(world_rank == MASTER){
		//assign init tasks
		for(picIndx = 0; picIndx < world_size - 1; picIndx++){
			int procRank = picIndx + 1;
			MPI_Send(&picIndx, 1, MPI_INT, procRank, MASTER, MPI_COMM_WORLD);
		}

		//recv both notifies proc is ready for more and returns computed work
		for(picIndx = 0; picIndx < height; picIndx++){
			MPI_Recv(localMandelbrot, width, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, 
							MPI_COMM_WORLD, &status);
			
			int localIndx = status.MPI_TAG;			
			int procRank = status.MPI_SOURCE;

			//update final		
			for (int j = 0; j < width; ++j) 
   			img_view(j, localIndx) = render(localMandelbrot[j]);

			if((picIndx + world_size -1) < height)
				nextIndx = picIndx +  world_size -1;
			else 
				nextIndx = -1;
			
			//send task based on procRank 
			MPI_Send(&nextIndx, 1, MPI_INT, procRank, MASTER, MPI_COMM_WORLD);
		}
	} else { //slaves
		while(1){
			MPI_Recv(&picIndx, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,	MPI_COMM_WORLD, &status);  
			if(picIndx < 0) { // done with all rows, time to terminate
			//	printf("%i is done\n", world_rank);					
				break;
			}
			compute_single_row(picIndx, width, localMandelbrot, minX, minY,it, jt);	
			MPI_Send(localMandelbrot, width, MPI_FLOAT, MASTER, picIndx, MPI_COMM_WORLD);
		}
	}

  gil::png_write_view("mandelbrot.png", const_view(img));

	MPI_Barrier(MPI_COMM_WORLD);
	double total = MPI_Wtime() - startTime;
	if(world_rank == 0){
		printf("%0.3lf\n", total);
	}
	free(localMandelbrot);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

/* eof */
