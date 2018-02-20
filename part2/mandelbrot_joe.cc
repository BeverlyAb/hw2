/**
 *  \file mandelbrot_serial.cc
 *  \brief Lab 2: Mandelbrot set serial code
 */


#include <iostream>
#include <cstdlib>
#include <cmath>
#include "render.hh"
#include <vector>
#include <mpi.h>
#include <assert.h>

using namespace std;

#define WIDTH 1000
#define HEIGHT 1000

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
void compute_row(int p,int width,Color_Point *img,int N,double minX,double minY,double it,double jt)
{
	#ifdef DEBUG	
	printf("Computing for rank=%d\n",p);
	#endif
	double y=minY + p*N*it;
	int curr_index = 0;
  	for (int i = p*N; i < N*(p+1); ++i) 
  	{	
    		double x = minX;
    		for (int j = 0; j < width; ++j) 
		{
      			
			curr_index= (i-p*N)*width + j;
      			Color_Point t = render_color(mandelbrot(x,y)/512.0);
			img[curr_index].r = t.r;
			img[curr_index].g = t.g;
			img[curr_index].b = t.b;

			#ifdef DEBUG_EXTENDED
			
			int curr_index= (i-p*N)*width + j;
			printf("CI:%d r:%d ir:%d",curr_index,t.r,img[curr_index].r);		
			#endif
			x += jt;
    		}
    		y += it;
		
		#ifdef DEBUG
		printf("\n");
		#endif
  	}
	#ifdef DEBUG_EXTENDED
	printf("/n Now printing in compute after computation loop\n");
	for(int i=0;i<N;i++)
	{
		for(int j= 0;j<width;j++)
		{
			curr_index= i*width + j;
			printf("CI:%d r:%d ",curr_index,img[curr_index].r);		
		}
		printf("\n");
	}
	#endif 


}
int
main(int argc, char* argv[]) {
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

  y = minY;
 
 
 MPI_Init(NULL, NULL);

  MPI_Datatype dt_point;
  MPI_Type_contiguous(3, MPI_INT, &dt_point);
  MPI_Type_commit(&dt_point);
   int world_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

   int world_size;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   int N = ceil(float(height)/float(world_size)); //world = number of processors , this is to find num of rows per processor

   // Create a random array of elements on the root process. Its total
   // size will be the number of elements per process times the number
   // of processes

  int num_elements_per_proc = N*width;


   Color_Point *all_rows = NULL;
   if (world_rank == 0) {
     all_rows = (Color_Point *)malloc(sizeof(Color_Point) * num_elements_per_proc*world_size);
	
	#ifdef DEBUG	
	printf("World size: %d\n",world_size);   
	#endif 
}



   Color_Point *row_blocks = (Color_Point *)malloc(sizeof(Color_Point) * num_elements_per_proc);
   assert(row_blocks!=NULL);
   //Computing for all blocks of rows
   double t_start, t_elapsed;  
   MPI_Barrier (MPI_COMM_WORLD); 
   t_start = MPI_Wtime();
  compute_row(world_rank,width,row_blocks,N,minX,minY,it,jt);




	#ifdef DEBUG	
  printf("\n-------Compute Results-----------\n");
  printf("my_rank: %d\n",world_rank);
	printf("Gathering values\n");
	printf("My values");
	for(int i=0;i<N;i++)
	{
		for(int j= 0;j<width;j++)
		{
			
			int curr_index= i*width + j;
			printf(" oCI:%d r:%d   ",curr_index,row_blocks[curr_index].r);		
		}
		printf("\n");
	}
	#endif
	

   MPI_Gather(row_blocks, num_elements_per_proc,dt_point , all_rows, num_elements_per_proc, dt_point, 0, MPI_COMM_WORLD);
   
if (world_rank == 0) {
  	gil::rgb8_image_t img(height, width);
  	auto img_view = gil::view(img);

     for(int i=0;i<height;i++)
     {
       for(int j=0;j<width;j++)
       {
         int r =  all_rows[i*width+j].r;
         int g =  all_rows[i*width+j].g;
         int b =  all_rows[i*width+j].b;
	int curr_index = i*width + j;
	#ifdef DEBUG
	printf(" rCI:%d r:%d   ",curr_index,r);		
       #endif
	  img_view(j, i) = gil::rgb8_pixel_t(r,g,b);//This was removed for checking rest of code
      
	}
     }
     gil::png_write_view("mandelbrot.png", const_view(img));
   }

  
  MPI_Barrier (MPI_COMM_WORLD); 
  t_elapsed = MPI_Wtime () - t_start; /* Stop timer */
   // Clean up
   if (world_rank == 0) {
    printf("%0.3lf\n",t_elapsed); 
    free(all_rows);
   }
   free(row_blocks);

   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();

}

/* eof */
