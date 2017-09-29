#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "mkl_vsl.h"

int main(int argc, char **argv)
{

   int niter = atoi(argv[1]);
   
   int numThreads=1; 

   int niterThread=1;

   int totalCount=0;

   
   /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
   /* ~~~~~~~~~~~~~~~~~PARALLELIZE ME~~~~~~~~~~~~~~~~~~ */
   /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

   /* Random Number Generation via MKL Random Number Stream */  
   

   #pragma omp parallel firstprivate (numThreads, niterThread) shared(totalCount)
   {
   numThreads=omp_get_num_threads();
   niterThread=niter/numThreads;
   
   float * rand_buffer_x = (float*)malloc(niterThread*sizeof(float)); 
   float * rand_buffer_y = (float*)malloc(niterThread*sizeof(float)); 
   VSLStreamStatePtr stream;
   int SEED = omp_get_thread_num() + time(NULL); /* RNG seed dependent on time of day and thread number*/
   
   vslNewStream( &stream, VSL_BRNG_SFMT19937, SEED );
   vsRngUniform( VSL_RNG_METHOD_UNIFORM_STD, stream, niterThread, rand_buffer_x, 0.0, 1.0 );
   vsRngUniform( VSL_RNG_METHOD_UNIFORM_STD, stream, niterThread, rand_buffer_y, 0.0, 1.0 );
   
   /* Loop over Randomly Generated Numbers */
   
   int count = 0;
   
   int i;

   #pragma omp parallel for reduction (+:count)
 
   for(i = 0; i < niterThread; i++)
   {
   float z = rand_buffer_y[i]*rand_buffer_y[i] + rand_buffer_x[i]*rand_buffer_x[i];
      if (z<=1) count++;
   }
   
   #pragma omp critical
   {
	   totalCount+=count;
   }
  
   /* Delete the stream and buffers*/        
   
   vslDeleteStream( &stream );
   free(rand_buffer_x);
   free(rand_buffer_y);
   
   /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
   /* ~~~~~~~~~~~~~~~~~PARALLELIZE ME~~~~~~~~~~~~~~~~~~ */
   /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
   

   
   
   /*  
   double pi=(double) count/ niterThread*4;
   printf("num trials=%d, estimate of pi=%f\n", niter, pi);
   */
   } 
   
   double pi=(double) totalCount/ niter*4;
   printf("num trials=%d, estimate of pi=%f\n", niter, pi);

   
   return 0;
  
}
