#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define DEFAULT_H 1000
#define DEFAULT_W 1000
#define DEFAULT_CNT 200
#define DEFAULT_FILENAME "output/dz_julia"
 
int main (int argc, char *argv[]);
unsigned char *julia_set ( int w, int h, int cnt, float xl, float xr, float yb, float yt );
unsigned char *julia_set_parallel ( int w, int h, int cnt, float xl, float xr, float yb, float yt );
int julia ( int w, int h, float xl, float xr, float yb, float yt, int i, int j, int cnt );
int check_validity( unsigned char *rgb_sequential, unsigned char *rgb_parallel, int w, int h );
void tga_write ( int w, int h, unsigned char rgb[], char *filename );
void timestamp ( );

int main (int argc, char *argv[] )  {
  int h = DEFAULT_H;
  int w = DEFAULT_W;
  int cnt = DEFAULT_CNT;
  char filename[256] = DEFAULT_FILENAME; 
  char buffer[256];
  unsigned char *rgb_sequential, *rgb_parallel;
  double t1;
  double t2;
  float xl = - 1.5;
  float xr = + 1.5;
  float yb = - 1.5;
  float yt = + 1.5;

  if (argc == 4) {
  h = atoi(argv[1]);
  w = atoi(argv[2]);
  cnt = atoi(argv[3]);
  if (!h || !w || !cnt) return 1;
  }
  
  strcat(filename, "_");
  sprintf(buffer, "%d", h);
  strcat(filename, buffer);
  strcat(filename, "_");
  sprintf(buffer, "%d", w);
  strcat(filename, buffer);
  strcat(filename, "_");
  sprintf(buffer, "%d", cnt);
  strcat(filename, buffer);
  strcat(filename, ".tga");
    
  // timestamp();
  printf ( "\n" );
  printf ( "JULIA Set\n" );
  printf ( "  Plot a version of the Julia set for Z(k+1)=Z(k)^2-0.8+0.156i\n" );

  double time1_sequential, time2_sequential, time1_parallel, time2_parallel;
  time1_sequential = omp_get_wtime();
  rgb_sequential = julia_set ( w, h, cnt, xl, xr, yb, yt );
  time2_sequential = omp_get_wtime();

  time1_parallel = omp_get_wtime();
  rgb_parallel = julia_set_parallel( w, h, cnt, xl, xr, yb, yt);
  time2_parallel = omp_get_wtime();
  printf("Elapsed time, sequential: %2.2f \n", time2_sequential - time1_sequential);
  printf("Elapsed time, parallel: %2.2f \n", time2_parallel - time1_parallel);

  if(check_validity(rgb_sequential, rgb_parallel, w, h))
  {
    printf("Test PASSED\n");
  }
  else
  {
    printf("Test FAILED\n");
  }

  tga_write ( w, h, rgb_parallel, filename );

  free ( rgb_sequential );
  free ( rgb_parallel );

  // printf ( "\n" );
  // printf ( "JULIA set:\n" );
  // printf ( "  Normal end of execution.\n" );

  // timestamp();
  
  
  return 0;
}

unsigned char *julia_set ( int w, int h, int cnt, float xl, float xr, float yb, float yt )
{
  int i;
  int j;
  int juliaValue;
  int k;
  unsigned char *rgb;

  rgb = ( unsigned char * ) malloc ( w * h * 3 * sizeof ( unsigned char ) );

  for ( j = 0; j < h; j++ )
  {
    for ( i = 0; i < w; i++ )
    {
    juliaValue = julia ( w, h, xl, xr, yb, yt, i, j, cnt );

    k = 3 * ( j * w + i );

    rgb[k]   = 255 * ( 1 - juliaValue );
    rgb[k+1] = 255 * ( 1 - juliaValue );
    rgb[k+2] = 255;
    }
  }
 

  return rgb;
}

unsigned char *julia_set_parallel (int w, int h, int cnt, float xl, float xr, float yb, float yt )
{
  int i;
  int j;
  int juliaValue;
  int k;
  unsigned char *rgb;
  int myId, numThreads;
  int start, end, chunk, cyclicChunk, offset;

  rgb = ( unsigned char * ) malloc ( w * h * 3 * sizeof ( unsigned char ) );

#pragma omp parallel default(none) \
private(i, j, juliaValue, k, myId, start, end, offset) \
shared(rgb) \
firstprivate(w, h, cnt, xl, xr, yb, yt, numThreads, chunk, cyclicChunk) 
{
  myId = omp_get_thread_num();
  numThreads = omp_get_num_threads();
  chunk = (h + numThreads - 1) / numThreads / 10;
  cyclicChunk = chunk * numThreads;
  start = myId * chunk;
  end = (start + chunk) < h ? start + chunk : h;
  
  for(offset = 0; offset < h; offset += cyclicChunk)
  {
    start = offset + myId * chunk;
    end = (start + chunk) < h ? start + chunk : h;
    for ( j = start; j < end; j++ )
    {
      for ( i = 0; i < w; i++ )
      {
        juliaValue = julia ( w, h, xl, xr, yb, yt, i, j, cnt );

        k = 3 * ( j * w + i );

        rgb[k]   = 255 * ( 1 - juliaValue );
        rgb[k+1] = 255 * ( 1 - juliaValue );
        rgb[k+2] = 255;
      }
    }
  }
} // parallel
  return rgb;
}

int julia ( int w, int h, float xl, float xr, float yb, float yt, int i, int j, int cnt )
{
  float ai;
  float ar;
  float ci = 0.156;
  float cr = -0.8;
  int k;
  float t;
  float x;
  float y;

  x = ( ( float ) ( w - i - 1 ) * xl
      + ( float ) (     i     ) * xr ) 
      / ( float ) ( w     - 1 );

  y = ( ( float ) ( h - j - 1 ) * yb
      + ( float ) (     j     ) * yt ) 
      / ( float ) ( h     - 1 );

	  ar = x;
  ai = y;

  for ( k = 0; k < cnt; k++ )
  {
    t  = ar * ar - ai * ai + cr;
    ai = ar * ai + ai * ar + ci;
    ar = t;

    if ( 1000 < ar * ar + ai * ai )
    {
      return 0;
    }
  }

  return 1;
}

int check_validity ( unsigned char *rgb_sequential, unsigned char * rgb_parallel, int w, int h)
{
  for(int i = 0; i < w * h * 3; ++i)
  {
    if(rgb_parallel[i] != rgb_sequential[i])
      return 0;
  }
  return 1;
}

void tga_write ( int w, int h, unsigned char rgb[], char *filename )
{
  FILE *file_unit;
  unsigned char header1[12] = { 0,0,2,0,0,0,0,0,0,0,0,0 };
  unsigned char header2[6] = { w%256, w/256, h%256, h/256, 24, 0 };

  file_unit = fopen ( filename, "wb" );

  fwrite ( header1, sizeof ( unsigned char ), 12, file_unit );
  fwrite ( header2, sizeof ( unsigned char ), 6, file_unit );

  fwrite ( rgb, sizeof ( unsigned char ), 3 * w * h, file_unit );

  fclose ( file_unit );

  printf ( "\n" );
  printf ( "TGA_WRITE:\n" );
  printf ( "  Graphics data saved as '%s'\n", filename );

  return;
}

void timestamp ( void )
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}