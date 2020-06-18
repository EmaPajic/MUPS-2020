# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>

# define DEFAULT_H 1000
# define DEFAULT_W 1000
# define DEFAULT_CNT 200
# define DEFAULT_FILENAME "output/dz_julia"

#define NUM_OF_GPU_THREADS 1024
#define ACCURACY 2 // average error

int main (int argc, char *argv[]);
unsigned char *julia_set_sequential ( int w, int h, int cnt, float xl, float xr, float yb, float yt );
unsigned char *julia_set_GPU ( int w, int h, int cnt, float xl, float xr, float yb, float yt );
__host__ __device__ int julia ( int w, int h, float xl, float xr, float yb, float yt, int i, int j, int cnt );
__global__ void julia_set_kernel(unsigned char *d_rgb, int w, int h, int cnt, float xl, float xr, float yb, float yt);
void tga_write ( int w, int h, unsigned char rgb[], char *filename );
void timestamp ( );
int check_validity ( unsigned char *rgb_sequential, unsigned char * rgb_parallel, int w, int h);

void checkCUDAError(const char *msg);


int main (int argc, char *argv[] )  {
  int h = DEFAULT_H;
  int w = DEFAULT_W;
  int cnt = DEFAULT_CNT;
  char filename[256] = DEFAULT_FILENAME; 
  char buffer[256];

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

  //timestamp();
  printf ( "\n" );
  printf ( "JULIA Set\n" );
  printf ( "  Plot a version of the Julia set for Z(k+1)=Z(k)^2-0.8+0.156i\n" );
  fflush(stdout);

  unsigned char *rgb_sequential;
  unsigned char *rgbGPU;

  cudaEvent_t start_time, end_time;
  cudaEventCreate( &start_time );
  cudaEventCreate( &end_time );

  cudaEventRecord( start_time, 0);
  
  rgbGPU = julia_set_GPU( w, h, cnt, xl, xr, yb, yt );

  cudaEventRecord( end_time, 0 );
  cudaEventSynchronize( end_time );

  float elapsed_time_GPU = 0.f;
  cudaEventElapsedTime( &elapsed_time_GPU, start_time, end_time);
  printf("GPU time: %2.3f\n", elapsed_time_GPU / 1000);
  cudaEventDestroy( start_time );
  cudaEventDestroy( end_time );


  /*printf ( "\n" );
  printf ( "JULIA set:\n" );
  printf ( "  Normal end of execution.\n" );*/

  //timestamp();

  clock_t t;
  t = clock();
  rgb_sequential = julia_set_sequential ( w, h, cnt, xl, xr, yb, yt );
  t = clock() - t;
  double elapsed_time_sequential = ((double)t)/CLOCKS_PER_SEC;
  printf("Sequential time: %2.3f\n", elapsed_time_sequential);
  fflush(stdout);
  
  if(check_validity(rgb_sequential, rgbGPU, w, h))
  {
    printf("Test PASSED\n");
  }
  else
  {
    printf("Test FAILED\n");
  }
  
  tga_write ( w, h, rgbGPU, filename );

  cudaFree(rgbGPU);
  free ( rgb_sequential );
  return 0;
}

unsigned char *julia_set_sequential ( int w, int h, int cnt, float xl, float xr, float yb, float yt )
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

unsigned char *julia_set_GPU ( int w, int h, int cnt, float xl, float xr, float yb, float yt )
{
  int j;
  unsigned char *h_rgb;
  unsigned char *d_rgb;

  h_rgb = ( unsigned char * ) malloc ( w * h * 3 * sizeof ( unsigned char ) );

  int numBlocks = ceil((float)h * w / NUM_OF_GPU_THREADS);
  
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(NUM_OF_GPU_THREADS, 1, 1);

  size_t memSize = w * h * 3 * sizeof(unsigned char);
  cudaMalloc((void **) &d_rgb, memSize);

  julia_set_kernel<<<dimGrid, dimBlock>>>(d_rgb, w, h, cnt, xl, xr, yb, yt);
  checkCUDAError("kernel invocation");

  cudaMemcpy(h_rgb, d_rgb, memSize, cudaMemcpyDeviceToHost);
  checkCUDAError("memcpy");
  cudaFree(d_rgb);
  
  return h_rgb;
}

__global__ void julia_set_kernel(unsigned char *d_rgb, int w, int h, int cnt, float xl, float xr, float yb, float yt) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int row = idx / w;
  int col = idx % w;

  if (row < h) {
  int k = 3 * idx;
  int juliaValue = julia ( w, h, xl, xr, yb, yt, col, row, cnt );

  d_rgb[k]   = 255 * ( 1 - juliaValue );
  d_rgb[k+1] = 255 * ( 1 - juliaValue );
  d_rgb[k+2] = 255;
  }
}

__host__ __device__ int julia ( int w, int h, float xl, float xr, float yb, float yt, int i, int j, int cnt )
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
  /*
  int count = 0;
  for(int i = 0; i < w * h * 3; ++i)
  {
    if(rgb_parallel[i] != rgb_sequential[i])
      ++count;
  }

  if (count <= ACCURACY)
    return 1;
  return 0;
  */
  double mse = 0;
  for(int i = 0; i < w * h * 3; ++i)
  {
    mse += fabs((double)rgb_parallel[i] - rgb_sequential[i]);
  }
  mse /= w * h * 3;
  printf("%lf\n", mse);
  if(mse < ACCURACY)
    return 1;
  return 0;
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

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}