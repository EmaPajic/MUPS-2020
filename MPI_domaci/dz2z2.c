# include <mpi.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>

# define DEFAULT_H 1000
# define DEFAULT_W 1000
# define DEFAULT_CNT 200
# define DEFAULT_FILENAME "output/dz_julia"
# define MASTER 0
# define N 4

int main (int argc, char *argv[]);
unsigned char *julia_set_sequential ( int w, int h, int cnt, float xl, float xr, float yb, float yt );
unsigned char *julia_set_parallel ( int w, int h, int cnt, float xl, float xr, float yb, float yt, int row );
int julia ( int w, int h, float xl, float xr, float yb, float yt, int i, int j, int cnt );
void tga_write ( int w, int h, unsigned char rgb[], char *filename );
void timestamp ( );
int check_validity ( unsigned char *rgb_sequential, unsigned char * rgb_parallel, int w, int h);

enum Tags 
{ 
	ROW_TAG = 1000, 
  ROW_RESULT_TAG,
	RESULT_TAG,
	END_TAG
};

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
  int size, rank;
  int row;

  if (argc == N) {
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

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size > 4) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  double start_time_parallel, end_time_parallel;

  if(rank == MASTER) {
    //timestamp();
    printf ( "\n" );
    printf ( "JULIA Set\n" );
    printf ( "  Plot a version of the Julia set for Z(k+1)=Z(k)^2-0.8+0.156i\n" );
    fflush(stdout);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == MASTER) {
    start_time_parallel = MPI_Wtime();
    unsigned sentCount = 0, processedCount = 0, excessSlavesCount = 0;
    MPI_Request request;
    // prvo posalji svakom procesu po jedan posao
		while (sentCount + excessSlavesCount < size - 1)
		{
			// ako je proces potreban, posalji mu podatke
			if (sentCount < h)
			{
        //posalji red svakom procesu
        row = sentCount;
        MPI_Isend(&row, 1, MPI_INT, (int)sentCount+1, ROW_TAG, MPI_COMM_WORLD, &request);
        ++sentCount;
			}
			else
			// inace, iskljuci ga
			{
				// salji bilo koji podatak, posto nije bitan podatak, nego tema poruke
				++excessSlavesCount;
				MPI_Isend(&sentCount, 1, MPI_UNSIGNED, sentCount + excessSlavesCount, END_TAG, MPI_COMM_WORLD, &request);
			}
		}
    rgb_parallel = ( unsigned char * ) calloc ( w * h * 3, sizeof ( unsigned char ) );
    // sakupljaj obradjene elemente i salji podatke za obradu ili poruku za kraj rada
    MPI_Status status;
    MPI_Request request1;
		while (processedCount < h)
		{
			// primaj jedan po jedan rezultat
			MPI_Recv(&row, 1, MPI_INT, MPI_ANY_SOURCE, ROW_RESULT_TAG, MPI_COMM_WORLD, &status);
      int workerRank = status.MPI_SOURCE;
      MPI_Irecv(rgb_parallel+row*w*3, w*3, MPI_UNSIGNED_CHAR, workerRank, RESULT_TAG, MPI_COMM_WORLD, &request1);
			++processedCount;

			// ako ima jos paketa podataka koje treba obraditi, salji istom koji je poslao rezultat
			if (sentCount < h)
			{
				row = sentCount;
        MPI_Wait(&request1, &status);
        MPI_Isend(&row, 1, MPI_INT, workerRank, ROW_TAG, MPI_COMM_WORLD, &request);
        ++sentCount;
			} 
			else
			{
				// Salji bilo koji podatak, posto nije bitan podatak, nego tema poruke
        MPI_Wait(&request1, &status);
				MPI_Isend(&sentCount, 1, MPI_UNSIGNED, workerRank, END_TAG, MPI_COMM_WORLD, &request);
			}
		}

    end_time_parallel = MPI_Wtime();
    printf("Parallel time: %2.3f\n", end_time_parallel - start_time_parallel);
    fflush(stdout);

    /*printf ( "\n" );
    printf ( "JULIA set:\n" );
    printf ( "  Normal end of execution.\n" );*/

    //timestamp();

  } else {

    int isSlaveNeeded = 1;
    MPI_Request request;
    MPI_Status status;
		while (isSlaveNeeded == 1)
		{
			MPI_Recv(&row, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			int receivedTag = status.MPI_TAG;
			if (END_TAG == receivedTag)
			{	// ako tema poruke oznacava kraj rada, nema razloga da se dalje vrti u ciklusu
				isSlaveNeeded = 0;
			} 
			else
			{	
				// izracunaj sta treba i posalji nazad
        unsigned char *rgb_row = ( unsigned char * ) malloc ( w * 3 * sizeof ( unsigned char ) );
				rgb_row = julia_set_parallel ( w, h, cnt, xl, xr, yb, yt, row );
        MPI_Isend(&row, 1, MPI_INT, MASTER, ROW_RESULT_TAG, MPI_COMM_WORLD, &request);
				MPI_Isend(rgb_row, w*3, MPI_UNSIGNED_CHAR, MASTER, RESULT_TAG, MPI_COMM_WORLD, &request);
        free(rgb_row);
			}
		}
  }

  if (rank == MASTER) {
    double start_time_sequential, end_time_sequential;
    start_time_sequential = MPI_Wtime();
    rgb_sequential = julia_set_sequential ( w, h, cnt, xl, xr, yb, yt );
    end_time_sequential = MPI_Wtime();
    printf("Sequential time: %2.3f\n", end_time_sequential - start_time_sequential);
    fflush(stdout);
    
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
  }

  MPI_Finalize();

  return 0;
}

unsigned char *julia_set_parallel ( int w, int h, int cnt, float xl, float xr, float yb, float yt, int j )
{
  int i;
  int juliaValue;
  int k;
  unsigned char *rgb;

  rgb = ( unsigned char * ) malloc ( w * 3 * sizeof ( unsigned char ) );

      for ( i = 0; i < w; i++ )
      {
      juliaValue = julia ( w, h, xl, xr, yb, yt, i, j, cnt );

      k = 3 * i;

      rgb[k]   = 255 * ( 1 - juliaValue );
      rgb[k+1] = 255 * ( 1 - juliaValue );
      rgb[k+2] = 255;
      }

  return rgb;
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
