#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>

#define MAXLINE 128
#define PIXPERLINE 16
#define MASTER 0
#define N 4

char c[MAXLINE];

double wtime();
void pgmsize(char *filename, int *nx, int *ny);
void pgmread(char *filename, void *vp, int nxmax, int nymax, int *nx, int *ny);
void pgmwrite(char *filename, void *vx, int nx, int ny);

void dosharpen(char *infile, int nx, int ny);
double filter(int d, int i, int j);

int  **int2Dmalloc(int nx, int ny);
double **double2Dmalloc(int nx, int ny);

double wtime(void)
{
  struct timeval tp;
  gettimeofday (&tp, NULL);
  return tp.tv_sec + tp.tv_usec/(double)1.0e6;
}

void pgmsize(char *filename, int *nx, int *ny)
{ 
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank == MASTER) {
    FILE *fp;

    if (NULL == (fp = fopen(filename,"r")))
    {
        fprintf(stderr, "pgmsize: cannot open <%s>\n", filename);
        exit(-1);
    }

    fgets(c, MAXLINE, fp);
    fgets(c, MAXLINE, fp);

    fscanf(fp,"%d %d",nx,ny);

    fclose(fp);
  }
  MPI_Bcast(nx, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(ny, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
}


void pgmread(char *filename, void *vp, int nxmax, int nymax, int *nx, int *ny)
{ 
  FILE *fp;

  int nxt, nyt, i, j, t;

  int *pixmap = (int *) vp;

  if (NULL == (fp = fopen(filename,"r")))
  {
    fprintf(stderr, "pgmread: cannot open <%s>\n", filename);
    exit(-1);
  }

  fgets(c, MAXLINE, fp);
  fgets(c, MAXLINE, fp);

  fscanf(fp,"%d %d",nx,ny);

  nxt = *nx;
  nyt = *ny;

  if (nxt > nxmax || nyt > nymax)
  {
    fprintf(stderr, "pgmread: image larger than array\n");
    fprintf(stderr, "nxmax, nymax, nxt, nyt = %d, %d, %d, %d\n",
	    nxmax, nymax, nxt, nyt);
    exit(-1);
  }

  fscanf(fp,"%d", &t);

  for (j=0; j<nyt; j++)
  {
    for (i=0; i<nxt; i++)
    {
      fscanf(fp,"%d", &t);
      pixmap[(nyt-j-1)+nyt*i] = t;
    }
  }

  fclose(fp);
}

void pgmwrite(char *filename, void *vx, int nx, int ny)
{
  FILE *fp;

  int i, j, k, grey;

  double xmin, xmax, tmp;
  double thresh = 255.0;

  double *x = (double *) vx;

  if (NULL == (fp = fopen(filename,"w")))
  {
    fprintf(stderr, "pgmwrite: cannot create <%s>\n", filename);
    exit(-1);
  }

  xmin = fabs(x[0]);
  xmax = fabs(x[0]);

  for (i=0; i < nx*ny; i++)
  {
    if (fabs(x[i]) < xmin) xmin = fabs(x[i]);
    if (fabs(x[i]) > xmax) xmax = fabs(x[i]);
  }

  fprintf(fp, "P2\n");
  fprintf(fp, "# Written by pgmwrite\n");
  fprintf(fp, "%d %d\n", nx, ny);
  fprintf(fp, "%d\n", (int) thresh);

  k = 0;

  for (j=ny-1; j >=0 ; j--)
  {
    for (i=0; i < nx; i++)
    {
 
      tmp = x[j+ny*i];

 
      if (xmin < 0 || xmax > thresh)
      {
        tmp = (int) ((thresh*((fabs(tmp-xmin))/(xmax-xmin))) + 0.5);
      }
      else
      {
        tmp = (int) (fabs(tmp) + 0.5);
      }

      grey = tmp;
 
      fprintf(fp, "%3d ", grey);

      if (0 == (k+1)%PIXPERLINE) fprintf(fp, "\n");

      k++;
    }
  }

  if (0 != k%PIXPERLINE) fprintf(fp, "\n");
  fclose(fp);
}

double filter(int d, int i, int j)
{
  double rd4sq, rsq, sigmad4sq, sigmasq, x, y, delta;

  int d4 = 4;

  double sigmad4 = 1.4;
  double filter0 = -40.0;

  rd4sq = d4*d4;
  rsq   = d*d;

  sigmad4sq = sigmad4*sigmad4;
  sigmasq   = sigmad4sq * (rsq/rd4sq);

  x = (double) i;
  y = (double) j;

  rsq = x*x + y*y;

  delta = rsq/(2.0*sigmasq);

  return(filter0 * (1.0-delta) * exp(-delta));
}


void dosharpen_sequential(char *infile, int nx, int ny)
{
  int d = 8;  

  double  norm = (2*d-1)*(2*d-1);
  double scale = 2.0;
  
  int xpix, ypix, pixcount;
  
  int i, j, k, l;
  double tstart, tstop, time;
  
  int **fuzzy = int2Dmalloc(nx, ny);                   /* Will store the fuzzy input image when it is first read in from file */
  double **fuzzyPadded = double2Dmalloc(nx+2*d, ny+2*d);  /* Will store the fuzzy input image plus additional border padding */
  double **convolutionPartial = double2Dmalloc(nx, ny);   /* Will store the convolution of the filter with parts of the fuzzy image computed by individual processes */
  double **convolution = double2Dmalloc(nx, ny);          /* Will store the convolution of the filter with the full fuzzy image */
  double **sharp = double2Dmalloc(nx, ny);                /* Will store the sharpened image obtained by adding rescaled convolution to the fuzzy image */
  double **sharpCropped = double2Dmalloc(nx-2*d, ny-2*d); /* Will store the sharpened image cropped to remove a border layer distorted by the algorithm */
  
  char outfile[256];
  strcpy(outfile, infile);
  *(strchr(outfile, '.')) = '\0';
  strcat(outfile, "_sharpened.pgm");
  
  for (i=0; i < nx; i++)
    {
      for (j=0; j < ny; j++)
        {
          fuzzy[i][j] = 0;
          sharp[i][j] = 0.0;
        }
    }
  
  //printf("Using a filter of size %d x %d\n", 2*d+1, 2*d+1);
  //printf("\n");

  //printf("Reading image file: %s\n", infile);
  //fflush(stdout);
       
  pgmread(infile, &fuzzy[0][0], nx, ny, &xpix, &ypix);
  //printf("... done\n\n");
  //fflush(stdout);
  
  if (xpix == 0 || ypix == 0 || nx != xpix || ny != ypix)
    {
      printf("Error reading %s\n", infile);
      fflush(stdout);
      exit(-1);
    }
  
  for (i=0; i < nx+2*d; i++)
    {
      for (j=0; j < ny+2*d; j++)
        {
          fuzzyPadded[i][j] = 0.0;
        }
    }
  
  for (i=0; i < nx; i++)
    { 
      for (j=0; j < ny; j++)
        {
          fuzzyPadded[i+d][j+d] = fuzzy[i][j];
        }
    }
  
  //printf("Starting calculation ...\n");
  
  //fflush(stdout);
  
  tstart = wtime();
  
  pixcount = 0;
  
  for (i=0; i < nx; i++)
    {
      for (j=0; j < ny; j++)
        {
          for (k=-d; k <= d; k++)
            {
              for (l= -d; l <= d; l++)
                {
                  convolution[i][j] = convolution[i][j] + filter(d,k,l)*fuzzyPadded[i+d+k][j+d+l];
                }
            }
          pixcount += 1;
        }
    }
  
  tstop = wtime();
  time = tstop - tstart;
  
  //printf("... finished\n");
  //printf("\n");
  //fflush(stdout);
  
  for (i=0 ; i < nx; i++)
    {
      for (j=0; j < ny; j++)
        {
          sharp[i][j] = fuzzyPadded[i+d][j+d] - scale/norm * convolution[i][j];
        }
    }
  
  //printf("Writing output file: %s\n", outfile);
  //printf("\n");
  
  for (i=d ; i < nx-d; i++)
    {
      for (j=d; j < ny-d; j++)
        {
          sharpCropped[i-d][j-d] = sharp[i][j];
        }
    }
  
  pgmwrite(outfile, &sharpCropped[0][0], nx-2*d, ny-2*d);
  
  //printf("... done\n");
  //printf("\n");
  printf("Sequential calculation time : %f seconds\n", time);
  fflush(stdout);

  free(fuzzy);
  free(fuzzyPadded);
  free(convolutionPartial);
  free(convolution);
  free(sharp);
  free(sharpCropped);
}

void dosharpen_parallel(char *infile, int nx, int ny)
{
  int d = 8;  

  double  norm = (2*d-1)*(2*d-1);
  double scale = 2.0;
  
  int xpix, ypix, pixcount;
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  
  int i, j, k, l;
  double tstart, tstop, time;
  
  int **fuzzy = int2Dmalloc(nx, ny);                   /* Will store the fuzzy input image when it is first read in from file */
  double **fuzzyPadded = double2Dmalloc(nx+2*d, ny+2*d);  /* Will store the fuzzy input image plus additional border padding */
  double **convolutionPartial = double2Dmalloc(nx, ny);   /* Will store the convolution of the filter with parts of the fuzzy image computed by individual processes */
  double **convolution = double2Dmalloc(nx, ny);          /* Will store the convolution of the filter with the full fuzzy image */
  double **sharp = double2Dmalloc(nx, ny);                /* Will store the sharpened image obtained by adding rescaled convolution to the fuzzy image */
  double **sharpCropped = double2Dmalloc(nx-2*d, ny-2*d); /* Will store the sharpened image cropped to remove a border layer distorted by the algorithm */
  char outfile[256];

  if (rank == MASTER) {  
    strcpy(outfile, infile);
    *(strchr(outfile, '.')) = '\0';
    strcat(outfile, "_sharpened_dz.pgm");
  }

  for (i=0; i < nx; i++)
    {
      for (j=0; j < ny; j++)
        {
          fuzzy[i][j] = 0;
          sharp[i][j] = 0.0;
        }
    }
  
  //printf("Using a filter of size %d x %d\n", 2*d+1, 2*d+1);
  //printf("\n");

  //printf("Reading image file: %s\n", infile);
  //fflush(stdout);
  if (rank == MASTER) {     
    pgmread(infile, &fuzzy[0][0], nx, ny, &xpix, &ypix);
    //printf("... done\n\n");
    //fflush(stdout);
    

    if (xpix == 0 || ypix == 0 || nx != xpix || ny != ypix) {
        printf("Error reading %s\n", infile);
        fflush(stdout);
        exit(-1);
    }
  }

  
  /*printf("Alive before Bcast %ld %ld\n", sizeof(int*),sizeof(int));
  fflush(stdout);*/
  MPI_Bcast(fuzzy[0], nx*ny, MPI_INT, MASTER, MPI_COMM_WORLD);
  //printf("Alive after Bcast\n");
  //fflush(stdout);
  

  for (i=0; i < nx+2*d; i++)
    {
      for (j=0; j < ny+2*d; j++)
        {
          fuzzyPadded[i][j] = 0.0;
        }
    }
  
  for (i=0; i < nx; i++)
    { 
      for (j=0; j < ny; j++)
        {
          fuzzyPadded[i+d][j+d] = fuzzy[i][j];
        }
    }
  
  //printf("Starting calculation ...\n");
  //fflush(stdout);
  //tstart = wtime();

  pixcount = 0;

  int start, end, chunk;
  chunk = (nx + size - 1) / size;
  start = chunk * rank;
  end = start + chunk < nx ? start + chunk : nx;

  for (i=start; i < end; i++)
  {
    for (j=0; j < ny; j++)
      {
        for (k=-d; k <= d; k++)
          {
            for (l= -d; l <= d; l++)
              {
                convolution[i][j] = convolution[i][j] + filter(d,k,l)*fuzzyPadded[i+d+k][j+d+l];
              }
          }
        pixcount += 1;
      }
  }
  
  if(rank == MASTER)
    MPI_Reduce(MPI_IN_PLACE, convolution[0], nx*ny, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
  else
    MPI_Reduce(convolution[0], convolution[0], nx*ny, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
  if(rank == MASTER)
    MPI_Reduce(MPI_IN_PLACE, &pixcount, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  else
    MPI_Reduce(&pixcount, &pixcount, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  // 
  //tstop = wtime();
  //time = tstop - tstart;
  
  //printf("... finished\n");
  //printf("\n");
  //fflush(stdout);
  if (rank == MASTER) {

  for (i=0 ; i < nx; i++)
    {
      for (j=0; j < ny; j++)
        {
          sharp[i][j] = fuzzyPadded[i+d][j+d] - scale/norm * convolution[i][j];
        }
    }
  
  //printf("Writing output file: %s\n", outfile);
  //printf("\n");
  
  for (i=d ; i < nx-d; i++)
    {
      for (j=d; j < ny-d; j++)
        {
          sharpCropped[i-d][j-d] = sharp[i][j];
        }
    }
  
  pgmwrite(outfile, &sharpCropped[0][0], nx-2*d, ny-2*d);
  }
  
  //printf("... done\n");
  //printf("\n");
  //printf("Parallel calculation time : %f seconds\n", time);
  //fflush(stdout);

  free(fuzzy);
  free(fuzzyPadded);
  free(convolutionPartial);
  free(convolution);
  free(sharp);
  free(sharpCropped);
}

int **int2Dmalloc(int nx, int ny)
{
  int i;
  int **idata;

  idata = (int **) malloc(nx*sizeof(int *) + nx*ny*sizeof(int));

  idata[0] = (int *) (idata + nx);

  for(i=1; i < nx; i++)
    {
      idata[i] = idata[i-1] + ny;
    }

  return idata;
}

double **double2Dmalloc(int nx, int ny)
{
  int i;
  double **ddata;

  ddata = (double **) malloc(nx*sizeof(double *) + nx*ny*sizeof(double));

  ddata[0] = (double *) (ddata + nx);

  for(i=1; i < nx; i++)
    {
      ddata[i] = ddata[i-1] + ny;
    }

  return ddata;
}

void check_validity(char *filename1, char *filename2)
{
  int nx1, ny1;
  int nx2, ny2;
  int xpix1, ypix1, xpix2, ypix2;
  int i, j;
  pgmsize(filename1, &nx1, &ny1);
  pgmsize(filename2, &nx2, &ny2);

  if (nx1 != nx2 || ny1 != ny2) 
  {
      printf("Test FAILED");
      return;
  }

  double **picture1 = double2Dmalloc(nx1, ny1); 
  double **picture2 = double2Dmalloc(nx2, ny2); 

  for (i=0; i < nx1; i++)
    {
      for (j=0; j < ny1; j++)
        {
          picture1[i][j] = 0.0;
          picture2[i][j] = 0.0;
        }
    }

  pgmread(filename1, &picture1[0][0], nx1, ny1, &xpix1, &ypix1);
  
  if (xpix1 == 0 || ypix1 == 0 || nx1 != xpix1 || ny1 != ypix1)
    {
      printf("Error reading %s\n", filename1);
      fflush(stdout);
      exit(-1);
    }

  pgmread(filename2, &picture2[0][0], nx2, ny2, &xpix2, &ypix2);
  
  if (xpix2 == 0 || ypix2 == 0 || nx2 != xpix2 || ny2 != ypix2)
    {
      printf("Error reading %s\n", filename2);
      fflush(stdout);
      exit(-1);
    }

  for (i=0; i < nx1; i++)
    { 
      for (j=0; j < ny1; j++)
        {
          if(picture1[i][j] != picture2[i][j]) 
          {
            printf("Test FAILED\n");
            free(picture1);
            free(picture2);
            return;
          } 
        }
    }
  
  printf("Test PASSED\n");
  free(picture1);
  free(picture2);
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  
  char *filename;
  int xpix, ypix;
  int size, rank;

  if (argc < 2) return 1;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  filename = argv[1];
  pgmsize(filename, &xpix, &ypix);
  //printf("\n");
  //printf("Image sharpening code running in serial\n");
  //printf("\n");
  
  if (size > N) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  double time_begin_parallel, time_end_parallel;
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == MASTER) {
    printf("Input file is: %s\n", filename);
    printf("Image size is %d x %d\n", xpix, ypix);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == MASTER) {
    time_begin_parallel = MPI_Wtime();
  }
  
  dosharpen_parallel(filename, xpix, ypix);

  if (rank == MASTER) {
  time_end_parallel = MPI_Wtime();
  
  printf("Parallel time: %2.2f \n", time_end_parallel - time_begin_parallel);
  fflush(stdout);
  
  char filename1[256];
  char filename2[256];

  strcpy(filename1, filename);
  *(strchr(filename1, '.')) = '\0';
  strcat(filename1, "_sharpened.pgm");

  strcpy(filename2, filename);
  *(strchr(filename2, '.')) = '\0';
  strcat(filename2, "_sharpened_dz.pgm");

  double time_begin_sequential, time_end_sequential;

  time_begin_sequential = MPI_Wtime();
  dosharpen_sequential(filename, xpix, ypix);
  time_end_sequential = MPI_Wtime();

  printf("Sequential time: %2.2f \n", time_end_sequential - time_begin_sequential);
  fflush(stdout);
  check_validity(filename1, filename2);
  printf("\n");
  } 

  MPI_Finalize();
  return 0;
}
