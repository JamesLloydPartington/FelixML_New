#include <stdio.h>
#include <omp.h>
#include <stdbool.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#undef I
complex I = _Complex_I; //Makes it so z = a + b * I

const int nPix = 128; //Size of square
const int FileSize = 841; //Number of lines inside each StructureFactor file
const int NumberImages = 19890; //Total number of crystals
const double RScattFacToVolts = 47.913838;
const double complex constPi = 2.0 * 3.1415926535 * _Complex_I;



int main()
{
  int i, j, k, n;
  double x, y;
  //char Path[] = "/home/ug-ml/Documents/GitHub/FelixML_Project/Preprocessing/CrystalGeneratrion";
  char PathOfPath[] = "/home/ug-ml/felix-ML/DataGenerator3/CifFolder/FilePaths.txt";

  FILE* file_ptr1 = fopen(PathOfPath, "r"); //writing to a file called Frac1
  char*** FilePaths = (char***)calloc(NumberImages, sizeof(char**));
  char As[2000], Bs[2000], Cs[2000];
  int Ad, Bd;
  double Af;
  for(i = 0; i < NumberImages; i++)
  {
    fscanf(file_ptr1, "%s %s %s %d %d %lf", As, Bs, Cs, &Ad, &Bd, &Af);
    FilePaths[i] = (char**)calloc(2, sizeof(char*));
    for(j = 0; j < 2; j++)
    {
      FilePaths[i][j] = (char*)calloc(2000, sizeof(char));
    }
    strcpy(FilePaths[i][0], Bs);
    strcpy(FilePaths[i][1], Cs);
    //printf("%s %s\n", FilePaths[i][0], FilePaths[i][1]);
  }
  fclose(file_ptr1);





  void OpenData(char*, double complex*, int**);

  double ReDotgVec;
  double complex ExpVar;
  double OneOvernPix = 1.0 / (nPix - 1);
  FILE* file_ptr2;



  #pragma omp parallel default(none) private(i, j, k, n, x, y, ReDotgVec, ExpVar, file_ptr2) shared(OneOvernPix, FilePaths)
  {
    double complex* Cug = (double complex*)calloc(FileSize, sizeof(double complex));
    int** gVec = (int**)calloc(FileSize, sizeof(int*));
    for(i = 0; i < FileSize; i++)
    {
      gVec[i] = (int*)calloc(3, sizeof(int));
    }

    double** RreUr = (double**)calloc(nPix, sizeof(double*));
    for(i = 0; i < nPix; i++)
    {
      RreUr[i] = (double*)calloc(nPix, sizeof(double));
    }
    //printf("Hello\n");

    #pragma omp for
    for(k = 0; k < NumberImages; k++)
    {

      //printf("%d %d %d\n", RreUr, gVec, Cug);
      //printf("Reading File\n");
      OpenData(FilePaths[k][1], Cug, gVec);
      //printf("%s\n", FilePaths[k][1]);
      for(i = 0; i < nPix; i++)
      {
        x = i * OneOvernPix;
        for(j = 0; j < nPix; j++)
        {
          y = j * OneOvernPix;
          for(n = 0; n < FileSize; n++)
          {
            //ReDotgVec = np.dot(Re, gVectors[n])
            ReDotgVec = x * gVec[n][0] + y * gVec[n][1];
            ExpVar = cexp(constPi * ReDotgVec);
            RreUr[j][i]+= creal(Cug[n] * ExpVar);
          }

        }
      }
      file_ptr2 = fopen(FilePaths[k][0], "w"); //writing to a file called Frac1
      for(i = 0; i < nPix; i++)
      {
        for(j = 0; j < nPix; j++)
        {
          RreUr[j][i] = RreUr[j][i] * RScattFacToVolts;
          //printf("%g ", RreUr[j][i]);
          fprintf(file_ptr2, "%g ", RreUr[j][i]);
          RreUr[j][i] = 0;
        }
        //printf("\n");
        fprintf(file_ptr2, "\n");
      }
      printf("%d of %d\n", k, NumberImages);
      fclose(file_ptr2);
    }
  }
  return(0);
}



void OpenData(char* NameofFile, double complex* Cug, int** gVec)
{
  FILE * file_ptr = fopen(NameofFile, "r"); //writing to a file called Frac1
  int i;
  double A, B, C, D;
  int E, F, G;
  //printf("%s\n ", NameofFile);
  for(i = 0; i < FileSize; i++)
  {
    fscanf(file_ptr, "\t%d\t%d\t%d\t%lf\t%lf\t%lf\t%lf", &gVec[i][0], &gVec[i][1], &gVec[i][2], &A, &B, &C, &D);
    Cug[i] = C + I * D;
    //printf("%g + i%g\n", creal(Cug[i]), cimag(Cug[i]));
    //printf("%d %d %d\n", gVec[i][0], gVec[i][1], gVec[i][2]);
  }
  fclose(file_ptr);
}




char* concat(char* s1, char* s2)
{
    char *result = malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}
