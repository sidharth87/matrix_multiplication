#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <assert.h>


#define N 2048
#define M 2048
#define P 2048

#define TILE_SIZE 8

static int **A;    // Input matrix A of size N x M
static int **B;    // Input matrix B of size M x P
static int **Bt;   // Transpose of matrix B

// output using the naive method
static int **C1;   // Output matrix (AxB) of size N x P

// output using the transpose of B
static int **C2;   // Output matrix (AxB) of size N x P

// output using tiling approach where you iterate through blocks in C
// but in rows in A and columns in B
static int **C3;   // Output matrix (AxB) of size N x P

// output using tiling approach where you iterate through blocks in C B and A
static int **C4;   // Output matrix (AxB) of size N x P

static void allocate_matrix_buffers();
static void initialize_matrices();
static void delete_matrix_buffers();
void verify_matrix_multiplication();

int main(int argc, char *argv[])
{
    srand (time(NULL));


    // allocate buffer on the stack
    allocate_matrix_buffers();

    // initialize matrices A and B and C (with 0s)
    initialize_matrices();

    // Approach 1
    // Naive matrix multiplication
    // https://www.youtube.com/watch?v=QYpH-847z0E
    clock_t naive_begin = std::clock();
    for (int i = 0; i < N ; i++)
        for (int k = 0; k < P; k++)
            for(int j = 0; j < M; ++j)
                C1[i][k] += A[i][j] * B[j][k];
    clock_t naive_end = std::clock();


    // Approach 2
    // Transposed matrix multiplication
    // https://www.youtube.com/watch?v=0u2K_dRLhWw
    clock_t transposed_begin = std::clock();
    for (int i = 0; i < N ; i++)
        for (int k = 0; k < P; k++)
            for(int j = 0; j < M; ++j)
                C2[i][k] += A[i][j] * Bt[k][j];
    clock_t transposed_end = std::clock();


    // Approach 3
    // Tiled matrix multiplication
    // We move in tiles in matrix A B and C
    // https://www.youtube.com/watch?v=aMvCEEBIBto
    clock_t tiled_begin = std::clock();
    for (int i0 = 0; i0 < N; i0 += TILE_SIZE)
      for (int j0 = 0; j0 < M; j0 += TILE_SIZE)
        for (int k0 = 0; k0 < P; k0 += TILE_SIZE)
          for (int i1 = i0; i1 < i0 + TILE_SIZE; ++i1)
            for (int j1 = j0; j1 < j0 + TILE_SIZE; ++j1)
              for (int k1 = k0; k1 < k0 + TILE_SIZE; ++k1)
                C3[i1][j1] += A[i1][k1] * Bt[j1][k1];

    clock_t tiled_end = std::clock();



    // Approach 4
    // Tiled matrix multiplication
    // We move in tiles in C but entire rows in A and colums in B
    // https://www.youtube.com/watch?v=G92BCtfTwOE
    clock_t flat_tiled_begin = std::clock();
    for (int i0 = 0; i0 < N; i0 += TILE_SIZE)
      for (int j0 = 0; j0 < M; j0 += TILE_SIZE)
        for (int i1 = i0; i1 < i0 + TILE_SIZE; ++i1)
          for (int j1 = j0; j1 < j0 + TILE_SIZE; ++j1)
            for (int k1 = 0; k1 < M; ++k1)
              C4[i1][j1] += A[i1][k1] * Bt[j1][k1];

    clock_t flat_tiled_end = std::clock();


    double naive_elapsed_secs = double(naive_end - naive_begin) / CLOCKS_PER_SEC;
    double transposed_elapsed_secs = double(transposed_end - transposed_begin) / CLOCKS_PER_SEC;
    double tiled_elapsed_secs = double(tiled_end - tiled_begin) / CLOCKS_PER_SEC;
    double flat_tiled_elapsed_secs = double(flat_tiled_end - flat_tiled_begin) / CLOCKS_PER_SEC;

    printf ("Naive method %f seconds \n", naive_elapsed_secs );
    printf ("Transposed method %f seconds \n", transposed_elapsed_secs );
    printf ("Tiled method %f seconds \n", tiled_elapsed_secs );
    printf ("Flat tiled method %f seconds \n", flat_tiled_elapsed_secs );

    verify_matrix_multiplication();


    delete_matrix_buffers();

    return 0;
}


// initialize input matrices A and B randomly
void initialize_matrices()
{
  // Initializing A with random numbers
  for (int i = 0; i < N; i++)
    for (int j = 0; j < M; j++)
      A[i][j] = rand() % (N * M) + 1;


  // Initializing B with random numbers
  for (int j = 0; j < M; j++)
    for (int k = 0; k < P; k++)
    {
      B[j][k] = rand() % (M * P) + 1;
      Bt[k][j] = B[j][k];
    }

  // Initializing C with 0s
  for (int i = 0; i < N; i++)
    for (int k = 0; k < P; k++)
    {
      C1[i][k] = 0;
      C2[i][k] = 0;
      C3[i][k] = 0;
      C4[i][k] = 0;
    }

  return;
}



void allocate_matrix_buffers()
{
  A = new int*[N];
  for(int i = 0; i < N; ++i)
      A[i] = new int[M];

    B = new int*[M];
    for(int i = 0; i < M; ++i)
        B[i] = new int[P];

    Bt = new int*[P];
    for(int i = 0; i < P; ++i)
        Bt[i] = new int[M];

    C1 = new int*[N];
    for(int i = 0; i < N; ++i)
        C1[i] = new int[P];

    C2 = new int*[N];
    for(int i = 0; i < N; ++i)
        C2[i] = new int[P];

    C3 = new int*[N];
    for(int i = 0; i < N; ++i)
        C3[i] = new int[P];

    C4 = new int*[N];
    for(int i = 0; i < N; ++i)
        C4[i] = new int[P];

    return;
}


static void delete_matrix_buffers()
{
  for(int i = 0; i < M; ++i)
      delete [] A[i];
  delete [] A;

  for(int i = 0; i < P; ++i)
      delete [] B[i];
  delete [] B;

  for(int i = 0; i < M; ++i)
      delete [] Bt[i];
  delete [] Bt;

  for(int i = 0; i < P; ++i)
      delete [] C1[i];
  delete [] C1;

  for(int i = 0; i < P; ++i)
      delete [] C2[i];
  delete [] C2;

  for(int i = 0; i < P; ++i)
      delete [] C3[i];
  delete [] C3;

  for(int i = 0; i < P; ++i)
      delete [] C4[i];
  delete [] C4;
}


void verify_matrix_multiplication()
{
  for (int i = 0; i < N; i++)
  {
    for (int k = 0; k < P; k++)
    {
      assert(C1[i][k] == C2[i][k]);
      assert(C2[i][k] == C3[i][k]);
      assert(C3[i][k] == C4[i][k]);
    }
  }
}
