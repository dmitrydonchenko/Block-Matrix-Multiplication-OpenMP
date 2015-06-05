#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <cmath>
//#include <Windows.h>

#pragma comment (lib, "msmpi.lib")

using namespace std;
using namespace std::chrono;

void matrix_mul(float **A, float **B, float **C, int size);
void matrix_mul_transposed(float **A, float **BT, float **C, int size);
void matrix_mul_parallel(float **A, float **B, float **C, int size);
void matrix_mul_parallel_transposed(float **A, float **B, float **C, int size);
void block_matrix_mul(float **A, float **B, float **C, int size, int block_size);
void block_matrix_mul_transposed(float **A, float **BT, float **C, int size, int block_size);
void block_matrix_mul_parallel(float **A, float **B, float **C, int size, int block_size);

void output_matrix(float **matrix, int size);

int main(int argc, char **argv)
{
	int size, block_size;
	cin >> size >> block_size;	
	float **A = new float*[size];
	float **B = new float*[size];
	float **BT = new float*[size];
	float **C = new float*[size];
	float **CC = new float*[size];
	cout << "Init has begun" << endl;
	for (int i = 0; i < size; i++)
	{
		A[i] = new float[size];
		B[i] = new float[size];
		BT[i] = new float[size];
		C[i] = new float[size];
		CC[i] = new float[size];
		for (int j = 0; j < size; j++)
		{
			A[i][j] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
			B[i][j] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
			C[i][j] = 0.0f;
			CC[i][j] = 0.0f;
		}
	}
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			BT[i][j] = B[j][i];
		}
	}

	for (int it = 0; it < 1; it++)
	{
		// serial 
		cout << "Computation has begun(serial)" << endl;
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		matrix_mul(A, B, C, size);

		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Time elapsed: " << static_cast <float> (duration) / 1000000.0f << endl;

		// serial transposed
		cout << "Computation has begun(serial transposed)" << endl;
		t1 = high_resolution_clock::now();

		matrix_mul_transposed(A, BT, C, size);

		t2 = high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Time elapsed: " << static_cast <float> (duration) / 1000000.0f << endl;

		// parallel
		cout << "Computation has begun(parallel)" << endl;
		t1 = high_resolution_clock::now();

		matrix_mul_parallel(A, B, C, size);

		t2 = high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Time elapsed: " << static_cast <float> (duration) / 1000000.0f << endl;

		// parallel transposed
		cout << "Computation has begun(parallel transposed)" << endl;
		t1 = high_resolution_clock::now();

		matrix_mul_parallel_transposed(A, BT, C, size);

		t2 = high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Time elapsed: " << static_cast <float> (duration) / 1000000.0f << endl;

		// serial block
		cout << "Computation has begun(serial block)" << endl;
		t1 = high_resolution_clock::now();

		block_matrix_mul(A, B, C, size, block_size);

		t2 = high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Time elapsed: " << static_cast <float> (duration) / 1000000.0f << endl;

		// serial block transposed
		cout << "Computation has begun(serial block, transposed)" << endl;
		t1 = high_resolution_clock::now();

		block_matrix_mul_transposed(A, BT, C, size, block_size);

		t2 = high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Time elapsed: " << static_cast <float> (duration) / 1000000.0f << endl;

		//parallel block
		cout << "Computation has begun(parallel block)" << endl;
		t1 = high_resolution_clock::now();

		block_matrix_mul_parallel(A, B, CC, size, block_size);

		t2 = high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Time elapsed: " << static_cast <float> (duration) / 1000000.0f << endl;
	}
	return 0;
}

void matrix_mul(float **A, float **B, float **C, int size)
{
	int i, j, k;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			for (k = 0; k < size; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}	
}

void matrix_mul_transposed(float **A, float **BT, float **C, int size)
{
	int i, j, k;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			for (k = 0; k < size; k++)
			{
				C[i][j] += A[i][k] * BT[j][k];
			}
		}
	}
}

void matrix_mul_parallel(float **A, float **B, float **C, int size)
{
	int i, j, k;
	int chunk = 100;
	int tid;
	#pragma omp parallel shared(A, B, C, size, chunk) private(i, j, k, tid)
	{
		tid = omp_get_thread_num();
		if (tid == 0)
		{
			cout << "Number of threads: " << omp_get_num_threads() << endl;
		}	
		#pragma omp for schedule (static, chunk)
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				for (k = 0; k < size; k++)
				{
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}
	}	
}

void matrix_mul_parallel_transposed(float **A, float **BT, float **C, int size)
{
	int i, j, k;
	int chunk = 100;
	int tid;
	#pragma omp parallel shared(A, BT, C, size, chunk) private(i, j, k, tid)
	{
		tid = omp_get_thread_num();
		if (tid == 0)
		{
			cout << "Number of threads: " << omp_get_num_threads() << endl;
		}
	#pragma omp for schedule (static, chunk)
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				for (k = 0; k < size; k++)
				{
					C[i][j] += A[i][k] * BT[j][k];
				}
			}
		}
	}
}

void block_matrix_mul_parallel(float **A, float **B, float **C, int size, int block_size)
{
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
	float tmp;
	int chunk = 1;
	int tid;
#pragma omp parallel shared(A, B, C, size, chunk) private(i, j, k, jj, kk, tid, tmp)
	{
		//omp_set_dynamic(0);
		//omp_set_num_threads(4);
		tid = omp_get_thread_num();
		if (tid == 0)
		{
			cout << "Number of threads: " << omp_get_num_threads() << endl;
		}
		#pragma omp for schedule (static, chunk)
		for (jj = 0; jj < size; jj += block_size)
		{
			//cout << "thread " << omp_get_thread_num() << "value " << i << endl;
			for (kk = 0; kk < size; kk += block_size)
			{
				for (i = 0; i < size; i++)
				{
					for (j = jj; j < ((jj + block_size) > size ? size : (jj + block_size)); j++)
					{
						tmp = 0.0f;
						for (k = kk; k < ((kk + block_size) > size ? size : (kk + block_size)); k++)
						{
							tmp += A[i][k] * B[k][j];
						}
						C[i][j] += tmp;
					}
				}
			}
		}
	}
}

void block_matrix_mul(float **A, float **B, float **C, int size, int block_size)
{
	int i, j, k;
	float tmp;
	for (int jj = 0; jj < size; jj += block_size)
	{
		for (int kk = 0; kk < size; kk += block_size)
		{
			for (int i = 0; i < size; i++)
			{
				for (int j = jj; j < ((jj + block_size) > size ? size : (jj + block_size)); j++)
				{
					tmp = 0.0f;
					for (int k = kk; k < ((kk + block_size) > size ? size : (kk + block_size)); k++)
					{
						tmp += A[i][k] * B[k][j];
					}
					C[i][j] += tmp;
				}
			}
		}
	}
}

void block_matrix_mul_transposed(float **A, float **BT, float **C, int size, int block_size)
{
	int i, j, k;
	float tmp;
	for (int jj = 0; jj < size; jj += block_size)
	{
		for (int kk = 0; kk < size; kk += block_size)
		{
			for (int i = 0; i < size; i++)
			{
				for (int j = jj; j < ((jj + block_size) > size ? size : (jj + block_size)); j++)
				{
					tmp = 0.0f;
					for (int k = kk; k < ((kk + block_size) > size ? size : (kk + block_size)); k++)
					{
						tmp += A[i][k] * BT[j][k];
					}
					C[i][j] += tmp;
				}
			}
		}
	}
}

void output_matrix(float **matrix, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
}



