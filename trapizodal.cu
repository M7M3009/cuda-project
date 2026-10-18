#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <device_launch_parameters.h>
#define N  1000000//number of trapezoids
#define A 50
#define B 100
// x^3+x^2+1;
double function(double* o) {
	double x = *o;
	double f = pow(x,3)+pow(x, 2) + 1;
	return f;
}
//destance between two points
double functionDx(double* a, double* b) {
	double d = (*b - *a) / N;
	return d;
}
// x^3+x^2+1;
__device__ double d_function(double* o) {
	double x = *o;
	double f = pow(x,3) + pow(x, 2) + 1;
	return f;
}
__global__ void reductionKernal(double* dx, double* area) {
	extern __shared__ double sdata[];//dynamic allocation
	unsigned int tid = threadIdx.x;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	double x = index * (*dx) + A;
	double localarea = 0.0, fl = 0.0, fr = 0.0;
	if (index < N) {//fall back in case of extra threads
		fl = d_function(&x);
		double xx = x + (*dx);
		fr = d_function(&xx);
		localarea = (fl + fr) * *(dx) / 2.0;
		
		
		//atomicAdd(area, localarea);//can be imroved by reduction 

	}
	else {
		//area[index] = 0.0;
	}


	sdata[tid] = localarea;
	
	__syncthreads();
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (threadIdx.x % (2 * s) == 0) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}
	//for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		//if (threadIdx.x < s) {
			//sdata[threadIdx.x] += sdata[threadIdx.x + s];
		//}
		//__syncthreads();

	//}

	if (tid == 0) /*area = sdata[0];*/ atomicAdd(area, sdata[0]);


}
__global__ void kernal(double* dx,double* area) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	double x = index * (*dx) + A;
	double localarea = 0.0, fl = 0.0, fr = 0.0;
	if (index < N) {//fall back in case of extra threads
		fl = d_function(&x);
		double xx = x + (*dx);
		fr = d_function(&xx);
		localarea =(fl + fr) * *(dx) / 2.0;
		atomicAdd(area, localarea);//can be imroved by reduction 

	}
	

}



int main() {
	cudaEvent_t event1, event2;
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);
	double fl=0.0,  fr=0.0,area=0.0 ,dx=0.0,//cpu
		  *d_fl, *d_fr,*d_area,*d_dx;// gbu
	int size = N* sizeof(double);

	//memory allocation for atomicAdd
	
	cudaMalloc((void**)&d_area, sizeof(double));
	cudaMalloc((void**)&d_dx, sizeof(double));
	
	//memory allocation for reduction
	//cudaMalloc((void**)&d_area, size);
	//cudaMalloc((void**)&d_dx, sizeof(double));
	//d_area = (double*)malloc(size);

	double a = A;
	double b = B;
	fl = function(&a);
	dx = functionDx(&a, &b);

	//memory for attomicAdd
	cudaMemcpy(d_area, &area, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dx, &dx, sizeof(double), cudaMemcpyHostToDevice);
	
	//memory for reduction
	//cudaMemcpy(d_area, &area, size, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_dx, &dx, sizeof(double), cudaMemcpyHostToDevice);

	//kernal
	int threads = 1024;
	int block =(N + threads - 1) / threads;

	
	
	cudaEventRecord(event1);

	kernal<<<block, threads >>> (d_dx, d_area);
	cudaEventRecord(event2);

	//reductionKernal <<<block, threads,threads*sizeof(double) >> > (d_dx, d_area);
	//reductionKernal <<<1, threads, threads*sizeof(double) >>> (d_dx, d_area);
	cudaDeviceSynchronize();
	
	

	cudaMemcpy(&area, d_area, sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(event2);
	float ms = 0;
	cudaEventElapsedTime(&ms, event1, event2);
	printf("area=%f\n", area);
	printf("Time taken of GBU: %f ms\n", ms);
	
	cudaFree(d_area);
	cudaFree(d_dx);

    return 0;
}