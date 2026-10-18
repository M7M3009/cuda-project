#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <time.h>

//cpu functions
double function(double* o) {
	double x = *o;
	double f = pow(x, 3) + pow(x, 2) + 1;
	return f;
}
double xPlusOne(double* o) {
	double x = *o;
	double f = x + 1;
	return f;
}
double xSquarePlusOne(double* o) {
	double x = *o;
	double f = x*x + 1;
	return f;
}
double xCubePlusOne(double* o) {
	double x = *o;
	double f = x*x*x + 1;
	return f;
}
double sinPlusOne(double* o) {
	double x = *o;
	double f = sin(x) + 1;
	return f;
}
double cosPlusOne(double* o) {
	double x = *o;
	double f = cos(x) + 1;
	return f;
}
double expoOfTwoPlusOne(double* o) {
	double x = *o;
	double f = pow(2,x) + 1;
	return f;
}
double logOfTwoPlusOne(double* o) {
	double x = *o;
	double f = log2(x) + 1;
	return f;
}
double expoOfthreePlusOne(double* o) {
	double x = *o;
	double f = pow(3, x) + 1;
	return f;
}
double xCubePlusXSquarePlusOne(double* o) {
	double x = *o;
	double f = x*x*x + x*x + 1;
	return f;
}


//gbu functions
__device__ double d_function(double* o) {
	double x = *o;
	double f = pow(x, 3) + pow(x, 2) + 1;
	return f;
}
__device__ double d_xPlusOne(double* o) {
	double x = *o;
	double f = x + 1;
	return f;
}
__device__ double d_xSquarePlusOne(double* o) {
	double x = *o;
	double f = x * x + 1;
	return f;
}
__device__ double d_xCubePlusOne(double* o) {
	double x = *o;
	double f = x * x * x + 1;
	return f;
}
__device__ double d_sinPlusOne(double* o) {
	double x = *o;
	double f = sin(x) + 1;
	return f;
}
__device__ double d_cosPlusOne(double* o) {
	double x = *o;
	double f = cos(x) + 1;
	return f;
}
__device__ double d_expoOfTwoPlusOne(double* o) {
	double x = *o;
	double f = pow(2, x) + 1;
	return f;
}
__device__ double d_logOfTwoPlusOne(double* o) {
	double x = *o;
	double f = log2(x) + 1;
	return f;
}
__device__ double d_expoOfthreePlusOne(double* o) {
	double x = *o;
	double f = pow(3, x) + 1;
	return f;
}
__device__ double d_xCubePlusXSquarePlusOne(double* o) {
	double x = *o;
	double f = x * x * x + x * x + 1;
	return f;
}

__global__ void kernal(double lArea, double rArea, double m, double* area, double a, double b,double epslon,int functionNumber) {
	int idx = threadIdx.x;
	double totalLocalArea = 0;
	double (*d_arrFunction[])(double*) = { d_xPlusOne,d_xSquarePlusOne,d_xCubePlusOne,d_sinPlusOne,d_cosPlusOne,d_expoOfTwoPlusOne,d_expoOfthreePlusOne,d_logOfTwoPlusOne,d_xCubePlusXSquarePlusOne };
	

	if (idx == 0) {
		double currentArea, currentM, mFunction, fl, fr;
		fl = (*d_arrFunction[functionNumber])(&a);
		fr = (*d_arrFunction[functionNumber])(&m);
		currentArea = (fl + fr) * (m - a) / 2;
		currentM = (m + a) / 2.0;
		mFunction = (*d_arrFunction[functionNumber])(&currentM);
		double lArea = (fl + mFunction) * (currentM - a) / 2;
		double rArea = (mFunction + fr) * (m - currentM) / 2;
		if (fabs((lArea + rArea) - currentArea) > epslon) {
			kernal << <1, 2 >> > (lArea, rArea, currentM, area, a, m,epslon,functionNumber);
		}
		else
			atomicAdd(area, lArea + rArea);
	}
	else if (idx == 1) {
		double currentArea, currentM, mFunction, fl, fr;
		fl = (*d_arrFunction[functionNumber])(&m);
		fr = (*d_arrFunction[functionNumber])(&b);
		currentArea = (fl + fr) * (b - m) / 2;
		currentM = (b + m) / 2.0;
		mFunction = (*d_arrFunction[functionNumber])(&currentM);
		double lArea = (fl + mFunction) * (currentM - m) / 2;
		double rArea = (mFunction + fr) * (b - currentM) / 2;
		if (fabs((lArea + rArea) - currentArea) > epslon) {
			kernal << <1, 2 >> > (lArea, rArea, currentM, area, m, b,epslon,functionNumber);
		}
		else
			atomicAdd(area, lArea + rArea);
	}

}




int main() {
	cudaEvent_t event1, event2;
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);
	double a, b, epslon;//user inputs
	int  functionNum;// user input

	double fl = 0.0, fr = 0.0, area = 0.0, dx = 0.0, m = 0.0, //cpu
		* d_fl, * d_fr, * d_area, * d_dx, * d_m;			   // gbu
	double (*arrFunction[])(double*) = {xPlusOne,xSquarePlusOne,xCubePlusOne,sinPlusOne,cosPlusOne,expoOfTwoPlusOne,expoOfthreePlusOne,logOfTwoPlusOne,xCubePlusXSquarePlusOne};


	
	cudaMalloc((void**)&d_area, sizeof(double));
	

	printf("Welcome to cuda calculating area under the curve using \n\n      -Recursive Adaptive Quadrature Procedure-\n\n\n");
	printf("choose your function: \n1- x+1\n2- x^2+1\n3- x^3+1\n4- sin(x)+1\n5- cos(x)+1\n6- 2^x+1\n7- 3^x+1\n8- log2(x)+1\n9- x^3+x^2+1\n");
	scanf("%d", &functionNum);
	functionNum = functionNum - 1;
	printf("choose your upper value: ");
	scanf("%lf", &b);
	printf("\nchoose your lower value: ");
	scanf("%lf", &a);
	printf("\nchoose your epsilon value: ");
	scanf("%lf", &epslon);



	fl = (*arrFunction[functionNum])(&a);
	fr = (*arrFunction[functionNum])(&b);
	m = (b + a) / 2.0;
	area = (fl + fr) * (b - a) / 2;
	double fm = (*arrFunction[functionNum])(&m);
	double lArea = (fl + fm) * (m - a) / 2;
	double rArea = (fm + fr) * (b - m) / 2;
	area = 0.0;
	
	cudaMemcpy(d_area, &area, sizeof(double), cudaMemcpyHostToDevice);
	
	area = (fl + fr) * (b - a) / 2;

	if (fabs(((lArea + rArea) - area)) > epslon) {
		cudaEventRecord(event1);
		kernal << <1, 2 >> > (lArea, rArea, m, d_area, a, b,epslon,functionNum);
		cudaEventRecord(event2);
		cudaMemcpy(&area, d_area, sizeof(double), cudaMemcpyDeviceToHost);

	}
	else{
		area = lArea + rArea;
	}
	cudaDeviceSynchronize();

	cudaEventSynchronize(event2);
	float ms = 0;
	cudaEventElapsedTime(&ms, event1, event2);
	printf("\n\narea=%f\n", area);
	printf("Time taken of GBU: %f ms\n", ms);

	
	cudaFree(d_area);
	

	return 0;
}