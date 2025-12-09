#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "params.h"
#include "yuvRead.h"
#include "displayRGB.h"
#include "yuv2RGB.h"
#include "rgb2Gray.h"
#include "census.h"
#include "costConstruction.h"
#include "offsetGen.h"
#include "computeWeights.h"
#include "aggregateCost.h"
#include "disparitySelect.h"
#include "medianFilter.h"
#include "md5.h"
#include <omp.h>

int stopThreads = 0;
int Frame = 0;
double T;
int main(void) {
	printf("Stereo Matching App\n");
	omp_set_num_threads(14);
	// Open YUV Files (left & right)
	initReadYUV(0, WIDTH, HEIGHT);
	initReadYUV(1, WIDTH, HEIGHT);

	// Init display
	displayRGBInit(0, HEIGHT, WIDTH);
	displayRGBInit(1, HEIGHT, WIDTH);

	while (!stopThreads) {
		// Read images
		static unsigned char yL[HEIGHT * WIDTH], uL[HEIGHT * WIDTH / 4], vL[HEIGHT * WIDTH / 4];
		static unsigned char yR[HEIGHT * WIDTH], uR[HEIGHT * WIDTH / 4], vR[HEIGHT * WIDTH / 4];
		#pragma omp parallel sections shared(yL,yR,uL,uR,vL,vR) num_threads(2)
				{
					#pragma omp section
					{
						readYUV(0, WIDTH, HEIGHT, yL, uL, vL);
					}
					#pragma omp section 
					{
						readYUV(1, WIDTH, HEIGHT, yR, uR, vR);
					}
				}
	

		// Convert images to RGB
		static unsigned char rgbL[HEIGHT * WIDTH * 3], rgbR[HEIGHT * WIDTH * 3];
		double t_0 = omp_get_wtime();
		yuv2rgb(WIDTH, HEIGHT, yL, uL, vL, rgbL);
		//double t_1 = omp_get_wtime();
		//T += t_1 - t_0;
		//t_0 = omp_get_wtime();
		yuv2rgb(WIDTH, HEIGHT, yR, uR, vR, rgbR);
		//t_1 = omp_get_wtime();
		//T += t_1 - t_0;
		// Convert to gray
		static float grayL[HEIGHT * WIDTH], grayR[HEIGHT * WIDTH];
		//double t_0 = omp_get_wtime();
		rgb2Gray(HEIGHT * WIDTH, rgbL, grayL);
		//double t_1 = omp_get_wtime();
		//T += t_1 - t_0;
		//t_0 = omp_get_wtime();
		rgb2Gray(HEIGHT * WIDTH, rgbR, grayR);
		//t_1 = omp_get_wtime();
		//T += t_1 - t_0;
	
		// Census
		static unsigned char cenL[HEIGHT * WIDTH], cenR[HEIGHT * WIDTH];
		//double t_0 = omp_get_wtime();
		census(HEIGHT, WIDTH, grayL, cenL);
		//double t_1 = omp_get_wtime();
		//T += t_1 - t_0;
		//t_0 = omp_get_wtime();
		census(HEIGHT, WIDTH, grayR, cenR);
		//t_1 = omp_get_wtime();
		//T += t_1 - t_0;
	
		// Pre-compute weights for offset aggregation
		int offsets[NB_ITERATIONS];
		static float weightsHor[NB_ITERATIONS * HEIGHT * WIDTH * 3], weightsVert[NB_ITERATIONS * HEIGHT * WIDTH * 3];
		offsetGen(NB_ITERATIONS, offsets);
		for (unsigned idx = 0; idx < NB_ITERATIONS; idx++) {
			//double t_0 = omp_get_wtime();
			computeWeights(HEIGHT, WIDTH, 0, offsets + idx, rgbL, weightsHor + idx * (3 * HEIGHT * WIDTH));
			//double t_1 = omp_get_wtime();
			//T += t_1 - t_0;
			//t_0 = omp_get_wtime();
			computeWeights(HEIGHT, WIDTH, 1, offsets + idx, rgbL, weightsVert + idx * (3 * HEIGHT * WIDTH));
			//t_1 = omp_get_wtime();
			//T += t_1 - t_0;
		}

		// Find for each pixel, the disparity level minimizing the aggregated costs.
		static unsigned char depthMap[HEIGHT * WIDTH];
		memset(depthMap, 0, HEIGHT * WIDTH*sizeof(char));
		static float bestCost[HEIGHT * WIDTH];

		// For each degree of disparity
		for (char disp = MIN_DISPARITY; disp <= MAX_DISPARITY; disp++) {

			// Cost construction
			static float dispError[HEIGHT * WIDTH];
			//double t_0 = omp_get_wtime();
			costConstruction(HEIGHT, WIDTH, 12 /*Magic number*/, &disp, grayL, grayR, cenL, cenR, dispError);
			//double t_1 = omp_get_wtime();
			//T += t_1-t_0;
			static float aggregatedDisparityCost[HEIGHT * WIDTH];
			//double t_0 = omp_get_wtime();
			aggregateCost(HEIGHT, WIDTH, NB_ITERATIONS, dispError, offsets, weightsHor, weightsVert, aggregatedDisparityCost);
			//double t_1 = omp_get_wtime();
			//T += t_1-t_0;
			if (disp == MIN_DISPARITY) {
				memcpy(bestCost, aggregatedDisparityCost, HEIGHT * WIDTH * sizeof(float));
			}
			else {
				// Compare the current disparity cost to previous ones
				//double t_0 = omp_get_wtime();
				disparitySelect(HEIGHT, WIDTH, 12, MIN_DISPARITY, &disp, aggregatedDisparityCost, bestCost, depthMap);
				//double t_1 = omp_get_wtime();
				//T += t_1-t_0;
			}
		}

		// Apply median filter on result
		static unsigned char filteredDepthMap[HEIGHT * WIDTH];
		//double t_0 = omp_get_wtime();
		medianFilter(HEIGHT, WIDTH, 1, depthMap, filteredDepthMap);
		//double t_1 = omp_get_wtime();
		//T += t_1-t_0;

		// Display
		displayRGB(0, HEIGHT, WIDTH, rgbL);
		displayLum(1, filteredDepthMap);

		// MD5
		MD5_Update(HEIGHT* WIDTH * sizeof(char), filteredDepthMap);
		/*Frame++;
		if (Frame % 70 == 0)
		{
			printf("temps passé par la fonction est = %.3f ms", (T / Frame)*1000);
		}*/
	}

	return 0;
}