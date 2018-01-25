#ifndef _POWERMEASURE_H_
#define _POWERMEASURE_H_

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <nvml.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <sys/time.h>

#include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>

// #include <cuda/cuda.h>
// #include <cuda/cuda_runtime_api.h>

#define SAMPLE_TIME_MS_DEFAULT  16
#define END_TIME_S_DEFAULT  0
#define SAMPLE_MAX_SIZE_DEFAULT 5000
#define CORRECTION_PARAM_K20 833
#define IDLE_POWER_MAX_THRESHOLD 5

/*+++++++++++++++++++++++++++++++++++++++++++++++++++
 *              POWER_MEASURE FUNCTIONALITY         +
 *++++++++++++++++++++++++++++++++++++++++++++++++++*/

 void sig_handler(int);
 void *threadWork(void*);
 float correctedPowerSerial();
 int ParseOptions ( int , char** );
 int initialize();
 int endProgram();
 int main();

#endif
