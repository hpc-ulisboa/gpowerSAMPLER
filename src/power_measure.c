#include "power_measure.h"

#define ADDITIONAL_INFO "\n\
ADDITIONAL INFORMATION\n\
\n\
  Energy estimates are obtained by multiplying the Power readings by the sampling time.\n\
\n\
[1] Burtscher, Martin and Zecena, Ivan and Zong, Ziliang. Measuring GPU Power with the K20 Built-in Sensor.\n\
    Workshop on General Purpose Processing Using GPUs, (2014). Page 28. ACM.\n"

#define HELP_INFO "\n\
UTILIZATION\n\
\n\
  With power_measure the user is able to get information of the GPU's power and\n\
  energy real-time consumption.\n\
\n\
  Usage:\n\
    power_measure [-d device_id]  [-e] [-t power_level] [-s sample_time_ms] [-f end_time_s] [-w] -a application_to_monitor\n\
    power_measure -h\n\
    power_measure -i\n\
\n\
  Options:\n\
\n\
    -d : device id\n\
\n\
    -e : calculates the energy consumed\n\
\n\
    -t : indicates a GPU power value, above which the power samples will be considered\n\
\n\
    -s : sample time in milliseconds ( default is 16 ). This time refers to the time interval between measurements.\n\
\n\
    -f : time interval in seconds ( default is 0 ) to continue sampling the power readings after the application has GPU kernels have finished.\n\
\n\
    -a : application to be executed, containing one or multiple GPU kernels.\n\
\n\
    -w : samples the performance state, instead of the power consumption.\n\
\n\
    -h : shows 'USAGE' section of this file\n\
\n\
    -i : shows 'ADITIONAL INFORMATION' section of this file\n\n"


/*+++++++++++++++++++++++++++++++++++++++++++++++++++
 *                  POWER_MEASURE VARIABLES         +
 *++++++++++++++++++++++++++++++++++++++++++++++++++*/

int terminate; //END PROGRAM
nvmlDevice_t nvDevice;
nvmlReturn_t nvResult;
double times[SAMPLE_MAX_SIZE_DEFAULT];
// lld elapsedTimes[SAMPLE_MAX_SIZE_DEFAULT];
unsigned int powers[SAMPLE_MAX_SIZE_DEFAULT];
nvmlPstates_t pStates[SAMPLE_MAX_SIZE_DEFAULT];
unsigned int gpuUtil[SAMPLE_MAX_SIZE_DEFAULT];
unsigned int memoryUtil[SAMPLE_MAX_SIZE_DEFAULT];
int n_values;
int begin_interval; //Start taking samples
int end_interval; //finish taking samples
int deviceID; //Device id
int i_begin, i_end; //INSTANTS WHEN THE SAMPLING BEGINS AND ENDS
pid_t           appPid;
int             sampleTimeMs; // sample time in ms (if -s)
int             threshold;     // threshold value in W indicating the power ranges when gpu is active (above threshold)
int             endTimeS;	  // end time in s (if -f)
int             sFlag;        // if -gt 0, sample time option activated
int             dFlag;        // if -gt 0, deviceID option activated
int             aFlag;        // if -gt 0, application option activated
int             fFlag;        // if -gt 0, sample after finish option activated
int             eFlag;        // if -gt 0, energy option activated
int             tFlag;         // if -gt 0, threshold option activated
int             wFlag;        // if -gt 0, performance state option activated
char*           appPath;      // application path
char            **appArgs;      // application arguments in a Vector
char            appArgsS[1024];      // application arguments combined in a string

pthread_t thread_sampler;

unsigned int count_mem;
unsigned int count_graph;
unsigned int clocks_mem[10];
unsigned int clocks_graph[10];

int core_clock_id, mem_clock_id;
unsigned int core_clock, mem_clock;



/*+++++++++++++++++++++++++++++++++++++++++++++++++++
 *              POWER_MEASURE FUNCTIONALITY         +
 *++++++++++++++++++++++++++++++++++++++++++++++++++*/

void sig_handler(int sig) {
    if(sig == SIGUSR1) {
		return;
	}
	if(sig == SIGUSR2) {
		return;
	}
}

void *threadWork(void * arg) {
	unsigned int power;
	// double start, aux;
	int i=0;
	FILE *fp2;
	struct timespec time_aux;
    struct timeval tv_start, tv_aux;
    nvmlUtilization_t util;

	time_aux.tv_sec=0;
	time_aux.tv_nsec=sampleTimeMs*1000000;

	if (aFlag == 0) fp2 = fopen("out_power_measured_interval.csv", "w");

	if (aFlag == 0) fprintf(fp2,"sep=;\nTimestamp [us];Power measure [mW]");
	// start = PAPI_get_real_usec();
    gettimeofday(&tv_start,NULL);

	while (!terminate) {
		//GET POWER SAMPLE
        if (wFlag == 0)
		    nvResult = nvmlDeviceGetPowerUsage(nvDevice, &power);
        else
            nvResult = nvmlDeviceGetUtilizationRates (nvDevice, &util);

		if (NVML_SUCCESS != nvResult) {
			printf("Failed to get power usage: %s\n", nvmlErrorString(nvResult));
			pthread_exit(NULL);
		}

		if(i < SAMPLE_MAX_SIZE_DEFAULT) {
			// aux = PAPI_get_real_usec(); //GET TIMESTAMP
            gettimeofday(&tv_aux,NULL);

            times[i] = (tv_aux.tv_sec - tv_start.tv_sec)*1000000;
            times[i] += (tv_aux.tv_usec - tv_start.tv_usec);
            if (wFlag == 0)
                powers[i] = power;
            else {
                gpuUtil[i] = util.gpu;
                memoryUtil[i] = util.memory;
            }

			if (end_interval == 0 && begin_interval == 1) {
				if (i_begin == -1)
					i_begin = i;
				if (aFlag == 0) fprintf(fp2, "\n%f; %d", times[i],  powers[i]);
			} else {
				if (i_end == -1 && begin_interval == 1) {
					i_end = i;
				}
			}
		}
		else {
			printf("ERROR: POWER VECTOR SIZE EXCEEDED!\n");
			pthread_exit(NULL);
		}

		i++;
		n_values = i;
		nanosleep(&time_aux, NULL);
	}
	if (i_end == -1) i_end = i-1;
	if (aFlag == 0) fclose(fp2);
	pthread_exit(NULL);
}


float correctedPowerSerial() {
	int i, values_threshold=0;
	float acc3 = 0.0;
	float acc4 = 0.0;
	float p_average;
	float p_average_interval;
	float energy;
	float energy_interval;
	double interval;
    double interval_GPU;
    int begin_gpu=-1, end_gpu=n_values-1;
	float peak=0;
	FILE  *fp, *fp2;
    if (wFlag == 0) {
	    fp = fopen("out_power.txt", "a");
	    fp2 = fopen("out_power_samples.csv", "w");
    } else
        fp2 = fopen("out_util_samples.csv", "w");

	fprintf(fp2,"sep=;\nTimestamp [us];Power measure [W]");

	for(i=0; i<n_values;i++) {
        if ( wFlag == 0 ) fprintf(fp2, "\n%f;%.4f", times[i], powers[i]/1000.0);
        else fprintf(fp2, "\n%f;%d;%d", times[i], gpuUtil[i], memoryUtil[i]);
		if ( aFlag == 0 ) {
			if (i >= i_begin && i <= i_end) {
				acc3 = acc3 + (powers[i]);
			}
		}
        if (wFlag == 0 && powers[i] > peak) {
			peak = powers[i];
		}

        if ( powers[i]/1000.0 > 35 && begin_gpu == -1 ) begin_gpu = i;
        if ( powers[i]/1000.0 < 35 && begin_gpu != -1 ) end_gpu = i;

        if (wFlag == 0 && ((tFlag == 0) || powers[i]/1000.0 >= threshold)) {
    		acc4 = acc4 + powers[i];
            values_threshold++;
        }
	}
    if (wFlag == 0) {
    	if (values_threshold>0) {
            p_average = acc4 / (values_threshold*1.0);
    		if ( aFlag == 0 ) p_average_interval = acc3 / ((i_end-i_begin)*1.0);
    	} else {
    		printf("ERROR: DIVISION BY 0\n");
    		exit(-1);
    	}
    }

	interval = times[n_values-1] - times[0];
    interval_GPU = times[end_gpu] - times[begin_gpu];
	if ( wFlag == 0 && eFlag == 1 ) {
		energy = (p_average/1000.0) * (interval/1000000.0);
		if ( aFlag == 0 ) {
			energy_interval = (p_average_interval / 1000.0) * ((times[i_end]-times[i_begin])/1000000.0);
		}
	}

    if (wFlag == 0) printf("\tAt current frequency (%d,%d) MHz:  Average Power: %.2f W;  Max Power: %.2f W;  Sampling Duration: %.2f ms;  GPU active duration: %.2f ms", mem_clock, core_clock, p_average/1000.0, peak/1000.0, (interval)/1000, interval_GPU/1000);
	if (wFlag == 0) {
        if ( eFlag == 1) printf("\tEnergy: %.3f J\n\n", energy);
	    else printf("\n\n");
    }

    if (wFlag == 0) fprintf(fp,"Average Power [W],%.2f,Max Power [W],%.2f,Sampling Duration [ms],%f, Current frequency,(%d,%d) MHz,GPU active duration [ms],%f", p_average/1000.0, peak/1000.0, (interval)/1000, mem_clock, core_clock, interval_GPU/1000);
	if (wFlag == 0) {
        if ( eFlag == 1) fprintf(fp,"\tEnergy: %.3f J\n", energy);
        else fprintf(fp,"\n");
    }

	if (wFlag == 0 && aFlag == 0 ) {
		printf("\t\tAverage Power during interval: %.2f W\tSampling Time: %f ms", p_average_interval/1000, (times[i_end]-times[i_begin])/1000);
		if ( eFlag == 1) printf("\tEnergy: %.3f J\n\n", energy_interval);
		else printf("\n\n");
		fprintf(fp,"Average Power during interval [W],%.2f,Sampling Time [ms],%f", p_average_interval/1000, (times[i_end]-times[i_begin])/1000);
		if ( eFlag == 1) fprintf(fp,"\tEnergy: %.3f J\n", energy_interval);
		else fprintf(fp,"\n");
	}

    if (wFlag == 0) fclose(fp);

    fclose(fp2);

	return energy;
}


int ParseOptions ( int argc, char** argv )
{
  extern char *optarg;
  extern int optind;
  int c;
  int err = 0;
  const char usage[] = "Usage: %s [-e] [-d device_id] [-t power_level] [-s sample_time_ms] [-f end_time_s] [-w] -a application_path\nType 'power_measure -h' for help.\n";
  int i=0;
  int aux=2;

  // Defaults
  sampleTimeMs = SAMPLE_TIME_MS_DEFAULT;
  endTimeS = END_TIME_S_DEFAULT;
  deviceID = 0;
  appPath = NULL;

  while ( (c = getopt(argc, argv, "hcirewd:s:a:f:p:t:")) != -1 )
  {
    i++;
    switch (c)
    {
      case 'h':
        printf ( HELP_INFO );
        return 1;
        break;
      case 'i':
        printf ( ADDITIONAL_INFO );
        return 2;
        break;
	case 'e':
		if ( eFlag == 1)
		{
          fprintf ( stderr, "%s: warning: -e is set multiple times.\n", argv[0] );
        }
        eFlag = 1;
        aux++;
		break;
      case 's':
        if ( sFlag == 1 )
        {
          fprintf ( stderr, "%s: warning: -s is set multiple times.\n", argv[0] );
        }
        sFlag = 1;
        sampleTimeMs = atoi (optarg);
        aux++;
        aux++;
        break;
      case 'w':
        if ( wFlag == 1 )
        {
          fprintf ( stderr, "%s: warning: -w is set multiple times.\n", argv[0] );
        }
        wFlag = 1;
        aux++;
        break;
      case 'd':
        if ( dFlag == 1 )
        {
        fprintf ( stderr, "%s: warning: -d is set multiple times.\n", argv[0] );
        }
        dFlag = 1;
        deviceID = atoi (optarg);
        aux++;
        aux++;
        break;
      case 't':
        if ( tFlag == 1 )
        {
        fprintf ( stderr, "%s: warning: -t is set multiple times.\n", argv[0] );
        }
        tFlag = 1;
        threshold = atoi (optarg);
        aux++;
        aux++;
        break;
	  case 'f':
        if ( fFlag == 1 )
        {
          fprintf ( stderr, "%s: warning: -f is set multiple times.\n", argv[0] );
        }
        fFlag = 1;
        endTimeS = atoi (optarg);
        aux++;
        aux++;
        break;
	  case 'a':
		if ( aFlag == 1 )
        {
          fprintf ( stderr, "%s: warning: -a is set multiple times.\n", argv[0] );
        }
        aFlag = 1;
        appPath = optarg;
        int j;
        appArgs=(char**)malloc((argc-aux+1)*sizeof(char*));
        for (j=aux+1;j<argc+1;j++) {
            appArgs[j-aux-1]=(char*)malloc(sizeof(char)*500);
            appArgs[j-aux-1]=argv[j-1];
            sprintf(appArgsS,"%s %s",appArgsS, argv[j-1]);
        }
        appArgs[argc-aux]=NULL;
		return 0;
      default:
        err = 1;
        break;
    }
  }

  if ( err )
  {
    fprintf ( stderr, usage, argv[0] );
    return -1;
  }

  return 0;
}

int initialize() {
	unsigned int device_count;
	// unsigned int clock;
	int a;
     int major;

    // int deviceNum = 0;
    // CUdevice device = 0;
    // cudaDeviceProp deviceProp;
    //
    // cuDeviceGet(&device, deviceNum);
	// cudaGetDeviceProperties(&deviceProp, device);
    CUresult result;
    CUdevice device=0;

    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        printf("Error code %d on cuInit\n", result);
        exit(-1);
    }
    result = cuDeviceGet(&device,0);
    if (result != CUDA_SUCCESS) {
        printf("Error code %d on cuDeviceGet\n", result);
        exit(-1);
    }

    result = cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (result != CUDA_SUCCESS) {
        printf("Error code %d on cuDeviceGetAttribute\n", result);
        exit(-1);
    }

	i_begin = -1;
	i_end = -1;

	terminate = 0;
	begin_interval = 0;
	end_interval = 0;

	// NVML INITIALIZATIONS
	nvResult = nvmlInit();
	if (NVML_SUCCESS != nvResult)
    {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(nvResult));

        printf("Press ENTER to continue...\n");
        getchar();
        return -1;
    }

	nvResult = nvmlDeviceGetCount(&device_count);
    if (NVML_SUCCESS != nvResult)
    {
        printf("Failed to query device count: %s\n", nvmlErrorString(nvResult));
        return -1;
    }

	printf("Found %d device%s\n\n", device_count, device_count != 1 ? "s" : "");
    if (deviceID >= device_count) {
        printf("Device_id is out of range.\n");
        return -1;
    }
	nvResult = nvmlDeviceGetHandleByIndex(deviceID, &nvDevice);
	if (NVML_SUCCESS != nvResult)
	{
		printf("Failed to get handle for device 1: %s\n", nvmlErrorString(nvResult));
		 return -1;
	}
	nvmlDeviceGetApplicationsClock  ( nvDevice, NVML_CLOCK_GRAPHICS, &core_clock);

    // nvmlDeviceGetApplicationsClock  ( nvDevice, NVML_CLOCK_SM, &clock);
    // printf("Current SM clock: %d\n", clock);
    nvmlDeviceGetApplicationsClock  ( nvDevice, NVML_CLOCK_MEM, &mem_clock);


	//LAUNCH POWER SAMPLER
	a = pthread_create(&thread_sampler, NULL, threadWork, NULL);
	if(a) {
		fprintf(stderr,"Error - pthread_create() return code: %d\n",a);
		return -1;
	}

	return 0;
}

int endProgram() {
	terminate = 1;
	pthread_join(thread_sampler, NULL);
	correctedPowerSerial();
    if (wFlag == 0) {
        if (aFlag == 0) printf("File out_power_measured_interval.csv created with measured power samples in defined interval.\n");
        printf("File out_power_samples.csv created with all measured samples.\n");
        printf("File out_power.txt average power consumption.\n");
    } else {
        printf("File out_util_samples.csv created with measured utilization samples in defined interval.\n");
    }

	printf("Finished\n\n");

	return 0;
}

int main(int argc, char ** argv) {

	struct timespec time_aux;
	pid_t pid;
	int a;



	// Parse input arguments
	a = ParseOptions ( argc, argv );
	if ( a != 0 ) _exit(a);

		time_aux.tv_nsec=0;
		time_aux.tv_sec=endTimeS;

    // Initializations
	if ( initialize() != 0 ) {
		fprintf ( stderr, "%s: error: initializing...\n", argv[0] );
		_exit (1);
	}

	printf("\n\n*************************************************\n");
	printf("Starting gpowerSAMPLER!\n\n");
	printf("Execution parameters:\n");
    printf("\tDevice id: %d\n", deviceID);
    printf("\tCurrent graphics clock: %d (id: %d)\n", core_clock, core_clock_id);
    printf("\tCurrent memory clock: %d (id: %d)\n", mem_clock, mem_clock_id);
	printf("\tSampling period: %d ms\n", sampleTimeMs);
	if ( aFlag == 1) {
         printf("\tApplication to monitor: %s\n", appPath);
         printf("\tArguments for the application to monitor: %s\n", appArgsS);
     }
     if (wFlag == 0) {
        // printf("\tApplying threshold filtering to the measured values: %s\n", tFlag == 1 ? "yes" : "no");
        printf("\tApplying threshold filtering to the measured values: %s\n", (tFlag == 1) ? "yes" : "no");

        if ( tFlag == 1 ) printf("\tThreshold value: %d W\n", threshold);

    	printf("\tKeep sampling for %d s after finishing application.\n", endTimeS);
    	printf("\tCalculate energy consumed: %s\n", eFlag == 1 ? "yes" : "no");
    } else {
        printf("\tSample utilization values: %s\n", wFlag == 1 ? "yes" : "no");
    }
	printf("\n");
	if (aFlag == 0) {
		signal(SIGUSR1, sig_handler); //SIGNAL TO FINISH THE SAMPLING
		signal(SIGUSR2, sig_handler); //SIGNAL TO START THE SAMPLING
	}



	//Start monitoring
	if (aFlag == 0)
		pause();

	begin_interval = 1;
	printf("Starting the sampling\n");

	//Launch application if application was given
	if (aFlag == 1) {
		if(!(pid = fork())) {
			printf("Launching application to monitor\n");
			printf("\n************** APPLICATION OUTPUT ***************\n");
			execv (appPath, appArgs);
			printf("ERROR: EXECL on process %s failed\n", appPath);
			exit(-1);
		}
		appPid = pid;
	}


	//Wait for application to finish
	if (aFlag == 0)
		pause(); //WAIT for the second signal, signalling that the power samples are over
	else {
		waitpid(appPid, NULL, 0);
		printf("\n*************************************************\n");
	}
	end_interval = 1;

	printf("Sampling finished\n");

	//Keep sampling in the given interval
	nanosleep(&time_aux, NULL);

	// Terminate
	if ( endProgram() != 0 )
	{
		fprintf ( stderr, "%s: error: terminating...\n", argv[0] );
		_exit (1);
	}

	return 0;
}
