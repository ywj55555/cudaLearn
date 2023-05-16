#include <stdio.h>
int main()
{
   int nDevices;

   cudaGetDeviceCount(&nDevices);
   for (int i = 0; i < nDevices; i++)
   {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Num: %d\n", i);
      printf("Device name: %s\n", prop.name);
      printf("Device SM Num: %d\n", prop.multiProcessorCount);
      printf("Maximum number of registers available per block: %d\n", prop.regsPerBlock);
      // printf("Maximum shared memory available per block in bytes: %d\n", prop.sharedMemPerBlock);
      printf("Share Mem Per Block: %.2fKB\n", prop.sharedMemPerBlock / 1024.0);
      printf("Max Thread Per Block: %d\n", prop.maxThreadsPerBlock);
      printf("Memory Clock Rate (KHz): %d\n",
             prop.memoryClockRate);
      printf("Memory Bus Width (bits): %d\n",
             prop.memoryBusWidth);
      printf("Peak Memory Bandwidth (GB/s): %.2f\n\n",
             2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
   }
   return 0;
}
