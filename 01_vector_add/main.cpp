#define __CL_ENABLE_EXCEPTIONS

#include <cl.hpp>
#include <util.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <cstdio>

void printKernelInfo(const cl::Kernel& kernel, const cl::Device& device)
{
  //size_t kernelGlobalWorkSizes[3];
  size_t kernelWorkGroupSize;

  //kernel.getWorkGroupInfo(device, CL_KERNEL_GLOBAL_WORK_SIZE, kernelGlobalWorkSizes);
  kernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &kernelWorkGroupSize);

  //fprintf(stderr, "Kernel global work size: [0] = %d, [1] = %d, [2] = %d\r\n", kernelGlobalWorkSizes[0], kernelGlobalWorkSizes[1], kernelGlobalWorkSizes[2]);
  fprintf(stderr, "Kernel work group size: %lu\r\n", kernelWorkGroupSize);
  fprintf(stderr, "\r\n");
}
void printDeviceInfo(const cl::Device& device)
{
  std::string deviceName;
  cl_device_type deviceType;
  std::string deviceVersion;

  unsigned int deviceMaxComputeUnits;				// Nombre maximum d'unités de calculs parallèles (1 workgroup s'exécute sur une unité de calculs)
  unsigned int deviceMaxWorkItemDimensions;		// Nombre maximum de dimensions des workitems
  std::vector<size_t> deviceMaxWorkItemSizes;		// Nombre maximum de workitems assignable par dimension
  size_t deviceMaxWorkGroupSize;					// Nombre maximum de workitems p/ workgroup p/ unité de calcul

  size_t deviceLocalMemSize;			// Taille de l'espace mémoire local en octets (mémoire commune aux workitems d'un workgroup)
  size_t deviceGlobalMemSize;			// Taille de l'espace mémoire global en octets (mémoire commune à tous les workgroups)

  device.getInfo(CL_DEVICE_NAME, &deviceName);
  device.getInfo(CL_DEVICE_TYPE, &deviceType);
  device.getInfo(CL_DEVICE_VERSION, &deviceVersion);

  device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &deviceMaxComputeUnits);
  device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &deviceMaxWorkItemDimensions);
  device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &deviceMaxWorkItemSizes);
  device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &deviceMaxWorkGroupSize);

  device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &deviceLocalMemSize);
  device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &deviceGlobalMemSize);

  fprintf(stderr, "Device name: %s\r\n", deviceName.c_str());
  fprintf(stderr, "Device type: ");
  switch (deviceType)
  {
  case CL_DEVICE_TYPE_CPU: fprintf(stderr, "CPU\r\n"); break;
  case CL_DEVICE_TYPE_GPU: fprintf(stderr, "GPU\r\n"); break;
  case CL_DEVICE_TYPE_ACCELERATOR: fprintf(stderr, "ACCELERATOR\r\n"); break;
  case CL_DEVICE_TYPE_DEFAULT: fprintf(stderr, "DEFAULT\r\n"); break;
  default: fprintf(stderr, "N/A\r\n"); break;
  }
  fprintf(stderr, "Device version: %s\r\n", deviceVersion.c_str());

  fprintf(stderr, "Device max compute units:%d\r\n", deviceMaxComputeUnits);
  fprintf(stderr, "Device max workitem dimensions:%d\r\n", deviceMaxWorkItemDimensions);
  fprintf(stderr, "Device max workitem sizes: [0]=%lu, [1]=%lu, [2]=%lu\r\n", deviceMaxWorkItemSizes[0], deviceMaxWorkItemSizes[1], deviceMaxWorkItemSizes[2]);
  fprintf(stderr, "Device max workgroup size:%lu\r\n", deviceMaxWorkGroupSize);

  fprintf(stderr, "Device local memory size:%lu octets\r\n", deviceLocalMemSize);
  fprintf(stderr, "Device global memory size:%lu octets\r\n", deviceGlobalMemSize);
}
void printPlatformInfo(const cl::Platform& platform)
{
  std::string platformName;
  std::string platformVendor;
  std::string platformProfile;
  std::string platformVersion;
  std::string platformExtensions;

  platform.getInfo(CL_PLATFORM_NAME, &platformName);
  platform.getInfo(CL_PLATFORM_VENDOR, &platformVendor);
  platform.getInfo(CL_PLATFORM_PROFILE, &platformProfile);
  platform.getInfo(CL_PLATFORM_VERSION, &platformVersion);
  platform.getInfo(CL_PLATFORM_EXTENSIONS, &platformExtensions);

  fprintf(stderr, "Name: %s\r\n", platformName.c_str());
  fprintf(stderr, "Vendor: %s\r\n", platformVendor.c_str());
  fprintf(stderr, "Profile: %s\r\n", platformProfile.c_str());
  fprintf(stderr, "Version: %s\r\n", platformVersion.c_str());
  fprintf(stderr, "Extensions: %s\r\n", platformExtensions.c_str());
}
void printAllPlaformInfo()
{
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;

  fprintf(stderr, "Obtention des plateformes OpenCL disponibles... ");
  if (cl::Platform::get(&platforms) != CL_SUCCESS)
  {
    fprintf(stderr, "Impossible d'obtenir les plateformes OpenCL !\r\n");
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "OK\r\n");

  for (int i = 0; i < platforms.size(); ++i)
  {
    fprintf(stderr, "-------------------- Platform[%d] --------------------\r\n", i);
    printPlatformInfo(platforms[i]);

    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    for (int j = 0; j < devices.size(); ++j)
      printDeviceInfo(devices[j]);
  }

  fprintf(stderr, "------------------------------------------------------\r\n");
}

int main(int argc, char **argv)
{
  // Contexte
  printAllPlaformInfo();

  fprintf(stderr, "Initialisation du contexte OpenCL... ");

  cl::Context context(CL_DEVICE_TYPE_ALL);
  VECTOR_CLASS<cl::Device> devices;

  if (context.getInfo<VECTOR_CLASS<cl::Device> >(CL_CONTEXT_DEVICES, &devices) != CL_SUCCESS)
  {
    fprintf(stderr, "cl::Context::getInfo() error\r\n");
    return EXIT_FAILURE;
  }

  fprintf(stderr, "OK\r\n");

  fprintf(stderr, "\r\n");
  for (int i = 0; i < devices.size(); ++i)
  {
    fprintf(stderr, "--------------- Context Device[%d] ---------------\r\n", i);
    printDeviceInfo(devices[i]);
  }
  fprintf(stderr, "\r\n");

  // File de commandes
  const int queueDeviceId = 0;
  fprintf(stderr, "Initialisation d'une file de commandes pour le device %d... ", queueDeviceId);

  cl::CommandQueue queue(context, devices[queueDeviceId]);

  fprintf(stderr, "OK\r\n");
  fprintf(stderr, "\r\n");

  // Programme
  const std::string programFile = "vadd.cl";
  fprintf(stderr, "Chargement du programme '%s'... ", programFile.c_str());

  cl::Program program;
  try
  {
    program = cl::Program(context, util::loadProgram(programFile));
    program.build();
  }
  catch (...)
  {
    std::string log;

    program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);

    fprintf(stderr, "Echec\r\n");
    fprintf(stderr, "\r\n%s\r\n", log.c_str());

    return EXIT_FAILURE;
  }

  fprintf(stderr, "OK\r\n");

  // Kernel initialization & invocation
  const size_t vecLength = 4;
  const char* kernelName = "vadd";

  std::vector<float> h_a(vecLength);
  std::vector<float> h_b(vecLength);
  std::vector<float> h_c(vecLength);
  std::vector<float> h_d(vecLength);

  h_a[0] = 0;
  h_a[1] = 0;
  h_a[2] = 0;
  h_a[3] = 0;

  h_b[0] = 1;
  h_b[1] = 1;
  h_b[2] = 1;
  h_b[3] = 1;

  h_c[0] = -1;
  h_c[1] = -1;
  h_c[2] = -1;
  h_c[3] = -1;

  cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
  cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
  cl::Buffer d_c(context, h_c.begin(), h_c.end(), true);
  cl::Buffer d_d(context, CL_MEM_READ_WRITE, sizeof(float) * vecLength);

  cl::Kernel vaddKernel(program, kernelName);
  printKernelInfo(vaddKernel, devices[0]);

  cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> vaddFunc(vaddKernel);

  util::Timer timer;

  vaddFunc(cl::EnqueueArgs(queue, vecLength), d_a, d_b, d_c, d_d);

  queue.finish();

  cl::copy(queue, d_d, h_d.begin(), h_d.end());

  printf("Kernel '%s' execute en %ld ms\r\n", kernelName, timer.getTimeMilliseconds());

  for (int i = 0; i < 4; ++i)
    printf("h_d[%d] = %f\r\n", i, h_d[i]);

  return EXIT_SUCCESS;
}
