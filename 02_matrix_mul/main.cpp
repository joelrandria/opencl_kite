#define __CL_ENABLE_EXCEPTIONS

#include <cl.hpp>
#include <util.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <cstdio>

/* ========== Platform/Kernel infos ========== */

void printKernelInfo(const cl::Kernel& kernel, const cl::Device& device)
{
  //size_t kernelGlobalWorkSizes[3];
  size_t kernelWorkGroupSize;
  cl_ulong kernelPrivateMemSize;

  //kernel.getWorkGroupInfo(device, CL_KERNEL_GLOBAL_WORK_SIZE, kernelGlobalWorkSizes);
  kernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &kernelWorkGroupSize);
  //kernel.getWorkGroupInfo(device, CL_KERNEL_PRIVATE_MEM_SIZE, &kernelPrivateMemSize);

  //fprintf(stderr, "Kernel global work size: [0] = %d, [1] = %d, [2] = %d\r\n", kernelGlobalWorkSizes[0], kernelGlobalWorkSizes[1], kernelGlobalWorkSizes[2]);
  fprintf(stderr, "Kernel work group size: %lu\r\n", kernelWorkGroupSize);
  //fprintf(stderr, "Kernel private memory size: %lu bytes\r\n", kernelPrivateMemSize);
  fprintf(stderr, "\r\n");
}
void printDeviceInfo(const cl::Device& device)
{
  std::string deviceName;
  cl_device_type deviceType;
  std::string deviceVersion;

  unsigned int deviceMaxComputeUnits;		// Nombre maximum d'unités de calculs parallèles (1 workgroup s'exécute sur une unité de calculs)
  unsigned int deviceMaxWorkItemDimensions;	// Nombre maximum de dimensions des workitems
  std::vector<size_t> deviceMaxWorkItemSizes;	// Nombre maximum de workitems assignable par dimension
  size_t deviceMaxWorkGroupSize;		// Nombre maximum de workitems p/ workgroup p/ unité de calcul

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

/* ========== Matrix operations ========== */

bool isIdentity(const int size, const std::vector<float>& m)
{
  int r;
  int c;

  for (r = 0; r < size; ++r)
    for (c = 0; c < size; ++c)
      if (r == c && m[r * size + c] != 1)
	return false;
      else if (r != c && m[r * size + c] != 0)
	return false;

  return true;
}
void setIdentity(const int size, std::vector<float>& m)
{
  int r;
  int c;

  for (r = 0; r < size; ++r)
    for (c = 0; c < size; ++c)
      m[r * size + c] = (r == c) ? 1 : 0;
}
void setNull(std::vector<float>& m)
{
  int i;

  for (i = 0; i < m.size(); ++i)
    m[i] = 0;
}
void printMatrix(const int size, const std::vector<float>& m)
{
  int r;
  int c;

  for (r = 0; r < size; ++r)
  {
    for (c = 0; c < size; ++c)
      printf("%f\t", m[r * size + c]);
    printf("\r\n");
  }
}

/* ========== Kernel executions ========== */

int main(int argc, char **argv)
{
  printAllPlaformInfo();

  // Context
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

  // Command queue
  const int queueDeviceId = 0;
  fprintf(stderr, "Initialisation d'une file de commandes pour le device %d... ", queueDeviceId);

  cl::CommandQueue queue(context, devices[queueDeviceId]);

  fprintf(stderr, "OK\r\n");
  fprintf(stderr, "\r\n");

  // Program build
  const std::string programFile = "mmul.cl";
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
  fprintf(stderr, "\r\n");

  // Kernel arguments initialization
  const int matrixOrder = 1024;
  const int matrixTotalSize = matrixOrder * matrixOrder;

  std::vector<float> h_m1(matrixTotalSize);
  std::vector<float> h_m2(matrixTotalSize);

  std::vector<float> h_r1(matrixTotalSize);
  std::vector<float> h_r2(matrixTotalSize);
  std::vector<float> h_r3(matrixTotalSize);
  std::vector<float> h_r4(matrixTotalSize);

  setIdentity(matrixOrder, h_m1);
  setIdentity(matrixOrder, h_m2);

  setNull(h_r1);
  setNull(h_r2);
  setNull(h_r3);
  setNull(h_r4);

  cl::Buffer d_m1(context, h_m1.begin(), h_m1.end(), true);
  cl::Buffer d_m2(context, h_m2.begin(), h_m2.end(), true);

  cl::Buffer d_r1(context, CL_MEM_READ_WRITE, sizeof(float) * matrixTotalSize);
  cl::Buffer d_r2(context, CL_MEM_READ_WRITE, sizeof(float) * matrixTotalSize);
  cl::Buffer d_r3(context, CL_MEM_READ_WRITE, sizeof(float) * matrixTotalSize);
  cl::Buffer d_r4(context, CL_MEM_READ_WRITE, sizeof(float) * matrixTotalSize);

  util::Timer timer;

  std::string kernelName;
  cl::Kernel matrixMulKernel;

  // ---------- Kernel #1: C(i,j) p/ work item (NxN work items), Global memory ----------
  kernelName = "mmul_cij_gmem";

  printf("---------- C(i,j) p/ work item (NxN work items), Global memory ----------\r\n");
  printf("\r\n");

  matrixMulKernel = cl::Kernel(program, kernelName.c_str());
  printKernelInfo(matrixMulKernel, devices[0]);

  cl::make_kernel<int, int, cl::Buffer,
		  int, int, cl::Buffer,
		  cl::Buffer> matrixMulFunc1(matrixMulKernel);

  matrixMulFunc1(cl::EnqueueArgs(queue, cl::NDRange(matrixOrder, matrixOrder)),
		 matrixOrder, matrixOrder, d_m1,
		 matrixOrder, matrixOrder, d_m2,
		 d_r1);

  timer.reset();
  queue.finish();

  cl::copy(queue, d_r1, h_r1.begin(), h_r1.end());

  printf("Resultat: %s\r\n", isIdentity(matrixOrder, h_r1) ? "OK" : "ERREUR");
  printf("Kernel '%s' execute en %lu us\r\n", kernelName.c_str(), timer.getTimeMicroseconds());
  printf("\r\n");

  // ---------- Kernel #2: C(i,*) p/ work item (N work items), Global memory ----------
  kernelName = "mmul_ci_gmem";

  printf("---------- C(i,*) p/ work item (N work items), Global memory ----------\r\n");
  printf("\r\n");

  matrixMulKernel = cl::Kernel(program, kernelName.c_str());
  printKernelInfo(matrixMulKernel, devices[0]);

  cl::make_kernel<int, int, cl::Buffer,
		  int, int, cl::Buffer,
		  cl::Buffer> matrixMulFunc2(matrixMulKernel);

  matrixMulFunc2(cl::EnqueueArgs(queue, cl::NDRange(matrixOrder), cl::NDRange(matrixOrder / 4)),
		 matrixOrder, matrixOrder, d_m1,
		 matrixOrder, matrixOrder, d_m2,
		 d_r2);

  timer.reset();
  queue.finish();

  cl::copy(queue, d_r2, h_r2.begin(), h_r2.end());

  printf("Resultat: %s\r\n", isIdentity(matrixOrder, h_r2) ? "OK" : "ERREUR");
  printf("Kernel '%s' execute en %lu us\r\n", kernelName.c_str(), timer.getTimeMicroseconds());
  printf("\r\n");

  // ---------- Kernel #3: C(i,*) p/ work item (N work items), Row in private memory ----------
  kernelName = "mmul_ci_pmemr_gmemc";

  printf("---------- C(i,*) p/ work item (N work items), Row in private memory ----------\r\n");
  printf("\r\n");

  matrixMulKernel = cl::Kernel(program, kernelName.c_str());
  printKernelInfo(matrixMulKernel, devices[0]);

  cl::make_kernel<int, int, cl::Buffer,
		  int, int, cl::Buffer,
		  cl::Buffer> matrixMulFunc3(matrixMulKernel);

  matrixMulFunc3(cl::EnqueueArgs(queue, cl::NDRange(matrixOrder), cl::NDRange(matrixOrder / 4)),
		 matrixOrder, matrixOrder, d_m1,
		 matrixOrder, matrixOrder, d_m2,
		 d_r3);

  timer.reset();
  queue.finish();

  cl::copy(queue, d_r3, h_r3.begin(), h_r3.end());

  printf("Resultat: %s\r\n", isIdentity(matrixOrder, h_r3) ? "OK" : "ERREUR");
  printf("Kernel '%s' execute en %lu us\r\n", kernelName.c_str(), timer.getTimeMicroseconds());
  printf("\r\n");

  // ---------- Kernel #4: C(i,*) p/ work item (N work items), Private row, Local column ----------
  kernelName = "mmul_ci_pmemr_lmemc";

  printf("---------- C(i,*) p/ work item (N work items), Private row, Local column ----------\r\n");
  printf("\r\n");

  matrixMulKernel = cl::Kernel(program, kernelName.c_str());
  printKernelInfo(matrixMulKernel, devices[0]);

  cl::make_kernel<int, int, cl::Buffer,
		  int, int, cl::Buffer,
		  cl::Buffer,
		  cl::LocalSpaceArg> matrixMulFunc4(matrixMulKernel);

  cl::LocalSpaceArg localColumnBuffer = cl::Local(sizeof(float) * matrixOrder);

  matrixMulFunc4(cl::EnqueueArgs(queue, cl::NDRange(matrixOrder), cl::NDRange(matrixOrder / 4)),
		 matrixOrder, matrixOrder, d_m1,
		 matrixOrder, matrixOrder, d_m2,
		 d_r4,
		 localColumnBuffer);

  timer.reset();
  queue.finish();

  cl::copy(queue, d_r4, h_r4.begin(), h_r4.end());

  printf("Resultat: %s\r\n", isIdentity(matrixOrder, h_r4) ? "OK" : "ERREUR");
  printf("Kernel '%s' execute en %lu us\r\n", kernelName.c_str(), timer.getTimeMicroseconds());
  printf("\r\n");

  return EXIT_SUCCESS;
}
