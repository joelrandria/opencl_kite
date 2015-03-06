#define __CL_ENABLE_EXCEPTIONS

#include <cl.hpp>
#include <util.hpp>

#include <vector>
#include <string>

/* ========== OpenCL ========== */

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

  printf("Device name: %s\r\n", deviceName.c_str());
  printf("Device type: ");
  switch (deviceType)
  {
  case CL_DEVICE_TYPE_CPU: printf("CPU\r\n"); break;
  case CL_DEVICE_TYPE_GPU: printf("GPU\r\n"); break;
  case CL_DEVICE_TYPE_ACCELERATOR: printf("ACCELERATOR\r\n"); break;
  case CL_DEVICE_TYPE_DEFAULT: printf("DEFAULT\r\n"); break;
  default: printf("N/A\r\n"); break;
  }
  printf("Device version: %s\r\n", deviceVersion.c_str());

  printf("Device max compute units:%d\r\n", deviceMaxComputeUnits);
  printf("Device max workitem dimensions:%d\r\n", deviceMaxWorkItemDimensions);
  printf("Device max workitem sizes: [0]=%lu, [1]=%lu, [2]=%lu\r\n", deviceMaxWorkItemSizes[0], deviceMaxWorkItemSizes[1], deviceMaxWorkItemSizes[2]);
  printf("Device max workgroup size:%lu\r\n", deviceMaxWorkGroupSize);

  printf("Device local memory size:%lu octets\r\n", deviceLocalMemSize);
  printf("Device global memory size:%lu octets\r\n", deviceGlobalMemSize);
}
void printDevicesInfo(const std::vector<cl::Device>& devices)
{
  unsigned int i;
  unsigned int count;

  count = devices.size();

  for (i = 0; i < count; ++i)
  {
    printf("--------------- Context device [%d] ---------------\r\n", i);
    printf("\r\n");
    printDeviceInfo(devices[i]);
  }
}

void getContextDevices(const cl::Context& context, std::vector<cl::Device>& devices)
{
  devices.clear();

  if (context.getInfo<VECTOR_CLASS<cl::Device> >(CL_CONTEXT_DEVICES, &devices) != CL_SUCCESS)
  {
    fprintf(stderr, "getContextDevices(): Impossible d'accéder aux devices du contexte spécifié\r\n");
    exit(EXIT_FAILURE);
  }
}

void printKernelInfo(const cl::Kernel& kernel, const cl::Device& device)
{
  //size_t kernelGlobalWorkSizes[3];
  size_t kernelWorkGroupSize;
  size_t kernelPreferredWorkGroupSizeMultiple;
  unsigned long kernelPrivateMemSize;
  unsigned long kernelLocalMemSize;

  //kernel.getWorkGroupInfo(device, CL_KERNEL_GLOBAL_WORK_SIZE, kernelGlobalWorkSizes);
  kernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &kernelWorkGroupSize);
  kernel.getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &kernelPreferredWorkGroupSizeMultiple);
  kernel.getWorkGroupInfo(device, CL_KERNEL_PRIVATE_MEM_SIZE, &kernelPrivateMemSize);
  kernel.getWorkGroupInfo(device, CL_KERNEL_LOCAL_MEM_SIZE, &kernelLocalMemSize);

  //printf("Kernel global work size: [0] = %d, [1] = %d, [2] = %d\r\n", kernelGlobalWorkSizes[0], kernelGlobalWorkSizes[1], kernelGlobalWorkSizes[2]);
  printf("Kernel work group size: %lu\r\n", kernelWorkGroupSize);
  printf("Kernel preferred work group size multiple: %lu\r\n", kernelPreferredWorkGroupSizeMultiple);
  printf("Kernel private memory size: %lu bytes\r\n", kernelPrivateMemSize);
}

/* ========== Application ========== */

#define OPENCL_DEVICE_ID	0
#define PROGRAM_FILENAME	"pi.cl"
#define INTEGRAL_SUBDIV_COUNT	1024

void computePiWithOneWIPerIteration(const cl::Context& context,
				    const cl::Program& program,
				    const cl::Device& device,
				    cl::CommandQueue& queue)
{
  cl::Kernel kernel;

  std::vector<float> h_groupAreas;
  cl::Buffer d_groupAreas;

  int i;
  float pi;

  util::Timer timer;

  const int workGroupSize = 64;
  const int workGroupCount = INTEGRAL_SUBDIV_COUNT / workGroupSize;

  printf("\r\n");
  printf("--------------- Kernel #1: 1 work item p/ iteration ---------------\r\n");

  kernel = cl::Kernel(program, "pi_1wi_1iteration");

  printf("\r\n");
  printKernelInfo(kernel, device);

  h_groupAreas.resize(workGroupCount);
  d_groupAreas = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * workGroupCount);

  cl::make_kernel<cl::LocalSpaceArg, cl::Buffer> kernelFunc(kernel);
  kernelFunc(cl::EnqueueArgs(queue, cl::NDRange(INTEGRAL_SUBDIV_COUNT)),
	     cl::Local(sizeof(float) * workGroupSize),
	     d_groupAreas);

  timer.reset();

  queue.finish();
  cl::copy(queue, d_groupAreas, h_groupAreas.begin(), h_groupAreas.end());

  pi = 0;
  for (i = 0; i < workGroupCount; ++i)
    pi += h_groupAreas[i];

  printf("\r\n");
  printf("Résultat: %f\r\n", pi);
  printf("Execute en %lu us\r\n", timer.getTimeMicroseconds());
}

int main(int, char**)
{
  cl::Context context;

  std::vector<cl::Device> devices;
  cl::Device targetDevice;

  cl::CommandQueue queue;
  cl::Program program;

  util::Timer timer;

  // Contexte
  printf("Initialisation du contexte OpenCL... ");

  context = cl::Context(CL_DEVICE_TYPE_ALL);

  printf("OK\r\n");
  printf("\r\n");

  getContextDevices(context, devices);

  printDevicesInfo(devices);
  printf("\r\n");

  targetDevice = devices[OPENCL_DEVICE_ID];

  // File de commandes
  printf("Initialisation d'une file de commandes pour le device [%d]... ", OPENCL_DEVICE_ID);

  queue = cl::CommandQueue(context, targetDevice);

  printf("OK\r\n");

  // Programme
  try
  {
    printf("Chargement du programme '%s'... ", PROGRAM_FILENAME);

    program = cl::Program(context, util::loadProgram(PROGRAM_FILENAME));
    program.build();

    printf("OK\r\n");
  }
  catch (cl::Error e)
  {
    std::string log;

    program.getBuildInfo(targetDevice, CL_PROGRAM_BUILD_LOG, &log);

    fprintf(stderr, "Exception: %s\r\n", e.what());
    fprintf(stderr, "\r\n%s\r\n", log.c_str());

    return EXIT_FAILURE;
  }

  // Execution des kernels
  computePiWithOneWIPerIteration(context, program, targetDevice, queue);

  return EXIT_SUCCESS;
}
