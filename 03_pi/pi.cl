__kernel void pi_1wi_1iteration(__local float* l_itemAreas, __global float* g_groupAreas)
{
  int i;

  int lid;
  int lsize;

  float subdiv;
  float x;
  float sum;

  lid = get_local_id(0);
  lsize = get_local_size(0);

  subdiv = 1.0f / get_global_size(0);
  x =  (0.5f + get_global_id(0)) * subdiv;

  l_itemAreas[lid] = 4.0f / (1.0f + (x * x));

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0)
  {
    sum = 0;
    for (i = 0; i < lsize; ++i)
      sum += l_itemAreas[i];

    g_groupAreas[get_group_id(0)] = sum * subdiv;
  }
}
