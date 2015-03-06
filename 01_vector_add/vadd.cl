__kernel void vadd(__global const float *a,
				   __global const float *b,
				   __global const float *c,
				   __global       float *d)
{
    int gid = get_global_id(0);
    d[gid]  = a[gid] + b[gid] + c[gid];
}