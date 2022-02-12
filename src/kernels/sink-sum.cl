__kernel void sinksum(__global double *ranks,
                      __global const uint *sink_offsets,
                      uint nsinks,
                      __global double *p,
                      __local double *partial)
{
    uint lid = get_local_id(0);														
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);

    if (gid < nsinks) {
        partial[lid] = 0.0;

        #pragma unroll
        for (gid = get_global_id(0); gid < nsinks; gid += gsize) 
        {
            partial[lid] += ranks[sink_offsets[gid]];
            gid += 
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0)
        {
            p[get_group_id(0)] = 0.0;
            for (int i = 0; i < get_local_size(0); i++)
                p[get_group_id(0)] += partial[i];
        }
    }
}