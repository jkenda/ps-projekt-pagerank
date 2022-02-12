#define DELTA (1e-16)

__kernel void sortranks(__global double *ranks,
                        __global double *ranks_new,
                        uint nnodes)
{														
    uint gsize = get_global_size(0);

    #pragma unroll
    for (uint gid = get_global_id(0); gid < nnodes; gid += gsize) 
    {
        if (ranks_new[gid] != 0.0) {
            double diff = ranks[gid] - ranks_new[gid];
            if (diff < 0) {
                diff *= -1;
            }

            if (diff < DELTA) {
                ranks[gid] = ranks_new[gid];
                ranks_new[gid] = 0.0;
            } else {
                ranks[gid] = ranks_new[gid];
            }
        }
    }
}