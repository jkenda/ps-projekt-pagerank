__kernel void initranks(__global double *ranks, 
                        __global double *ranks_new,
                        uint nnodes)
{														
    uint gsize = get_global_size(0);

    #pragma unroll
    for (uint gid = get_global_id(0); gid < nnodes; gid += gsize) 
    {
        // nodes[gid].rank = (1.0 / nnodes);
        ranks[gid] = (1.0 / nnodes);
        ranks_new[gid] = 1.0;
    }
}