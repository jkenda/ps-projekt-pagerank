#define D (0.85)
#define DELTA (1e-16)

typedef struct 
{
    uint id;
    uint nlinks_out;
    uint nlinks_in;
    uint links_offset;
}
Node4CL;

__kernel void initranks(__global double *ranks, 
                        __global double *ranks_new,
                        uint nnodes)
{														
    uint gsize = get_global_size(0);

    #pragma unroll
    for (uint gid = get_global_id(0); gid < nnodes; gid += gsize) 
    {
        ranks[gid] = (1.0 / nnodes);
        ranks_new[gid] = 1.0;
    }
}

__kernel void calcranks(__global const Node4CL *nodes,
                       __global const uint *links,
                       __global double *ranks,
                       __global double *ranks_new,
                       __global bool *stop,
                       uint nnodes,
                       double sink_sum)
{
    uint index = 0;
    uint gsize = get_global_size(0);

    #pragma unroll
    for (uint gid = get_global_id(0); gid < nnodes; gid += gsize) 
    {
        if (ranks_new[gid] != 0.0) {    
            stop[0] = false;
        
            double sum = 0;

            uint from = nodes[gid].links_offset;
            uint to = from + nodes[gid].nlinks_in;

            for (uint i = from; i < to; i++) 
            {
                sum += ranks[links[i]] / nodes[links[i]].nlinks_out;
            }

            ranks_new[gid] = sink_sum + D * sum;
        }        
    }
}

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

// __kernel void sinksum(__global const uint *sink_offsets,
//                       __global double *ranks,
//                       __global double *p,
//                       uint nsinks,
//                       __local double *partial)
// {
//     uint lid = get_local_id(0);														
//     uint gid = get_global_id(0);
//     uint gsize = get_global_size(0);

//     if (gid < nsinks) {
//         partial[lid] = 0.0;

//         #pragma unroll
//         for (gid = get_global_id(0); gid < nsinks; gid += gsize) 
//         {
//             partial[lid] += ranks[sink_offsets[gid]];
//         }

//         barrier(CLK_LOCAL_MEM_FENCE);

//         if (lid == 0)
//         {
//             p[get_group_id(0)] = 0.0;
//             for (int i = 0; i < get_local_size(0); i++)
//                 p[get_group_id(0)] += partial[i];
//         }
//     }
// }