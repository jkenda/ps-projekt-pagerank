#define D 0.85

typedef struct 
{
    uint id;
    double rank_new, rank_prev;
    uint nlinks_out;
    uint nlinks_in;
    uint link_in_ids;
}
Node4CL;


__kernel void calcranks(__global Node4CL *nodes,
                       __global const uint *offsets,
                       __global const uint *link_ids,
                       __global bool *stop,
                       uint nnodes,
                       double sink_sum,
                       __global double *ranks)
{
    uint index = 0;
    uint gsize = get_global_size(0);

    #pragma unroll
    for (uint gid = get_global_id(0); gid < nnodes; gid += gsize) 
    {
        if (nodes[gid].rank_prev != 0.0) {    
            stop[0] = false;
        
            double sum = 0;

            uint from = nodes[gid].link_in_ids;
            uint to = from + nodes[gid].nlinks_in;

            for (uint i = from; i < to; i++) 
            {
                index = offsets[link_ids[i]];
                sum += ranks[index] / nodes[index].nlinks_out;
            }

            nodes[gid].rank_new = sink_sum + D * sum;
        }        
    }
}