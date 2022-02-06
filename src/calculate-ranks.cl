typedef struct 
{
    uint id;
    double rank, rank_new, rank_prev;
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
                       double sink_sum)
{														
    int gid = get_global_id(0);
    double d = 0.85;
    uint index = 0;
    // if (gid == 0) {
    //     stop[0] = false;
    // }

    while(gid < nnodes) 
    {
        if (nodes[gid].rank_prev != 0.0) {    
            stop[0] = false;
        
            double sum = 0;

            uint from = nodes[gid].link_in_ids;
            uint to = from + nodes[gid].nlinks_in;

            for (uint i = from; i < to; i++) 
            {
                index = offsets[link_ids[i]];
                sum += nodes[index].rank / nodes[index].nlinks_out;
            }

            nodes[gid].rank_new = ((1.0 - d) + d * sink_sum) / nnodes + d * sum;
        }
        
        gid += get_global_size(0);
    }
}