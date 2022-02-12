#define DELTA (1e-16)

typedef struct 
{
    uint id;
    double rank_new, rank_prev;
    uint nlinks_out;
    uint nlinks_in;
    uint link_in_ids;
}
Node4CL;


__kernel void sortranks(__global Node4CL *nodes, 
                        uint nnodes,
                        __global double *ranks)
{														
    uint gsize = get_global_size(0);

    #pragma unroll
    for (uint gid = get_global_id(0); gid < nnodes; gid += gsize) 
    {
        if (nodes[gid].rank_prev != 0.0) {
            // double diff = nodes[gid].rank - nodes[gid].rank_prev;
            double diff = ranks[gid] - nodes[gid].rank_prev;
            if (diff < 0) {
                diff *= -1;
            }

            if (diff < DELTA) {
                // nodes[gid].rank = nodes[gid].rank_new;
                nodes[gid].rank_prev = 0.0;
            } else {
                // nodes[gid].rank_prev = nodes[gid].rank;
                // nodes[gid].rank = nodes[gid].rank_new;
                nodes[gid].rank_prev = ranks[gid];
                ranks[gid] = nodes[gid].rank_new;
            }
        }
    }
}