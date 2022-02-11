typedef struct 
{
    uint id;
    double rank, rank_new, rank_prev;
    uint nlinks_out;
    uint nlinks_in;
    uint link_in_ids;
}
Node4CL;


__kernel void initranks(__global Node4CL *nodes, 
                        uint nnodes,
                        __global double *ranks)
{														
    int gid = get_global_id(0);

    while(gid < nnodes) 
    {
        // nodes[gid].rank = (1.0 / nnodes);
        ranks[gid] = (1.0 / nnodes);
        nodes[gid].rank_prev = 1.0;

        gid += get_global_size(0);
    }
}