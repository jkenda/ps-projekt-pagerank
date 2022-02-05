typedef struct 
{
    uint id;
    double rank, rank_new, rank_prev;
    uint nlinks_out;
    uint nlinks_in;
    uint link_in_ids;
}
Node4CL;


__kernel void sortranks(__global Node4CL *nodes, 
                        uint nnodes)
{														
    int gid = get_global_id(0);
    double delta = 1e-12;

    while(gid < nnodes) 
    {
        if (nodes[gid].rank_prev != 0.0) {
            double diff = nodes[gid].rank - nodes[gid].rank_prev;
            if (diff < 0) {
                diff *= -1;
            }

            if (diff < delta) {
                nodes[gid].rank = nodes[gid].rank_new;
                nodes[gid].rank_prev = 0.0;
            } else {
                nodes[gid].rank_prev = nodes[gid].rank;
                nodes[gid].rank = nodes[gid].rank_new;
            }
        }

        gid += get_global_size(0);
    }
}