typedef struct 
{
    uint id;
    double rank, rank_new, rank_prev;
    uint nlinks_out;
    uint nlinks_in;
    uint link_in_ids;
}
Node4CL;


__kernel void sinksum(__global Node4CL *nodes,
                      __global const uint sink_offsets,
                      uint nsinks,
                      __global double *p,
                      __local double *partial)
{
    int lid = get_local_id(0);														
    int gid = get_global_id(0);

    partial[lid] = 0.0;
    while(gid < nsinks) 
    {
        partial[lid] += nodes[sink_offsets[gid]].rank;
        gid += get_global_size(0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

	if (lid == 0)
	{
		p[get_group_id(0)] = 0.0;
		for (int i = 0; i < get_local_size(0); i++)
			p[get_group_id(0)] += partial[i];
	}
}