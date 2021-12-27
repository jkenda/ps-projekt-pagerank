#include "Graph4CL.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <CL/cl.h>

#define DELTA (2e-19L)
#define D (0.85L)

#define WORKGROUP_SIZE	(256)
#define MAX_SOURCE_SIZE (16384)

using namespace std;

Node4CL::Node4CL(id_t id, uint32_t nlinks_in, uint32_t nlinks_out, uint32_t links_offset)
: id(id), nlinks_in(nlinks_in), nlinks_out(nlinks_out), links_offset(links_offset)
{
}

Graph4CL::Graph4CL(const Graph& graph)
: nnodes(graph.nnodes), nedges(graph.nedges), max_id(graph.max_id), nsinks(graph.nsinks)
{
    offsets_v.reserve(max_id + 1);

    nodes_v.reserve(nnodes);
    link_ids_v.reserve(nedges);
    sink_offsets_v.reserve(nsinks);

    for (const auto &[id, node] : graph.nodes) {
        uint32_t nodes_offset = nodes_v.size();
        uint32_t links_offset = link_ids_v.size();
        uint32_t nlinks_in = node.links_in.size();

        offsets_v[id] = nodes_offset;
        
        nodes_v.emplace_back(id, nlinks_in, node.nlinks_out, links_offset);
        
        if (node.nlinks_out == 0) {
            sink_offsets_v.emplace_back(nodes_offset);
        }

        for (const Node *src : node.links_in) {
            link_ids_v.emplace_back(src->id);
        }

    }

    nodes    = nodes_v.data();
    offsets  = offsets_v.data(); 
    link_ids = link_ids_v.data();
    sink_offsets = sink_offsets_v.data();
}


uint32_t Graph4CL_rank(Graph4CL *graph)
{
    // #pragma omp parallel for
    for (uint32_t i = 0; i < graph->nnodes; i++) {
        Node4CL *node = &graph->nodes[i];

        node->rank = 1.0f / graph->nnodes;
        node->rank_prev = 0;
    }

    bool stop = false;
    rank_t sink_sum = 0;
    uint32_t iterations = 0;

    while (!stop) {
        stop = true;
        sink_sum = 0;
        iterations++;

        for (uint32_t i = 0; i < graph->nsinks; i++) {
            uint32_t offset = graph->sink_offsets[i];
            Node4CL *sink   = &graph->nodes[offset];

            sink_sum += sink->rank;
        }

        // #pragma omp parallel for
        for (uint32_t i = 0; i < graph->nnodes; i++) {
            Node4CL *node = &graph->nodes[i];

            if (abs(node->rank - node->rank_prev) < DELTA) continue;

            // #pragma omp atomic write
            stop = false;

            rank_t sum = 0;

            for (uint32_t i = 0; i < node->nlinks_in; i++) {
                uint32_t link_node = graph->link_ids[node->links_offset + i];
                uint32_t offset = graph->offsets[link_node];
                Node4CL *src = &graph->nodes[offset];

                sum += src->rank / src->nlinks_out;
            }

            sum *= D;
            node->rank_new = ((1 - D) + D * sink_sum) / graph->nnodes + sum;
        }

        // #pragma omp parallel for
        for (uint32_t i = 0; i < graph->nnodes; i++) {
            Node4CL *node = &graph->nodes[i];

            node->rank_prev = node->rank;
            node->rank      = node->rank_new;
        }
    }

    return iterations;
}

void Graph4CL_rank_GPU(Graph4CL *graph) {
    // copy paste template iz vaj
    // veckrat zažen kernel z različnimi podatki
    // ni treba vseh podatkov vedno prenasat?
    // array z stop booleani?

    uint32_t nsinks = graph->nsinks;
    uint32_t nnodes = graph->nnodes;

    char ch;
	int i;
	cl_int ret;
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("pagerank.cl", "r");
    if(!fp)
    {
        fprintf(stderr, ":-(\n");
        return 1;
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

    cl_platform_id	platform_id[10];
	cl_uint			ret_num_platforms;
	char			*buf;
	size_t			buf_len;
	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);

    cl_device_id	device_id[10];
	cl_uint			ret_num_devices;
	ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);

    cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
    
    // delitev dela
	size_t local_item_size = WORKGROUP_SIZE;
	size_t num_groups = ((graph->nnodes - 1) / local_item_size + 1);
	size_t global_item_size = num_groups * local_item_size;

    // stop array
    bool stop[num_groups] = {false};

    // TODO: create buffers
    
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &ret);

    ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

    // TODO: set arguments

    // TODO 
    while(!stop) {
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    }
}