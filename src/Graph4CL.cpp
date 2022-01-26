#include <cmath>
#include <cstdlib>
#include <CL/cl.h>
#include <iostream>
#include "Graph4CL.hpp"

#define WORKGROUP_SIZE	(256)
#define MAX_SOURCE_SIZE (16384)

using namespace std;

Node4CL::Node4CL(id_t id, uint32_t nlinks_in, uint32_t nlinks_out, uint32_t links_offset)
: id(id), nlinks_in(nlinks_in), nlinks_out(nlinks_out), links_offset(links_offset)
{
}

Graph4CL::Graph4CL(const Graph &graph)
: nnodes(graph.nnodes), nedges(graph.nedges), max_id(graph.max_id), nsinks(graph.nsinks)
{
    offsets_v.reserve(max_id + 1);

    nodes_v.reserve(nnodes);
    link_ids_v.reserve(nedges);
    sink_offsets_v.reserve(nsinks);

    for (const Node &node : graph.nodes) {
        uint32_t nodes_offset = nodes_v.size();
        uint32_t links_offset = link_ids_v.size();
        uint32_t nlinks_in = node.links_in.size();

        offsets_v[node.id] = nodes_offset;
        nodes_v.emplace_back(node.id, nlinks_in, node.nlinks_out, links_offset);
        
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

float Graph4CL::data_size()
{
    return (nodes_v.size() * sizeof(Node4CL)
         + offsets_v.size() * sizeof(decltype(*offsets))
         + link_ids_v.size() * sizeof(decltype(*link_ids))
         + sink_offsets_v.size() * sizeof(decltype(*sink_offsets)))
        / 1'000'000.0F;
}


uint32_t Graph4CL_rank(Graph4CL *graph)
{
    char ch;
	int i;
	cl_int ret;
    FILE *fp;
    char *source_str;
    size_t source_size;

    // STOP ARRAY
    bool *stop_arr = (bool *)malloc(graph->nnodes * sizeof(bool));

    fp = fopen("./src/pagerank.cl", "r");
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

    size_t local_item_size = WORKGROUP_SIZE;
	size_t num_groups = ((graph->nnodes - 1) / local_item_size + 1);
	size_t global_item_size = num_groups * local_item_size;

    for (uint32_t i = 0; i < graph->nnodes; i++) {
        graph->nodes[i].rank = 1.0 / graph->nnodes;
        graph->nodes[i].rank_prev = 1.0;
    }

    cl_mem nodes_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                          graph->nnodes * sizeof(Node4CL), graph->nodes, &ret);
    cl_mem offsets_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                            (graph->max_id + 1) * sizeof(uint32_t), graph->offsets, &ret);
    cl_mem link_ids_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                             graph->nedges * sizeof(uint32_t), graph->link_ids, &ret);
    cl_mem stop_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, 
                                         graph->nnodes * sizeof(bool), stop_arr, &ret);
                                    
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &ret);

    ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "pagerank", &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&nodes_mem_obj);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&offsets_mem_obj);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&link_ids_mem_obj);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&stop_mem_obj);
    ret |= clSetKernelArg(kernel, 4, sizeof(cl_uint), (void *)&(graph->nnodes));

    // PAGERANK
    bool stop = false;
    rank_t sink_sum = 0;
    uint32_t iterations = 0;

    while(true) {
        stop = true;
        sink_sum = 0;
        iterations++;

        for (uint32_t i = 0; i < graph->nsinks; i++) {
            sink_sum += graph->nodes[graph->sink_offsets[i]].rank;
        }
        ret |= clSetKernelArg(kernel, 5, sizeof(cl_double), (void *)&(sink_sum));

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, nodes_mem_obj, CL_TRUE, 0, graph->nnodes * sizeof(Node4CL), 
                                  graph->nodes, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, stop_mem_obj, CL_TRUE, 0, graph->nnodes * sizeof(bool), 
                                  stop_arr, 0, NULL, NULL);

        for (uint32_t i = 0; i < graph->nnodes; i++) {
            if (!stop_arr[i]) {
                stop = false;
                break;
            }
        }
        if (stop || iterations == 67) break;

        for (uint32_t i = 0; i < graph->nnodes; i++) {
            if (graph->nodes[i].rank_prev == 0.0) continue;

            if (abs(graph->nodes[i].rank - graph->nodes[i].rank_prev) < DELTA) {
                graph->nodes[i].rank = graph->nodes[i].rank_new;
                graph->nodes[i].rank_prev = 0.0;
            }
            else {
                graph->nodes[i].rank_prev = graph->nodes[i].rank;
                graph->nodes[i].rank = graph->nodes[i].rank_new;
            }
        }

        ret = clEnqueueWriteBuffer(command_queue, nodes_mem_obj, CL_TRUE, 0, graph->nnodes * sizeof(Node4CL), 
                                   graph->nodes, 0, NULL, NULL);
    }

    ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(nodes_mem_obj);
	ret = clReleaseMemObject(offsets_mem_obj);
	ret = clReleaseMemObject(link_ids_mem_obj);
	ret = clReleaseMemObject(stop_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

    free(stop_arr);

    return iterations;
}