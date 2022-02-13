#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <CL/cl.h>
#include "Graph4CL.hpp"

#define MAX_SOURCE_SIZE (16384)

using namespace std;

Node4CL::Node4CL(uint32_t nlinks_in, uint32_t nlinks_out, uint32_t links_offset)
: nlinks_in(nlinks_in), nlinks_out(nlinks_out), links_offset(links_offset)
{
}

Graph4CL::Graph4CL(const Graph &graph)
: nnodes(graph.nnodes), nedges(graph.nedges), nsinks(graph.nsinks)
{
    nodes_v.reserve(nnodes);
    links_v.reserve(nedges);
    sink_offsets_v.reserve(nsinks);
    ids.reserve(nnodes);

    for (const Node &node : graph.nodes) {
        uint32_t nodes_offset = nodes_v.size();
        uint32_t links_offset = links_v.size();
        uint32_t nlinks_in = node.links_in.size();

        nodes_v.emplace_back(nlinks_in, node.nlinks_out, links_offset);
        ids.emplace_back(node.id);
        
        if (node.nlinks_out == 0) {
            sink_offsets_v.emplace_back(nodes_offset);
        }

        for (const Node *src : node.links_in) {
            links_v.emplace_back(src - graph.nodes.data());
        }

    }

    nodes = nodes_v.data();
    links = links_v.data();
    sink_offsets = sink_offsets_v.data();

    ranks = (rank_t *)malloc(nnodes * sizeof(rank_t));

    // GPU
    char ch;
	cl_int ret;
    FILE *fp;
    size_t source_size;

    // READ KERNELS FROM .CL FILES
    char *godsrc;
    fp = fopen("./src/god-kernel.cl", "r");
    godsrc = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(godsrc, 1, MAX_SOURCE_SIZE, fp);
    godsrc[source_size] = '\0';
    fclose(fp);

    // ...
	cl_platform_id	platform_id[10];
	cl_uint			ret_num_platforms;
	char			*buf;
	size_t			buf_len;
	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);

    cl_device_id	device_id[10];
	cl_uint			ret_num_devices;
    ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);

   	context = clCreateContext(NULL, ret_num_devices, device_id, NULL, NULL, &ret);

   	command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);

    // PROGRAM OBJECT                              
    program = clCreateProgramWithSource(context, 1, (const char **)&godsrc, NULL, &ret);

    // BUILD PROGRAM
    ret = clBuildProgram(program, ret_num_devices, device_id, NULL, NULL, NULL);

    // CREATE KERNELS
    initranks_kernel = clCreateKernel(program, "initranks", &ret);
    calcranks_kernel = clCreateKernel(program, "calcranks", &ret);
    sortranks_kernel = clCreateKernel(program, "sortranks", &ret);

    free(godsrc);
}

Graph4CL::~Graph4CL() 
{
    clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(initranks_kernel);
	clReleaseKernel(calcranks_kernel);
	clReleaseKernel(sortranks_kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

float Graph4CL::data_size()
{
    return (nodes_v.size() * sizeof(Node4CL)
         + ids.size() * sizeof(id_t)
         + links_v.size() * sizeof(decltype(*links))
         + sink_offsets_v.size() * sizeof(decltype(*sink_offsets)))
        / 1'000'000.0F;
}

uint32_t Graph4CL_rank(Graph4CL *graph, const size_t wg_size)
{
	cl_int ret;
    
    // SET WORK SIZES
	size_t num_groups = ((graph->nnodes - 1) / wg_size + 1);
	size_t global_item_size = num_groups * wg_size;
    
    bool *stop = (bool *)malloc(sizeof(bool));
    *stop = true;
    
    // BUFFERS
    graph->nodes_mem_obj = clCreateBuffer(graph->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                          graph->nnodes * sizeof(Node4CL), graph->nodes, &ret);
    graph->ranks_mem_obj = clCreateBuffer(graph->context, CL_MEM_READ_WRITE, 
                                          graph->nnodes * sizeof(rank_t), NULL, &ret);
    graph->ranks_new_mem_obj = clCreateBuffer(graph->context, CL_MEM_READ_WRITE, 
                                              graph->nnodes * sizeof(rank_t), NULL, &ret);
    graph->links_mem_obj = clCreateBuffer(graph->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                             graph->nedges * sizeof(uint32_t), graph->links, &ret);
    graph->stop_mem_obj = clCreateBuffer(graph->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                         sizeof(bool), stop, &ret);

    // SET KERNEL ARGS
    // initranks_kernel
    ret = clSetKernelArg(graph->initranks_kernel, 0, sizeof(cl_mem), (void *)&graph->ranks_mem_obj);
    ret = clSetKernelArg(graph->initranks_kernel, 1, sizeof(cl_mem), (void *)&graph->ranks_new_mem_obj);
    ret = clSetKernelArg(graph->initranks_kernel, 2, sizeof(cl_uint), (void *)&graph->nnodes);
    // calcranks_kernel
    ret = clSetKernelArg(graph->calcranks_kernel, 0, sizeof(cl_mem), (void *)&graph->nodes_mem_obj);
    ret = clSetKernelArg(graph->calcranks_kernel, 1, sizeof(cl_mem), (void *)&graph->links_mem_obj);
    ret = clSetKernelArg(graph->calcranks_kernel, 2, sizeof(cl_mem), (void *)&graph->ranks_mem_obj);
    ret = clSetKernelArg(graph->calcranks_kernel, 3, sizeof(cl_mem), (void *)&graph->ranks_new_mem_obj);
    ret = clSetKernelArg(graph->calcranks_kernel, 4, sizeof(cl_mem), (void *)&graph->stop_mem_obj);
    ret = clSetKernelArg(graph->calcranks_kernel, 5, sizeof(cl_uint), (void *)&graph->nnodes);
    // sortranks_kernel
    ret = clSetKernelArg(graph->sortranks_kernel, 0, sizeof(cl_mem), (void *)&graph->ranks_mem_obj);
    ret = clSetKernelArg(graph->sortranks_kernel, 1, sizeof(cl_mem), (void *)&graph->ranks_new_mem_obj);
    ret = clSetKernelArg(graph->sortranks_kernel, 2, sizeof(cl_uint), (void *)&graph->nnodes);
    // sinksum_kernel

    
    // ALGORITEM
    rank_t sink_sum = 0;
    uint32_t iterations = 0;

    // initranks
    ret = clEnqueueNDRangeKernel(graph->command_queue, graph->initranks_kernel, 1, NULL, 
                                 &global_item_size, &wg_size, 0, NULL, NULL);
    while(true) {
        sink_sum = 0;
        iterations++;

        // sinksum
        ret = clEnqueueReadBuffer(graph->command_queue, graph->ranks_mem_obj, CL_TRUE, 0, graph->nnodes * sizeof(rank_t), 
                                  graph->ranks, 0, NULL, NULL);
        for (uint32_t i = 0; i < graph->nsinks; i++) {
            sink_sum += graph->ranks[graph->sink_offsets[i]];
        }
        sink_sum =  ((1.0 - D) + D * sink_sum) / graph->nnodes;
        ret = clSetKernelArg(graph->calcranks_kernel, 6, sizeof(cl_double), (void *)&sink_sum);

        // calcranks
        ret = clEnqueueNDRangeKernel(graph->command_queue, graph->calcranks_kernel, 1, NULL, 
                                     &global_item_size, &wg_size, 0, NULL, NULL);

        // stopcheck
        ret = clEnqueueReadBuffer(graph->command_queue, graph->stop_mem_obj, CL_TRUE, 0, sizeof(bool),
                                  stop, 0, NULL, NULL);
        if (*stop) break;
        *stop = true;
        ret = clEnqueueWriteBuffer(graph->command_queue, graph->stop_mem_obj, CL_TRUE, 0, sizeof(bool),
                                   stop, 0, NULL, NULL);
        
        // sortranks
        ret = clEnqueueNDRangeKernel(graph->command_queue, graph->sortranks_kernel, 1, NULL, 
                                     &global_item_size, &wg_size, 0, NULL, NULL);
    }

    ret = clEnqueueReadBuffer(graph->command_queue, graph->ranks_mem_obj, CL_TRUE, 0, graph->nnodes * sizeof(rank_t),
                              graph->ranks, 0, NULL, NULL);
    
    free(stop);

    clFlush(graph->command_queue);
	clFinish(graph->command_queue);

    clReleaseMemObject(graph->nodes_mem_obj);
    clReleaseMemObject(graph->ranks_mem_obj);
    clReleaseMemObject(graph->ranks_new_mem_obj);
    clReleaseMemObject(graph->links_mem_obj);
    clReleaseMemObject(graph->stop_mem_obj);

    return iterations;
}