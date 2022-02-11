#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <CL/cl.h>
#include "Graph4CL.hpp"

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

    ranks = (rank_t *)malloc(nnodes * sizeof(rank_t));

    // GPU
    char ch;
	cl_int ret;
    FILE *fp;
    size_t source_size;

    // READ KERNELS FROM .CL FILES
    char *initsrc;
    fp = fopen("./src/kernels/initialize-ranks.cl", "r");
    // if(!fp)
    // {
    //     fprintf(stderr, ":-(\n");
    //     return 1;
    // }
    initsrc = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(initsrc, 1, MAX_SOURCE_SIZE, fp);
    initsrc[source_size] = '\0';
    fclose(fp);


    char *calcsrc;
    fp = fopen("./src/kernels/calculate-ranks.cl", "r");
    // if(!fp)
    // {
    //     fprintf(stderr, ":-(\n");
    //     return 1;
    // }
    calcsrc = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(calcsrc, 1, MAX_SOURCE_SIZE, fp);
    calcsrc[source_size] = '\0';
    fclose(fp);

    char *sortsrc;
    fp = fopen("./src/kernels/sort-ranks.cl", "r");
    // if(!fp)
    // {
    //     fprintf(stderr, ":-(\n");
    //     return 1;
    // }
    sortsrc = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(sortsrc, 1, MAX_SOURCE_SIZE, fp);
    sortsrc[source_size] = '\0';
    fclose(fp);
    
    // char *sinksumsrc;
    // fp = fopen("./src/kernels/sink-sum.cl", "r");
    // if(!fp)
    // {
    //     fprintf(stderr, ":-(\n");
    //     return 1;
    // }
    // sinksumsrc = (char *)malloc(MAX_SOURCE_SIZE);
    // source_size = fread(sinksumsrc, 1, MAX_SOURCE_SIZE, fp);
    // sinksumsrc[source_size] = '\0';
    // fclose(fp);

    // ...
	cl_platform_id	platform_id[10];
	cl_uint			ret_num_platforms;
	char			*buf;
	size_t			buf_len;
	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);

    cl_device_id	device_id[10];
	cl_uint			ret_num_devices;
    ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);

   	context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);

   	command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);

    // PROGRAM OBJECTS                                
    cl_program program1 = clCreateProgramWithSource(context, 1, (const char **)&initsrc, NULL, &ret);
    cl_program program2 = clCreateProgramWithSource(context, 1, (const char **)&calcsrc, NULL, &ret);
    cl_program program3 = clCreateProgramWithSource(context, 1, (const char **)&sortsrc, NULL, &ret);
    // cl_program program4 = clCreateProgramWithSource(context, 1, (const char **)&sinksumsrc, NULL, &ret);

    ret = clBuildProgram(program1, 1, &device_id[0], NULL, NULL, NULL);
    ret = clBuildProgram(program2, 1, &device_id[0], NULL, NULL, NULL);
    ret = clBuildProgram(program3, 1, &device_id[0], NULL, NULL, NULL);
    // ret = clBuildProgram(program4, 1, &device_id[0], NULL, NULL, NULL);

    // CREATE KERNELS
    initranks_kernel = clCreateKernel(program1, "initranks", &ret);
    calcranks_kernel = clCreateKernel(program2, "calcranks", &ret);
    sortranks_kernel = clCreateKernel(program3, "sortranks", &ret);
    // sinksum_kernel = clCreateKernel(program4, "sinksum", &ret);
}

float Graph4CL::data_size()
{
    return (nodes_v.size() * sizeof(Node4CL)
         + offsets_v.size() * sizeof(decltype(*offsets))
         + link_ids_v.size() * sizeof(decltype(*link_ids))
         + sink_offsets_v.size() * sizeof(decltype(*sink_offsets)))
        / 1'000'000.0F;
}


uint32_t Graph4CL_rank(Graph4CL *graph, const size_t wg_size)
{
	cl_int ret;
    
    // SET WORK SIZES
	size_t num_groups = ((graph->nnodes - 1) / wg_size + 1);
	size_t global_item_size = num_groups * wg_size;

	// size_t sinksum_num_groups = ((graph->nsinks - 1) / local_item_size + 1);
	// size_t sinksum_global_item_size = num_groups * local_item_size;

    // double *p = (double *)malloc(sinksum_num_groups * sizeof(double));
    
    bool *stop = (bool *)malloc(sizeof(bool));
    stop[0] = true;
    
    // BUFFERS
    cl_mem nodes_mem_obj = clCreateBuffer(graph->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                          graph->nnodes * sizeof(Node4CL), graph->nodes, &ret);
    cl_mem ranks_mem_obj = clCreateBuffer(graph->context, CL_MEM_READ_WRITE, 
                                          graph->nnodes * sizeof(rank_t), NULL, &ret);
    cl_mem offsets_mem_obj = clCreateBuffer(graph->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                            (graph->max_id + 1) * sizeof(uint32_t), graph->offsets, &ret);
    cl_mem link_ids_mem_obj = clCreateBuffer(graph->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                             graph->nedges * sizeof(uint32_t), graph->link_ids, &ret);
    cl_mem stop_mem_obj = clCreateBuffer(graph->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                         sizeof(bool), stop, &ret);
    // cl_mem sink_offsets_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
    //                                              graph->nsinks * sizeof(uint32_t), graph->sink_offsets, &ret);
    // cl_mem p_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
    //                                   sinksum_num_groups * sizeof(double), NULL, &ret);

    // SET KERNEL ARGS
    // initranks_kernel
    ret = clSetKernelArg(graph->initranks_kernel, 0, sizeof(cl_mem), (void *)&nodes_mem_obj);
    ret = clSetKernelArg(graph->initranks_kernel, 1, sizeof(cl_uint), (void *)&(graph->nnodes));
    ret = clSetKernelArg(graph->initranks_kernel, 2, sizeof(cl_mem), (void *)&ranks_mem_obj);
    // calcranks_kernel
    ret = clSetKernelArg(graph->calcranks_kernel, 0, sizeof(cl_mem), (void *)&nodes_mem_obj);
    ret = clSetKernelArg(graph->calcranks_kernel, 1, sizeof(cl_mem), (void *)&offsets_mem_obj);
    ret = clSetKernelArg(graph->calcranks_kernel, 2, sizeof(cl_mem), (void *)&link_ids_mem_obj);
    ret = clSetKernelArg(graph->calcranks_kernel, 3, sizeof(cl_mem), (void *)&stop_mem_obj);
    ret = clSetKernelArg(graph->calcranks_kernel, 4, sizeof(cl_uint), (void *)&(graph->nnodes));
    ret = clSetKernelArg(graph->calcranks_kernel, 6, sizeof(cl_mem), (void *)&ranks_mem_obj);
    // sortranks_kernel
    ret = clSetKernelArg(graph->sortranks_kernel, 0, sizeof(cl_mem), (void *)&nodes_mem_obj);
    ret = clSetKernelArg(graph->sortranks_kernel, 1, sizeof(cl_uint), (void *)&(graph->nnodes));
    ret = clSetKernelArg(graph->sortranks_kernel, 2, sizeof(cl_mem), (void *)&ranks_mem_obj);
    // sinksum_kernel
    // ret = clSetKernelArg(sinksum_kernel, 0, sizeof(cl_mem), (void *)&ranks_mem_obj);
    // ret = clSetKernelArg(sinksum_kernel, 1, sizeof(cl_mem), (void *)&sink_offsets_mem_obj);
    // ret = clSetKernelArg(sinksum_kernel, 2, sizeof(cl_uint), (void *)&(graph->nsinks));
    // ret = clSetKernelArg(sinksum_kernel, 3, sizeof(cl_mem), (void *)&p_mem_obj);

    
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
        // https://stackoverflow.com/questions/18056677/opencl-double-precision-different-from-cpu-double-precision/18058130
        // ret = clSetKernelArg(sinksum_kernel, 4, local_item_size * sizeof(double), NULL);
        // ret = clEnqueueNDRangeKernel(graph->command_queue, graph->sinksum_kernel, 1, NULL, 
        //                              &sinksum_global_item_size, &wg_size, 0, NULL, NULL);
        // ret = clEnqueueReadBuffer(graph->command_queue, p_mem_obj, CL_TRUE, 0, sinksum_num_groups * sizeof(double),
        //                           p, 0, NULL, NULL);
        // for (i = 0; i < sinksum_num_groups; i++)
		//     sink_sum += p[i];
        //
        ret = clEnqueueReadBuffer(graph->command_queue, ranks_mem_obj, CL_TRUE, 0, graph->nnodes * sizeof(rank_t), 
                                  graph->ranks, 0, NULL, NULL);
        for (uint32_t i = 0; i < graph->nsinks; i++) {
            sink_sum += graph->ranks[graph->sink_offsets[i]];
        }
        ret = clSetKernelArg(graph->calcranks_kernel, 5, sizeof(cl_double), (void *)&sink_sum);

        // calcranks
        ret = clEnqueueNDRangeKernel(graph->command_queue, graph->calcranks_kernel, 1, NULL, 
                                     &global_item_size, &wg_size, 0, NULL, NULL);

        // stopcheck
        ret = clEnqueueReadBuffer(graph->command_queue, stop_mem_obj, CL_TRUE, 0, sizeof(bool),
                                  stop, 0, NULL, NULL);
        if (stop[0]) { 
            break;
        }
        stop[0] = true;
        ret = clEnqueueWriteBuffer(graph->command_queue, stop_mem_obj, CL_TRUE, 0, sizeof(bool),
                                   stop, 0, NULL, NULL);
        
        // sortranks
        ret = clEnqueueNDRangeKernel(graph->command_queue, graph->sortranks_kernel, 1, NULL, 
                                     &global_item_size, &wg_size, 0, NULL, NULL);
    }

    ret = clEnqueueReadBuffer(graph->command_queue, ranks_mem_obj, CL_TRUE, 0, graph->nnodes * sizeof(rank_t),
                              graph->ranks, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(graph->command_queue, nodes_mem_obj, CL_TRUE, 0, graph->nnodes * sizeof(Node4CL),
                              graph->nodes, 0, NULL, NULL);

    return iterations;
}