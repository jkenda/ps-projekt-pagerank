#pragma once
#include <cstdint>
#include <vector>
#include <CL/cl.h>
#include "Graph.hpp"

struct Node4CL
{
    uint32_t nlinks_out;        // št. izhodnih povezav
    uint32_t nlinks_in;         // št. vhodnih povezav
    uint32_t links_offset;      // odmik v tabeli links

    Node4CL(uint32_t nlinks_in, uint32_t nlinks_out, uint32_t links_offset);
};

struct Graph4CL
{
    Node4CL *nodes;         // strani
    uint32_t *links;        // povezave
    uint32_t *sink_offsets; // ponori
    
    rank_t *ranks;          // rangi

    uint32_t nnodes, nedges; // št. strani, povezav
    uint32_t nsinks;         // št. ponorov

    std::vector<Node4CL>  nodes_v;
    std::vector<uint32_t> links_v;
    std::vector<uint32_t> sink_offsets_v;
    std::vector<uint32_t> ids;
    
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel initranks_kernel;
    cl_kernel calcranks_kernel;
    cl_kernel sortranks_kernel;
    // cl_kernel sinksum_kernel;
    cl_mem nodes_mem_obj;
    cl_mem ranks_mem_obj;
    cl_mem ranks_new_mem_obj;
    cl_mem links_mem_obj;
    cl_mem stop_mem_obj;

    Graph4CL(const Graph &graph);
    float data_size();
};

void cleanup(Graph4CL *graph);
uint32_t Graph4CL_rank(Graph4CL *graph, const size_t wg_size);
