#pragma once
#include <cstdint>
#include <vector>
#include <CL/cl.h>
#include "Graph.hpp"

struct Node4CL
{
    id_t id;
    rank_t rank_new, rank_prev; // rangiranje
    uint32_t nlinks_out;        // št. izhodnih povezav
    uint32_t nlinks_in;         // št. vhodnih povezav
    uint32_t links_offset;      // odmik v tabeli links

    Node4CL(id_t id, uint32_t nlinks_in, uint32_t nlinks_out, uint32_t links_offset);
};

struct Graph4CL
{
    Node4CL *nodes;         // strani
    uint32_t *links;        // povezave
    uint32_t *sink_offsets; // ponori
    
    rank_t *ranks; // rangi

    uint32_t nnodes, nedges; // št. strani, povezav
    uint32_t nsinks;         // št. ponorov
    id_t  max_id;            // največji id strani

    std::vector<Node4CL>  nodes_v;
    std::vector<uint32_t> links_v;
    std::vector<uint32_t> sink_offsets_v;
    
    cl_context context;
    cl_command_queue command_queue;
    cl_kernel initranks_kernel;
    cl_kernel calcranks_kernel;
    cl_kernel sortranks_kernel;
    // cl_kernel sinksum_kernel;

    Graph4CL(const Graph &graph);
    float data_size();

};

uint32_t Graph4CL_rank(Graph4CL *graph, const size_t wg_size);
