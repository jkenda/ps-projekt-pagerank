#pragma once
#include <cstdint>
#include <vector>
#include "Graph.hpp"

struct Node4CL
{
    rank_t rank, rank_new, rank_prev; // rangiranje
    uint32_t nlinks_out;              // št. izhodnih povezav
    uint32_t nlinks_in;               // št. vhodnih povezav
    uint32_t links_offset;            // odmik v tabeli links

    Node4CL();
    Node4CL(uint32_t id, uint32_t nlinks_in, uint32_t nlinks_out, uint32_t links_offset);
};

struct Graph4CL
{
    Node4CL *nodes;   // strani
    int32_t *offsets; // id-ji veljavnih strani

    int32_t *link_ids;     // povezave
    int32_t *sink_offsets; // sink-i

    uint32_t nnodes, nedges; // št. strani, povezav
    uint32_t nsinks;         // št. ponorov
    int32_t  max_id;         // največji id strani

    std::vector<Node4CL> nodes_v;
    std::vector<int32_t> offsets_v;
    std::vector<int32_t> link_ids_v;
    std::vector<int32_t> sink_offsets_v;

    Graph4CL(const Graph& graph);
};

uint32_t Graph4CL_rank(Graph4CL *graph);

/*
definicija struktur v OpenCL:

typedef struct 
{
    uint32_t id;
    rank_t rank, rank_new, rank_prev;
    uint32_t nlinks_out;
    uint32_t nlinks_in;
    uint32_t link_in_ids;
}
Node4CL;

typedef struct 
{
    Node4CL *nodes;
    uint32_t *links;

    uint32_t nnodes, nedges;
    uint32_t max_id;
}
Graph4CL;

*/
