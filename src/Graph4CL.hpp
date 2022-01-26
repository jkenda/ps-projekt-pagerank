#pragma once
#include <cstdint>
#include <vector>
#include <CL/cl.h>
#include "Graph.hpp"

struct Node4CL
{
    id_t id;
    rank_t rank, rank_new, rank_prev; // rangiranje
    uint32_t nlinks_out;              // št. izhodnih povezav
    uint32_t nlinks_in;               // št. vhodnih povezav
    uint32_t links_offset;            // odmik v tabeli links

    Node4CL(id_t id, uint32_t nlinks_in, uint32_t nlinks_out, uint32_t links_offset);
};

struct Graph4CL
{
    Node4CL *nodes;    // strani
    uint32_t *offsets; // pozicije v tabeli (*offsets[id] = <node z id-jem>)

    uint32_t *link_ids;     // povezave
    uint32_t *sink_offsets; // sink-i

    uint32_t nnodes, nedges; // št. strani, povezav
    uint32_t nsinks;         // št. ponorov
    id_t  max_id;            // največji id strani

    std::vector<Node4CL>  nodes_v;
    std::vector<uint32_t> offsets_v;
    std::vector<uint32_t> link_ids_v;
    std::vector<uint32_t> sink_offsets_v;

    Graph4CL(const Graph &graph);
    float data_size();
};

uint32_t Graph4CL_rank(Graph4CL *graph);

/*
definicija struktur v OpenCL:

typedef struct 
{
    id_t id;
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
    id_t max_id;
}
Graph4CL;
*/