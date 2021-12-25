#pragma once
#include <cstdint>
#include <vector>
#include "Graph.hpp"

struct Node4CL
{
    std::int32_t id;                 // id (številka) strani
    float rank, rank_new, rank_prev; // rangiranje
    std::uint32_t nlinks_out;        // št. izhodnih povezav
    std::uint32_t nlinks_in;         // št. vhodnih povezav

    Node4CL();
    Node4CL(std::uint32_t id, std::uint32_t links_in, std::size_t nlinks_in, std::size_t nlinks_out);
};

struct Graph4CL
{
    Node4CL *nodes;         // strani
    std::int32_t *ids;      // id-ji veljavnih strani
    std::int32_t *links;    // povezave
    std::int32_t *sinks;    // sink-i
    std::int32_t *offsets;  // offset-i

    std::uint32_t nnodes, nedges; // št. strani, povezav
    std::int32_t max_id;          // največji id strani
    std::int32_t nsinks;          // št. sink-ov

    std::vector<Node4CL> nodes_v;
    std::vector<std::int32_t> ids_v;
    std::vector<std::int32_t> links_v;
    std::vector<std::int32_t> sinks_v;
    std::vector<std::int32_t> offsets_v;

    Graph4CL(const Graph& graph);
};

void Graph4CL_rank(Graph4CL *graph);

/*
definicija struktur v OpenCL:

typedef struct 
{
    uint32_t id;
    float rank, rank_new, rank_prev;
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
