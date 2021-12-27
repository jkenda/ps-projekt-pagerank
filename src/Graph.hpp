#pragma once
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#define CHUNK_SIZE 128

#define DELTA 2e-19L
#define D 0.85L

typedef double rank_t;
typedef std::uint32_t id_t;

struct Node
{
    id_t id;                     // id (številka strani)
    rank_t rank, rank_new, rank_prev;     // rangiranje
    uint32_t nlinks_out;            // povezave iz strani
    std::vector<const Node *> links_in;  // povezave do strani

    Node(const id_t id);

    void add_link_in(const Node& link);
    void add_link_out();
};

struct Graph
{
    std::unordered_map<std::uint32_t, Node> nodes; // strani
    std::vector<Node *> nodes_v;                   // kazalci do veljavnih strani
    std::vector<Node *> sink_nodes;                // kazalci do ponornih strani (strani brez izhodnih povezav)
    uint32_t nnodes, nedges, nsinks;          // št. strani, povezav
    id_t max_id;                          // največji id strani

    void read(const char *filename);

    uint32_t rank();
    uint32_t rank_omp();
};