#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

struct Node
{
    std::int32_t id;                     // id (številka strani)
    float rank, rank_new, rank_prev;     // rangiranje
    std::vector<const Node *> links_in;  // povezave do strani
    std::uint32_t nlinks_out;            // povezave iz strani

    Node();
    Node(const std::uint32_t id);

    void add_link_in(const Node& link);
    void add_link_out();
};

struct Graph
{
    std::unordered_map<std::uint32_t, Node> nodes; // strani
    std::vector<Node *> nodes_v;                   // kazalci do veljavnih strani
    std::uint32_t nnodes, nedges;                  // št. strani, povezav
    std::uint32_t max_id;                          // največji id strani

    Graph(const char *filename);
    bool has_connection(const std::uint32_t a, const std::uint32_t b) const;

    void rank();
    void rank_omp();
};