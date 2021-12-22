#pragma once
#include <unordered_map>
#include <vector>

struct Node
{
    const std::uint32_t id;
    float rank, rank_now, rank_prev;
    std::vector<const Node *> links_in;  // povezave do strani
    std::vector<const Node *> links_out; // povezave iz strani

    Node();
    Node(const std::uint32_t id);
    void add_link_in(const Node& link);
    void add_link_out(const Node& link);
};

struct Graph
{
    std::unordered_map<std::uint32_t, Node> nodes;
    std::size_t nnodes, nedges;
    std::size_t max_links;
    std::uint32_t max_id;

    Graph(const char *filename);
    bool has_connection(const std::uint32_t a, const std::uint32_t b) const;

    void rank();
    void rank_omp();
};