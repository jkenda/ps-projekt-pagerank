#pragma once
#include <unordered_map>
#include <vector>

struct Node
{
    const std::uint32_t id;
    std::vector<const Node *> links; // povezave do vozlišča
    Node();
    Node(const std::uint32_t id);
    void add_link(const Node& link);
};

struct Graph
{
    std::unordered_map<std::uint32_t, Node> nodes;
    std::size_t nnodes, nedges;
    std::size_t max_links;
    std::uint32_t max_id;

    Graph(const char *filename);
    bool has_connection(const std::uint32_t a, const std::uint32_t b) const;
};