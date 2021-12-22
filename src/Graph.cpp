#include <iostream>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include "Graph.hpp"

using std::size_t; using std::uint32_t; using std::string; using std::max;

Node::Node()
: id(0)
{
}
Node::Node(const uint32_t id)
: id(id)
{
}

void Node::add_link(const Node& link)
{
    links.push_back(&link);
}


Graph::Graph(const char *filename)
{
    std::ifstream is(filename, std::ios::in);
    if (!is.is_open()) {
        exit(1);
    }

    string line, word;

    std::stringstream words;
    std::getline(is, line, '\n');

    // preskoči komentarje in preberi št. povezav
    while (line[0] == '#') {
        words.clear();
        words.str(line);

        while (words >> word) {
            if (word == "Nodes:") {
                words >> nnodes;
            }
            else if (word == "Edges:") {
                words >> nedges;
            }
        }

        std::getline(is, line, '\n');
    }

    nodes.reserve(nnodes);

    uint32_t a, b;

    // dodaj prvo povezavo v množico
    words.clear();
    words.str(line);
    words >> a; words >> b;
    const auto &[it_a, _a] = nodes.try_emplace(a, a);
    const auto &[it_b, _b] = nodes.try_emplace(b, b);
    Node &node_a = it_a->second, &node_b = it_b->second;
    node_b.add_link(node_a);
    std::cout << a << " " << b << std::endl;

    // največji id, največje število povezav do nekega vozlišča
    max_id = max(a, b);
    max_links = node_b.links.size();

    while (is >> a, is >> b) {
        //std::cout << a << " " << b << std::endl;

        // dodaj povezavo v množico
        const auto &[it_a, _a] = nodes.try_emplace(a, a);
        const auto &[it_b, _b] = nodes.try_emplace(b, b);
        Node &node_a = it_a->second, &node_b = it_b->second;
        node_b.add_link(node_a);

        // največji id, največje število povezav do nekega vozlišča
        max_id = max(max_id, max(a, b));
        max_links = max(max_links, node_b.links.size());
    }
}

bool Graph::has_connection(const uint32_t a, const uint32_t b) const
{
    auto node = nodes.find(b);
    if (node == nodes.end()) return false;

    for (const Node *n : node->second.links) {
        if (n->id == a) return true;
    }
    return false;
}