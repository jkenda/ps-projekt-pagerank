#include <iostream>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include "Graph.hpp"

const float delta = 1e-24;

using std::size_t; using std::uint32_t; using std::string; using std::max;

Node::Node()
: id(0), rank(0)
{
}
Node::Node(const uint32_t id)
: id(id), rank(0)
{
}

void Node::add_link_in(const Node& link)
{
    links_in.push_back(&link);
}

void Node::add_link_out(const Node& link)
{
    links_out.push_back(&link);
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
    words.clear();
    words.str(line);

    nodes.reserve(nnodes);

    uint32_t a, b;
    words >> a; words >> b;

    // dodaj prvi vozlišči
    const auto &[it_a, _a] = nodes.try_emplace(a, a);
    const auto &[it_b, _b] = nodes.try_emplace(b, b);
    Node &node_a = it_a->second, &node_b = it_b->second;

    // dodaj prvo povezavo v množico
    node_b.add_link_in(node_a);
    node_a.add_link_out(node_b);

    // največji id, največje število povezav do nekega vozlišča
    max_id = max(a, b);
    max_links = node_b.links_in.size();

    while (is >> a, is >> b) {
        // dodaj vozlišči, če še ne obstajata
        const auto &[it_a, _a] = nodes.try_emplace(a, a);
        const auto &[it_b, _b] = nodes.try_emplace(b, b);
        Node &node_a = it_a->second, &node_b = it_b->second;

        // dodaj povezavi vozliščema
        node_b.add_link_in(node_a);
        node_a.add_link_out(node_b);

        // največji id, največje število povezav do nekega vozlišča
        max_id = max(max_id, max(a, b));
        max_links = max(max_links, node_b.links_in.size());
    }
}

bool Graph::has_connection(const uint32_t a, const uint32_t b) const
{
    auto node = nodes.find(b);
    if (node == nodes.end()) return false;

    for (const Node *n : node->second.links_in) {
        if (n->id == a) return true;
    }
    return false;
}

void Graph::rank()
{
    const float d = 0.85f;

    for (auto &[id, node] : nodes) {
        node.rank = 1.0f / nnodes;
        node.rank_prev = 0;
    }

    bool stop = false;

    while (!stop) {
        stop = true;

        for (auto &[id, node] : nodes) {

            if (std::abs(node.rank - node.rank_prev) >= delta) {
                stop = false;

                float sum = 0;

                for (const Node *src : node.links_in) {
                    sum += src->rank / src->links_out.size();
                }

                sum *= d;
                node.rank_now = (1 - d) / nnodes + sum;
            }
        }
        for (auto &[id, node] : nodes) {
            node.rank      = node.rank_now;
            node.rank_prev = node.rank_now;
        }
    }
}

void Graph::rank_omp()
{
    // TODO: optimizacija
    const float d = 0.85f;

    for (auto &[id, node] : nodes) {
        node.rank = 1.0f / nnodes;
        node.rank_prev = 0;
    }

    bool stop = false;

    while (!stop) {
        stop = true;

        #pragma omp parallel for
        for (size_t id = 0; id < max_id; id++) {
            const auto &it = nodes.find(id);
            if (it == nodes.end()) continue;
            Node &node = it->second;

            if (std::abs(node.rank - node.rank_prev) >= delta) {
                if (stop) {
                    #pragma omp atomic write
                    stop = false;
                }

                float sum = 0;

                for (const Node *src : node.links_in) {
                    sum += src->rank / src->links_out.size();
                }

                sum *= d;
                node.rank_now = (1 - d) / nnodes + sum;
            }
        }

        #pragma omp parallel for
        for (size_t id = 0; id < max_id; id++) {
            const auto &it = nodes.find(id);
            if (it == nodes.end()) continue;
            Node &node = it->second;

            node.rank      = node.rank_now;
            node.rank_prev = node.rank_now;
        }
    }
}
