#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include "Graph.hpp"

#define DELTA (1e-8)
#define D (0.85f)

using namespace std;

Node::Node()
: id(-1), rank(0)
{
}

Node::Node(const uint32_t id)
: id(id), nlinks_out(0)
{
}

void Node::add_link_in(const Node& link)
{
    links_in.push_back(&link);
}

void Node::add_link_out()
{
    nlinks_out++;
}


Graph::Graph(const char *filename)
{
    ifstream is(filename, ios::in);
    if (!is.is_open()) {
        exit(1);
    }

    string line, word;

    stringstream words;
    getline(is, line, '\n');

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

        getline(is, line, '\n');
    }
    words.clear();
    words.str(line);

    nodes.reserve(nnodes);
    nodes_v.reserve(nnodes);

    uint32_t a, b;
    words >> a; words >> b;

    // dodaj prvi vozlišči
    const auto &[it_a, _a] = nodes.try_emplace(a, a);
    const auto &[it_b, _b] = nodes.try_emplace(b, b);
    Node &node_a = it_a->second, &node_b = it_b->second;

    // dodaj prvo povezavo v množico
    node_b.add_link_in(node_a);
    node_a.add_link_out();

    // največji id
    max_id = max(a, b);

    while (is >> a, is >> b) {
        // dodaj vozlišči, če še ne obstajata
        const auto &[it_a, _a] = nodes.try_emplace(a, a);
        const auto &[it_b, _b] = nodes.try_emplace(b, b);
        Node &node_a = it_a->second, &node_b = it_b->second;

        // dodaj povezavi vozliščema
        node_b.add_link_in(node_a);
        node_a.add_link_out();

        // največji id
        max_id = max(max_id, max(a, b));
    }

    for (auto &[id, node] : nodes) {
        nodes_v.emplace_back(&node);
    }
}

void Graph::rank()
{
    for (Node *node : nodes_v) {
        node->rank = 1.0f / nnodes;
        node->rank_prev = 0;
    }

    bool stop = false;

    while (!stop) {
        stop = true;

        for (Node *node : nodes_v) {

            if (abs(node->rank - node->rank_prev) >= DELTA) {
                stop = false;

                float sum = 0;

                for (const Node *src : node->links_in) {
                    sum += src->rank / src->nlinks_out;
                }

                sum *= D;
                node->rank_new = (1 - D) / nnodes + sum;
            }
        }

        for (Node *node : nodes_v) {
            node->rank_prev = node->rank;
            node->rank      = node->rank_new;
        }
    }

}

void Graph::rank_omp()
{
    #pragma omp parallel for
    for (uint32_t i = 0; i < nnodes; i++) {
        Node &node = *nodes_v[i];

        node.rank = 1.0f / nnodes;
        node.rank_prev = 0;
    }

    bool stop = false;

    while (!stop) {
        stop = true;

        #pragma omp parallel for
        for (uint32_t i = 0; i < nnodes; i++) {
            Node &node = *nodes_v[i];

            if (abs(node.rank - node.rank_prev) >= DELTA) {
                #pragma omp atomic write
                stop = false;

                float sum = 0;

                for (const Node *src : node.links_in) {
                    sum += src->rank / src->nlinks_out;
                }

                sum *= D;
                node.rank_new = (1 - D) / nnodes + sum;
            }
        }

        #pragma omp parallel for
        for (uint32_t i = 0; i < nnodes; i++) {
            Node &node = *nodes_v[i];

            node.rank_prev = node.rank;
            node.rank      = node.rank_new;
        }
    }

}
