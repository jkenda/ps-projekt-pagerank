#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include "Graph.hpp"

using namespace std;

Node::Node(const id_t id)
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


Graph::Graph()
: nnodes(0), nedges(0), nsinks(0), max_id(0)
{
}

void Graph::read(const char *filename)
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
        }

        getline(is, line, '\n');
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
    node_a.add_link_out();
    nedges = 1;

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
        nedges++;

        // največji id
        max_id = max(max_id, max(a, b));
    }

    // rezerviraj prostor v vektorjih
    nnodes = nodes.size();
    nodes_v.reserve(nnodes);
    sink_nodes.reserve(nnodes);

    nsinks = 0;

    for (auto &[id, node] : nodes) {
        nodes_v.emplace_back(&node);

        if (node.nlinks_out == 0) {
            sink_nodes.emplace_back(&node);
            nsinks++;
        }
    }

    // zmanjšaj vektor na pravo velikost
    sink_nodes.shrink_to_fit();
}

uint32_t Graph::rank()
{
    bool stop = false;
    rank_t sink_sum = 0;
    uint32_t iterations = 0;

    for (Node *node : nodes_v) {
        node->rank = 1.0 / nnodes;
        node->rank_prev = 1.0;
    }

    while (true) {
        stop = true;
        sink_sum = 0;
        iterations++;

        for (Node *sink_node : sink_nodes) {
            sink_sum += sink_node->rank;
        }

        for (Node *node : nodes_v) {
            if (node->rank_prev == 0.0) continue;

            stop = false;

            rank_t sum = 0;

            for (const Node *src : node->links_in) {
                sum += src->rank / src->nlinks_out;
            }

            node->rank_new = ((1 - D) + D * sink_sum) / nnodes + D * sum;
        }

        if (stop) break;

        for (Node *node : nodes_v) {
            if (node->rank_prev == 0.0) continue;
            if (abs(node->rank - node->rank_prev) < DELTA) {
                node->rank_prev = 0.0;
            }
            else {
                node->rank_prev = node->rank;
                node->rank = node->rank_new;
            }
        }
    }

    return iterations;
}

uint32_t Graph::rank_omp()
{
    bool stop;
    rank_t sink_sum;
    uint32_t iterations = 0;

    #pragma omp parallel
    {
        #pragma omp for
        for (uint32_t i = 0; i < nnodes; i++) {
            nodes_v[i]->rank = 1.0 / nnodes;
            nodes_v[i]->rank_prev = 1.0;
        }

        while (true) {
            #pragma omp single
            {
                sink_sum = 0;
                iterations++;
            }

            bool l_stop = true;

            #pragma omp for reduction(+: sink_sum)
            for (uint32_t i = 0; i < nsinks; i++) {
                sink_sum += sink_nodes[i]->rank;
            }

            #pragma omp for schedule(dynamic, CHUNK_SIZE)
            for (uint32_t i = 0; i < nnodes; i++) {
                if (nodes_v[i]->rank_prev == 0.0) continue;

                l_stop = false;

                rank_t sum = 0;

                for (const Node *src : nodes_v[i]->links_in) {
                    sum += src->rank / src->nlinks_out;
                }

                nodes_v[i]->rank_new = ((1 - D) + D * sink_sum) / nnodes + D * sum;
            }

            #pragma omp single
            stop = true;
            #pragma omp barrier

            #pragma omp atomic
            stop &= l_stop;
            #pragma omp barrier

            if (stop) break;

            #pragma omp for
            for (uint32_t i = 0; i < nnodes; i++) {
                if (nodes_v[i]->rank_prev == 0.0) continue;
                if (abs(nodes_v[i]->rank - nodes_v[i]->rank_prev) < DELTA) {
                    nodes_v[i]->rank_prev = 0.0;
                }
                else {
                    nodes_v[i]->rank_prev = nodes_v[i]->rank;
                    nodes_v[i]->rank = nodes_v[i]->rank_new;
                }
            }
        }
    }

    return iterations;
}
