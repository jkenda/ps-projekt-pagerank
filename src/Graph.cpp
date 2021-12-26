#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include "Graph.hpp"

#define CHUNK_SIZE 128

#define DELTA (2e-19L)
#define D (0.85L)

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
    for (Node *node : nodes_v) {
        node->rank = 1.0f / nnodes;
        node->rank_prev = 0;
    }

    bool stop = false;
    // vsota rankov vseh ponorov
    // ponor = node oz. stran, ki nima izhodnih povezav 
    rank_t sink_sum = 0;

    uint32_t iterations = 0;

    while (!stop) {
        stop = true;
        sink_sum = 0;
        iterations++;

        for (Node *sink_node : sink_nodes) {
            sink_sum += sink_node->rank;
        }

        for (Node *node : nodes_v) {
            if (abs(node->rank - node->rank_prev) < DELTA) continue;

            stop = false;

            rank_t sum = 0;

            for (const Node *src : node->links_in) {
                sum += src->rank / src->nlinks_out;
            }

            sum *= D;
            node->rank_new = ((1 - D) + D * sink_sum) / nnodes + sum;
        }

        for (Node *node : nodes_v) {
            node->rank_prev = node->rank;
            node->rank      = node->rank_new;
        }
    }

    return iterations;
}

uint32_t Graph::rank_omp()
{
    bool stop = false;
    rank_t sink_sum = 0;
    uint32_t iterations = 0;

    #pragma omp parallel
    {
        #pragma omp for
        for (uint32_t i = 0; i < nnodes; i++) {
            Node &node = *nodes_v[i];

            node.rank = 1.0f / nnodes;
            node.rank_prev = 0;
        }

        while (!stop) {
            #pragma omp barrier
            #pragma omp single
            {
                stop = true;
                sink_sum = 0;
                iterations++;
            }

            #pragma omp for reduction(+: sink_sum) schedule(dynamic, CHUNK_SIZE)
            for (uint32_t i = 0; i < sink_nodes.size(); i++) {
                sink_sum  = sink_sum + sink_nodes[i]->rank;
            }

            #pragma omp for schedule(dynamic, CHUNK_SIZE)
            for (uint32_t i = 0; i < nnodes; i++) {
                Node &node = *nodes_v[i];
                if (abs(node.rank - node.rank_prev) < DELTA) continue;

                #pragma omp atomic write
                stop = false;

                rank_t sum = 0;

                for (const Node *src : node.links_in) {
                    sum += src->rank / src->nlinks_out;
                }

                sum *= D;
                node.rank_new = ((1 - D) + D * sink_sum) / nnodes + sum;
            }

            #pragma omp for schedule(dynamic, CHUNK_SIZE)
            for (uint32_t i = 0; i < nnodes; i++) {
                Node &node = *nodes_v[i];

                node.rank_prev = node.rank;
                node.rank      = node.rank_new;
            }
        }
    }

    return iterations;
}
