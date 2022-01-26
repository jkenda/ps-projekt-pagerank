#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "Graph.hpp"

using namespace std;

Node::Node(const id_t &id)
: id(id), nlinks_out(0)
{
}

void Node::add_link_in(const Node &link)
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

    unordered_map<uint32_t, Node *> nodes_map;
    nodes_map.reserve(nnodes);
    nodes.reserve(nnodes);

    uint32_t l_id, r_id;
    words >> l_id; words >> r_id;

    // dodaj prvi vozlišči
    nodes.emplace_back(l_id);
    nodes_map.insert({l_id, &nodes.back()});
    nodes.emplace_back(r_id);
    nodes_map.insert({r_id, &nodes.back()});
    Node &l_node = *nodes_map[l_id], &r_node = *nodes_map[r_id];

    // dodaj prvo povezavo
    r_node.add_link_in(l_node);
    l_node.add_link_out();
    nedges = 1;

    // začetni največji id
    max_id = max(l_id, r_id);

    while (is >> l_id, is >> r_id) {
        // dodaj vozlišči, če še ne obstajata
        if (nodes_map.find(l_id) == nodes_map.end()) {
            nodes.emplace_back(l_id);
            nodes_map.insert({l_id, &nodes.back()});
        }
        if (nodes_map.find(r_id) == nodes_map.end()) {
            nodes.emplace_back(r_id);
            nodes_map.insert({r_id, &nodes.back()});
        }
        Node &l_node = *nodes_map[l_id], &r_node = *nodes_map[r_id];

        // dodaj povezavi vozliščema
        r_node.add_link_in(l_node);
        l_node.add_link_out();
        nedges++;

        // največji id
        max_id = max(max_id, max(l_id, r_id));
    }

    // rezerviraj prostor v vektorjih
    nnodes = nodes.size();
    sink_nodes.reserve(nnodes);

    for (Node &node : nodes) {
        if (node.nlinks_out == 0) {
            sink_nodes.push_back(&node);
        }
    }

    // zmanjšaj vektor na pravo velikost
    sink_nodes.shrink_to_fit();
    nsinks = sink_nodes.size();
}

uint32_t Graph::rank()
{
    bool stop = false;
    rank_t sink_sum = 0;
    uint32_t iterations = 0;

    for (Node &node : nodes) {
        node.rank = 1.0 / nnodes;
        node.rank_prev = 1.0;
    }

    while (true) {
        stop = true;
        sink_sum = 0;
        iterations++;

        for (Node *sink_node : sink_nodes) {
            sink_sum += sink_node->rank;
        }

        for (Node &node : nodes) {
            if (node.rank_prev == 0.0) continue;

            stop = false;

            rank_t sum = 0;

            for (const Node *src : node.links_in) {
                sum += src->rank / src->nlinks_out;
            }

            node.rank_new = ((1.0 - D) + D * sink_sum) / nnodes + D * sum;
        }

        if (stop) break;

        for (Node &node : nodes) {
            if (node.rank_prev == 0.0) continue;
            if (abs(node.rank - node.rank_prev) < DELTA) {
                node.rank = node.rank_new;
                node.rank_prev = 0.0;
            }
            else {
                node.rank_prev = node.rank;
                node.rank = node.rank_new;
            }
        }
    }

    return iterations;
}

uint32_t Graph::rank_omp(const uint32_t &nthreads)
{
    bool stop;
    rank_t sink_sum;
    uint32_t iterations = 0;

    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for
        for (uint32_t i = 0; i < nnodes; i++) {
            nodes[i].rank = 1.0 / nnodes;
            nodes[i].rank_prev = 1.0;
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

            #pragma omp for
            for (uint32_t i = 0; i < nnodes; i++) {
                if (nodes[i].rank_prev == 0.0) continue;

                l_stop = false;

                rank_t sum = 0;

                for (const Node *src : nodes[i].links_in) {
                    sum += src->rank / src->nlinks_out;
                }

                nodes[i].rank_new = ((1.0 - D) + D * sink_sum) / nnodes + D * sum;
            }

            #pragma omp single
            stop = true;
            #pragma omp barrier

            // izpolnjeni vsi lokalni ustavitveni pogoji -> izpolnjen ustavitveni pogoj
            #pragma omp atomic
            stop &= l_stop;
            #pragma omp barrier

            if (stop) break;

            #pragma omp for
            for (uint32_t i = 0; i < nnodes; i++) {
                if (nodes[i].rank_prev == 0.0) continue;
                if (abs(nodes[i].rank_new - nodes[i].rank_prev) < DELTA) {
                    nodes[i].rank = nodes[i].rank_new;
                    nodes[i].rank_prev = 0.0;
                }
                else {
                    nodes[i].rank_prev = nodes[i].rank;
                    nodes[i].rank = nodes[i].rank_new;
                }
            }
        }
    }

    return iterations;
}
