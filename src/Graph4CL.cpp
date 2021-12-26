#include "Graph4CL.hpp"
#include <cstdio>
#include <cmath>

#define DELTA (1e-16f)
#define D (0.85f)

using namespace std;

Node4CL::Node4CL()
{
}

Node4CL::Node4CL(uint32_t id, size_t nlinks_in, size_t nlinks_out)
: id(id), nlinks_in(nlinks_in), nlinks_out(nlinks_out)
{
}

Graph4CL::Graph4CL(const Graph& graph)
: nnodes(graph.nnodes), nedges(graph.nedges), max_id(graph.max_id), nsinks(graph.nsinks)
{
    nodes_v.resize(max_id + 1);

    ids_v.reserve(nnodes);
    links_v.reserve(nedges);
    sinks_v.reserve(nsinks);
    offsets_v.reserve(nnodes);

    for (const auto &[id, node] : graph.nodes) {
        uint32_t offset = links_v.size();
        uint32_t nlinks_in = node.links_in.size();

        nodes_v[id] = Node4CL(id, nlinks_in, node.nlinks_out);
        
        ids_v.emplace_back(id);
        offsets_v.emplace_back(offset);
        
        for (const Node *src : node.links_in) {
            links_v.emplace_back(src->id);
        }

        if (node.nlinks_out == 0) {
            sinks_v.emplace_back(node.id);
        }
    }

    ids     = ids_v.data(); 
    links   = links_v.data();
    nodes   = nodes_v.data();
    sinks   = sinks_v.data();
    offsets = offsets_v.data();
}


uint32_t Graph4CL_rank(Graph4CL *graph)
{
    // #pragma omp parallel for
    for (uint32_t i = 0; i < graph->nnodes; i++) {
        int32_t id = graph->ids[i];
        Node4CL *node = &graph->nodes[id];

        node->rank = 1.0f / graph->nnodes;
        node->rank_prev = 0;
    }

    bool stop = false;
    rank_t sink_sum = 0;
    uint32_t iterations = 0;

    while (!stop) {
        stop = true;
        sink_sum = 0;
        iterations++;

        for (uint32_t i = 0; i < graph->nsinks; i++) {
            int32_t id = graph->sinks[i];
            Node4CL *sink = &graph->nodes[id];
            sink_sum += sink->rank;
        }

        // #pragma omp parallel for
        for (uint32_t i = 0; i < graph->nnodes; i++) {
            int32_t id = graph->ids[i];
            int32_t offset = graph->offsets[i];
            Node4CL *node = &graph->nodes[id];

            if (abs(node->rank - node->rank_prev) < DELTA) continue;

            // #pragma omp atomic write
            stop = false;

            rank_t sum = 0;

            for (uint32_t i = 0; i < node->nlinks_in; i++) {
                uint32_t link_id = graph->links[offset + i];
                Node4CL *src = &graph->nodes[link_id];
                sum += src->rank / src->nlinks_out;
            }

            sum *= D;
            node->rank_new = ((1 - D) + D * sink_sum) / graph->nnodes + sum;
        }

        // #pragma omp parallel for
        for (uint32_t i = 0; i < graph->nnodes; i++) {
            int32_t id = graph->ids[i];
            Node4CL *node = &graph->nodes[id];

            node->rank_prev = node->rank;
            node->rank      = node->rank_new;
        }
    }

    return iterations;
}