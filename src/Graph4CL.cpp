#include "Graph4CL.hpp"
#include <cstdio>
#include <cmath>

#define DELTA (1e-8)
#define D (0.85f)

using namespace std;

Node4CL::Node4CL()
: nlinks_in(-1)
{
}

Node4CL::Node4CL(std::uint32_t id, std::uint32_t links_in, std::size_t nlinks_in, std::size_t nlinks_out)
: id(id), nlinks_in(nlinks_in), nlinks_out(nlinks_out)
{
}

Graph4CL::Graph4CL(const Graph& graph)
: nnodes(graph.nnodes), nedges(graph.nedges), max_id(graph.max_id), nsinks(graph.nsinks)
{
    ids_v.reserve(nnodes);
    links_v.reserve(nedges);
    nodes_v.reserve(max_id + 1);
    sinks_v.reserve(nsinks);
    offsets_v.reserve(nnodes);

    uint32_t offset = 0;

    for (const auto &[id, node]: graph.nodes) {
        ids_v.push_back(id);
        nodes_v[id] = Node4CL(id, links_v.size(), node.links_in.size(), node.nlinks_out);
        
        offsets_v.push_back(offset);
        offset += node.links_in.size();
        
        for (const Node *l : node.links_in) {
            links_v.emplace_back(l->id);
        }

        if (node.nlinks_out == 0) {
            sinks_v.push_back(node.id);
        }
    }

    ids   = ids_v.data(); 
    links = links_v.data();
    nodes = nodes_v.data();
    sinks = sinks_v.data();
    offsets = offsets_v.data();
}


void Graph4CL_rank(Graph4CL *graph)
{
    // #pragma omp parallel for
    for (uint32_t i = 0; i < graph->nnodes; i++) {
        int32_t id = graph->ids[i];
        Node4CL *node = &graph->nodes[id];

        node->rank = 1.0f / graph->nnodes;
        node->rank_prev = 0;
    }

    bool stop = false;
    float sink_sum = 0;

    while (!stop) {
        stop = true;
        sink_sum = 0;

        for (uint32_t i = 0; i < graph->nsinks; i++) {
            int32_t sink_id = graph->sinks[i];
            Node4CL *sink = &graph->nodes[sink_id];
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

            float sum = 0;

            for (uint32_t i = 0; i < node->nlinks_in; i++) {
                uint32_t link_id = graph->links[i+offset];
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

}