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
: id(id), link_in_ids(links_in), nlinks_in(nlinks_in), nlinks_out(nlinks_out)
{
}

Graph4CL::Graph4CL(const Graph& graph)
: nnodes(graph.nnodes), nedges(graph.nedges), max_id(graph.max_id)
{
    ids_v.reserve(nnodes);
    links_v.reserve(nedges);
    nodes_v.resize(max_id + 1);

    for (const auto &[id, node]: graph.nodes) {
        ids_v.push_back(id);
        nodes_v[id] = Node4CL(id, links_v.size(), node.links_in.size(), node.nlinks_out);
        
        for (const Node *l : node.links_in) {
            links_v.emplace_back(l->id);
        }
    }

    ids   = ids_v.data(); 
    links = links_v.data();
    nodes = nodes_v.data();
}


void Graph4CL_rank(Graph4CL *graph)
{
    #pragma omp parallel for
    for (uint32_t i = 0; i < graph->nnodes; i++) {
        int32_t id = graph->ids[i];
        Node4CL *node = &graph->nodes[id];

        node->rank = 1.0f / graph->nnodes;
        node->rank_prev = 0;
    }

    bool stop = false;

    while (!stop) {
        stop = true;

        #pragma omp parallel for
        for (uint32_t i = 0; i < graph->nnodes; i++) {
            int32_t id = graph->ids[i];
            Node4CL *node = &graph->nodes[id];

            if (abs(node->rank - node->rank_prev) < DELTA) continue;

            #pragma omp atomic write
            stop = false;

            float sum = 0;

            for (uint32_t i = 0; i < node->nlinks_in; i++) {
                uint32_t link_id = graph->links[node->link_in_ids + i];
                Node4CL *src = &graph->nodes[link_id];
                sum += src->rank / src->nlinks_out;
            }

            sum *= D;
            node->rank_new = (1 - D) / graph->nnodes + sum;
        }

        #pragma omp parallel for
        for (uint32_t i = 0; i < graph->nnodes; i++) {
            int32_t id = graph->ids[i];
            Node4CL *node = &graph->nodes[id];

            node->rank_prev = node->rank;
            node->rank      = node->rank_new;
        }
    }

}