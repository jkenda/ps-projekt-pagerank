#include "Graph4CL.hpp"
#include <cstdio>
#include <cmath>

#define CHUNK_SIZE 100

#define DELTA (1e-16f)
#define D (0.85f)

using namespace std;

Node4CL::Node4CL()
{
}

Node4CL::Node4CL(uint32_t id, uint32_t links_in, size_t nlinks_in, size_t nlinks_out)
: id(id), link_in_ids(links_in), nlinks_in(nlinks_in), nlinks_out(nlinks_out)
{
}

Graph4CL::Graph4CL(const Graph& graph)
: nnodes(graph.nnodes), nedges(graph.nedges), nsinks(graph.nsinks), max_id(graph.max_id)
{
    ids_v.reserve(nnodes);
    links_v.reserve(nedges);
    sinks_v.reserve(nsinks);
    nodes_v.resize(max_id + 1);

    for (const auto &[id, node] : graph.nodes) {
        ids_v.push_back(id);
        nodes_v[id] = Node4CL(id, links_v.size(), node.links_in.size(), node.nlinks_out);
        
        for (const Node *src : node.links_in) {
            links_v.emplace_back(src->id);
        }
    }

    for (const Node *node : graph.sink_nodes) {
        sinks_v.emplace_back(node->id);
    }

    ids   = ids_v.data(); 
    links = links_v.data();
    nodes = nodes_v.data();
    sinks = sinks_v.data();
}


uint32_t Graph4CL_rank(Graph4CL *graph)
{
    #pragma omp parallel for
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
            Node4CL *sink_node = &graph->nodes[id];

            sink_sum += sink_node->rank;
        }

        #pragma omp parallel
        {   
            // https://stackoverflow.com/questions/4749493/strange-double-behaviour-in-openmp
            // #pragma omp reduction(+: sink_sum) schedule(dynamic, chunk_size)
            // for (Node *sink_node : sink_nodes) {
            //     sink_sum += sink_node->rank;
            // }

            #pragma omp for schedule(dynamic, CHUNK_SIZE)
            for (uint32_t i = 0; i < graph->nnodes; i++) {
                int32_t id = graph->ids[i];
                Node4CL *node = &graph->nodes[id];

                if (abs(node->rank - node->rank_prev) < DELTA) continue;

                #pragma omp atomic write
                stop = false;

                rank_t sum = 0;

                for (uint32_t i = 0; i < node->nlinks_in; i++) {
                    uint32_t link_id = graph->links[node->link_in_ids + i];
                    Node4CL *src = &graph->nodes[link_id];
                    sum += src->rank / src->nlinks_out;
                }

                sum *= D;
                node->rank_new = ((1 - D) + D * sink_sum) / graph->nnodes + sum;
            }

            #pragma omp for schedule(dynamic, CHUNK_SIZE)
            for (uint32_t i = 0; i < graph->nnodes; i++) {
                int32_t id = graph->ids[i];
                Node4CL *node = &graph->nodes[id];

                node->rank_prev = node->rank;
                node->rank      = node->rank_new;
            }
        }
    }

    return iterations;
}