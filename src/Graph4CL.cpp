#include "Graph4CL.hpp"
#include <cmath>
#include <cstdlib>
#include <CL/cl.h>
#include <iostream>

#define WORKGROUP_SIZE	(256)
#define MAX_SOURCE_SIZE (16384)

using namespace std;

Node4CL::Node4CL(id_t id, uint32_t nlinks_in, uint32_t nlinks_out, uint32_t links_offset)
: id(id), nlinks_in(nlinks_in), nlinks_out(nlinks_out), links_offset(links_offset)
{
}

Graph4CL::Graph4CL(const Graph &graph)
: nnodes(graph.nnodes), nedges(graph.nedges), max_id(graph.max_id), nsinks(graph.nsinks)
{
    offsets_v.reserve(max_id + 1);

    nodes_v.reserve(nnodes);
    link_ids_v.reserve(nedges);
    sink_offsets_v.reserve(nsinks);

    for (const Node &node : graph.nodes) {
        uint32_t nodes_offset = nodes_v.size();
        uint32_t links_offset = link_ids_v.size();
        uint32_t nlinks_in = node.links_in.size();

        offsets_v[node.id] = nodes_offset;
        nodes_v.emplace_back(node.id, nlinks_in, node.nlinks_out, links_offset);
        
        if (node.nlinks_out == 0) {
            sink_offsets_v.emplace_back(nodes_offset);
        }

        for (const Node *src : node.links_in) {
            link_ids_v.emplace_back(src->id);
        }

    }

    nodes    = nodes_v.data();
    offsets  = offsets_v.data(); 
    link_ids = link_ids_v.data();
    sink_offsets = sink_offsets_v.data();
}

float Graph4CL::data_size()
{
    return (nodes_v.size() * sizeof(Node4CL)
         + offsets_v.size() * sizeof(decltype(*offsets))
         + link_ids_v.size() * sizeof(decltype(*link_ids))
         + sink_offsets_v.size() * sizeof(decltype(*sink_offsets)))
        / 1'000'000.0F;
}


uint32_t Graph4CL_rank(Graph4CL *graph)
{
    bool stop;
    rank_t sink_sum;
    uint32_t iterations = 0;

    #pragma omp parallel
    {
        #pragma omp for
        for (uint32_t i = 0; i < graph->nnodes; i++) {
            Node4CL *node = &graph->nodes[i];

            node->rank = 1.0 / graph->nnodes;
            node->rank_prev = 0.0;
        }

        while (true) {

            #pragma omp single
            {
                sink_sum = 0;
                iterations++;
            }

            bool l_stop = true;

            #pragma omp for reduction(+: sink_sum)
            for (uint32_t i = 0; i < graph->nsinks; i++) {
                uint32_t offset = graph->sink_offsets[i];
                Node4CL *sink   = &graph->nodes[offset];

                sink_sum += sink->rank;
            }

            #pragma omp for schedule(dynamic, CHUNK_SIZE)
            for (uint32_t i = 0; i < graph->nnodes; i++) {
                Node4CL *node = &graph->nodes[i];
                if (abs(node->rank - node->rank_prev) < DELTA) continue;

                l_stop = false;

                rank_t sum = 0;

                for (uint32_t i = 0; i < node->nlinks_in; i++) {
                    uint32_t link_node = graph->link_ids[node->links_offset + i];
                    uint32_t offset = graph->offsets[link_node];
                    Node4CL *src = &graph->nodes[offset];

                    sum += src->rank / src->nlinks_out;
                }

                node->rank_new = ((1 - D) + D * sink_sum) / graph->nnodes + D * sum;
            }

            #pragma omp single
            stop = true;
            #pragma omp barrier

            // vsi lokalni ustavitveni pogoji izpolnjeni -> izpolnjen ustavitveni pogoj
            #pragma omp atomic
            stop &= l_stop;
            #pragma omp barrier

            if (stop) break;

            #pragma omp parallel for
            for (uint32_t i = 0; i < graph->nnodes; i++) {
                Node4CL *node = &graph->nodes[i];

                node->rank_prev = node->rank;
                node->rank      = node->rank_new;
            }

        }
    }

    return iterations;
}

void Graph4CL_rank_GPU(Graph4CL *graph) {
    
}