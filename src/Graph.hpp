#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#define CHUNK_SIZE 128

#define DELTA 1e-16L
#define D 0.85L

typedef double rank_t;
typedef std::uint32_t id_t;

struct Node
{
    id_t id;                            // id (številka strani)
    rank_t rank, rank_new, rank_prev;   // rangiranje
    uint32_t nlinks_out;                // povezave iz strani
    std::vector<const Node *> links_in; // povezave do strani

    Node(const id_t &id);

    void add_link_in(const Node &link);
    void add_link_out();
};

struct Graph
{
    std::vector<Node> nodes;          // strani
    std::vector<Node *> sink_nodes;   // kazalci do ponornih strani (strani brez izhodnih povezav)
    uint32_t nnodes, nedges, nsinks;  // št. strani, povezav
    id_t max_id;                      // največji id strani

    Graph();

    void read(const char *filename);

    uint32_t rank();
    uint32_t rank_omp(const uint32_t &nthreads);
};

/*
Input: Let G represents set of web pages or nodes 
Output: A file showing PageRank for each web page

for each node n in Graph do in parallel
    n.prev_PR := 1.0

for 1 to k do

    for each node n Graph do in parallel
        n.PR := 0

        for each page p in n. inlinkneighbors do
            sum := 0.0

            for each page q in p.outlinkneighbors do
                sum q.prev_PR
                n.PR += (1-d) + d * p.PR

    difference := 0
    for each node n Graph do in parallel
        difference := maximum(n.PR-n.prev_PR, mx)

    if difference < threshold do
        stop the program

    for each node n Graph do in parallel
        n.prev_PR=n.PR

*/