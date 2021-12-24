#include <iostream>
#include <algorithm>
#include "Graph.hpp"
#include "Graph4CL.hpp"
#include "Timer.hpp"

using namespace std;

bool comp(const Node &a, const Node &b)
{
    return a.rank > b.rank;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        cout << "uporaba: " << argv[0] << " <filename>\n";
        exit(1);
    }

    const char *filename = argv[1];

    Graph pages;

    cout << "berem ... "; flush(cout);
    {
        TIMER("")
        pages.read(filename);
    }

    cout << "gradim strukturo za OpenCL... "; flush(cout);
    Graph4CL pages4cl(pages);
    cout << "zgrajeno.\n\n";

    cout << "Število strani  : " << pages.nnodes << '\n';
    cout << "Število povezav : " << pages.nedges << '\n';
    cout << "Največji id     : " << pages.max_id << '\n';
    cout << std::endl;

    float sum_seq = 0, sum_omp = 0, sum_ocl = 0;

    {
        TIMER("zaporedno : ")
        pages.rank();
    }

    // seštej range strani
    for (const auto &[id, node] : pages.nodes) {
        sum_seq += node.rank;
    }

    {
        TIMER("OpenMP    : ")
        pages.rank_omp();
    }

    // seštej range strani
    for (const auto &[id, node] : pages.nodes) {
        sum_omp += node.rank;
    }

    {
        TIMER("OpenCL    : ")
        Graph4CL_rank(&pages4cl);
    }
    cout << '\n';

    // seštej range strani
    for (int i = 0; i < pages4cl.nnodes; i++) {
        int32_t id = pages4cl.ids[i];
        Node4CL &node = pages4cl.nodes[id];
        
        sum_ocl += node.rank;
    }

    // rangi se morajo sešteti v 1
    cout << "seštevki rangov: " << sum_seq << ", " << sum_omp << ", " << sum_ocl << '\n';
    cout << '\n';

    std::vector<Node> ranked;
    for (auto &[id, node] : pages.nodes) {
        ranked.emplace_back(node);
    }
    std::sort(ranked.begin(), ranked.end(), comp);

    cout << "strani z največjim rangom:\n";

    for (int i = 0; i < 10; i++) {
        printf("%8d: %.3e\n", ranked[i].id, ranked[i].rank);
    }
}