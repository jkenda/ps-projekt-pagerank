#include <iostream>
#include <algorithm>
#include "Graph.hpp"
#include "Graph4CL.hpp"
#include "Timer.hpp"

using std::cout; using std::endl;

bool comp(const Node &a, const Node &b)
{
    return a.rank > b.rank;
}

int main(int argc, char **argv)
{
    cout << "berem ...\r"; std::flush(cout);
    Graph pages("web-Google.txt");
    cout << "datoteka prebrana." << '\n';

    cout << "Število strani : " << pages.nnodes << '\n';
    cout << "Število povezav: " << pages.nedges << '\n';
    cout << "Največji id    : " << pages.max_id << std::endl;

    {
        TIMER("sequential")
        pages.rank();
    }

    for (int i = 0; i < 10; i++) {
        cout << i << ": " << pages.nodes[i].rank << '\n';
    }

    {
        TIMER("OpenMP")
        pages.rank_omp();
    }

    cout << '\n';

    for (int i = 0; i < 10; i++) {
        cout << i << ": " << pages.nodes[i].rank << '\n';
    }

    Graph4CL pages4cl(pages);

    {
        TIMER("OpenCL")
        Graph4CL_rank(&pages4cl);
    }

    cout << '\n';

    for (int i = 0; i < 10; i++) {
        cout << i << ": " << pages4cl.nodes[i].rank << '\n';
    }


    /*
    std::vector<Node> ranked;
    for (auto &[id, node] : pages.nodes) {
        ranked.emplace_back(node);
    }
    std::sort(ranked.begin(), ranked.end(), comp);

    for (int i = 0; i < 10; i++) {
        cout << i << ": " << ranked[i].rank << '\n';
    }
    */
}