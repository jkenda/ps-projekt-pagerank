#include <iostream>
#include <algorithm>
#include "Graph.hpp"
#include "Timer.hpp"

using std::cout; using std::endl;

bool comp(const Node &a, const Node &b)
{
    return a.rank > b.rank;
}

int main(int argc, char **argv)
{
    cout << "reading file ...\r"; std::flush(cout);
    Graph pages("web-Google.txt");
    cout << "file read.      " << std::endl;

    // debug - ne bo v končni verziji
    {
        cout << "0 -> 11342: " << (pages.has_connection(0, 11342) ? "yes" : "no") << endl;
        cout << "11342 -> 0: " << (pages.has_connection(11342, 0) ? "yes" : "no") << endl;
        cout << "11342 -> 1: " << (pages.has_connection(11342, 1) ? "yes" : "no") << endl;

        cout << pages.nodes[11342].id << ": ";
        for (const Node *l : pages.nodes[11342].links_in) {
            cout << l->id << " ";
        }
        cout << endl;
        cout << pages.max_id << '\n' << pages.max_links << endl;
    }

    cout << "Število strani : " << pages.nnodes << '\n';
    cout << "Število povezav: " << pages.nedges << '\n';

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