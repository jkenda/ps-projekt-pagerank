#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdarg>
#include <omp.h>
#include "Graph.hpp"
#include "Graph4CL.hpp"
#include "Timer.hpp"

using namespace std;

bool comp(const Node &a, const Node &b){ return a.rank > b.rank; }
bool comp4cl(const Node4CL &a, const Node4CL &b){ return a.rank > b.rank; }

int main(int argc, char **argv)
{
    if (argc != 2) {
        cout << "uporaba: " << argv[0] << " <filename>\n";
        exit(1);
    }

    const char *filename = argv[1];

    // instanciraj graf
    Graph pages;

    // preberi strukturo iz datoteke
    printf("\nBRANJE IZ DATOTEKE\n");
    float time_read;
    cout << "\tberem ... "; flush(cout);
    {
        TIMER(time_read)
        pages.read(filename);
    }
    cout << time_read << " s\n";

    // sestavi graph za OpenCL iz obstoječega grafa
    cout << "\tgradim strukturo za OpenCL... ";
    Graph4CL pages4cl(pages);
    cout << "zgrajeno.\n\n";

    printf("OSNOVNI PODATKI\n");
    // izpiši osnovne informacije o grafu
    cout << "\tŠtevilo strani    : " << pages.nnodes << ",\n";
    cout << "\tštevilo povezav   : " << pages.nedges << ",\n";
    cout << "\tštevilo ponorov   : " << pages.nsinks << ",\n";
    cout << "\tnajvečji id       : " << pages.max_id << ".\n";
    cout << "\tvelikost podatkov : " << pages4cl.data_size() << " MB.\n";
    cout << '\n';

    rank_t sum_seq = 0, sum_omp = 0, sum_ocl = 0;
    float time_seq, time_omp, time_ocl;
    uint32_t iter_seq, iter_omp, iter_ocl;

    /* RANGIRAJ STRANI */

    printf("IZVAJANJE\n");
    // sekvenčni algoritem
    printf("\t┌───────────┬──────┬───────────┬───────────┬────────────┐\n");
    printf("\t│ %-9s │ %-4s │ %-9s │ %-9s │ %-10s │\n", "nacin", "niti", "iteracije", "cas [s]", "pohitritev");
    printf("\t├───────────┼──────┼───────────┼───────────┼────────────┤\n");
    flush(cout);
    
    {
        TIMER(time_seq)
        iter_seq = pages.rank();
    }
    printf("\t│ %-9s │ %4u │ %9u │ %9.5f │ %10.5f │\n", "sekvencno", 1, iter_seq, time_seq, time_seq / time_seq);
    flush(cout);

    // seštej range strani
    for (const Node &node : pages.nodes) {
        sum_seq += node.rank;
    }

    // sortiraj range strani po velikosti
    vector<Node> ranked_seq;
    for (const Node &node : pages.nodes) {
        ranked_seq.push_back(node);
    }
    sort(ranked_seq.begin(), ranked_seq.end(), comp);

    // OpenMP
    uint32_t max_threads = omp_get_max_threads();

    for (int nthreads = 1; nthreads <= max_threads; nthreads *= 2) {
        {
            TIMER(time_omp)
            iter_omp = pages.rank_omp(nthreads);
        }
        printf("\t│ %-9s │ %4u │ %9u │ %9.5f │ %10.5f │\n", "OpenMP", nthreads, iter_omp, time_omp, time_seq / time_omp);
        flush(cout);
    }

    // seštej range strani
    for (const Node &node : pages.nodes) {
        sum_omp += node.rank;
    }

    // sortiraj range strani po velikosti
    vector<Node> ranked_omp;
    for (const Node &node : pages.nodes) {
        ranked_omp.push_back(node);
    }
    sort(ranked_omp.begin(), ranked_omp.end(), comp);

    // OpenCL    
    {
        TIMER(time_ocl)
        iter_ocl = Graph4CL_rank(&pages4cl);
    }
    printf("\t│ %-9s │ %4u │ %9u │ %9.5f │ %10.5f │\n", "OpenCL", max_threads, iter_ocl, time_ocl, time_seq / time_ocl);
    printf("\t└───────────┴──────┴───────────┴───────────┴────────────┘\n");
    printf("\n");

    // seštej range strani
    for (uint32_t i = 0; i < pages4cl.nnodes; i++) {
        // Node4CL &node = pages4cl.nodes[i]; 
        // sum_ocl += node.rank;
        sum_ocl += pages4cl.ranks[i];
    }

    // sortiraj range strani po velikosti
    vector<Node4CL> ranked_ocl;
    for (uint32_t i = 0; i < pages4cl.nnodes; i++) {
        Node4CL &node = pages4cl.nodes[i];
        node.rank = pages4cl.ranks[i];
        ranked_ocl.emplace_back(node);
    }
    sort(ranked_ocl.begin(), ranked_ocl.end(), comp4cl);

    // rangi se morajo sešteti v 1
    printf("SEŠTEVKI RANGOV\n");
    printf("\tRang strani je verjetnost - vsota verjetnosti mora biti 1.\n");
    printf("\t┌───────────┬─────────────────┐\n");
    printf("\t│ %-9s │ %-15s │\n", "nacin", "sestevek");
    printf("\t├───────────┼─────────────────┤\n");
    printf("\t│ %-9s │ %15.13lf │\n", "sekvencno", sum_seq);
    printf("\t│ %-9s │ %15.13lf │\n", "OpenMP"   , sum_omp);
    printf("\t│ %-9s │ %15.13lf │\n", "OpenCL"   , sum_ocl);
    printf("\t└───────────┴─────────────────┘\n");
    printf("\n");

    int32_t ndisplay = min(ranked_seq.size(), 10UL);

    printf("10 STRANI Z NAVIŠJIM RANGOM\n");
    printf("\t┌────────┬─────────┬───────────┐    ┌────────┬─────────┬───────────┐    ┌────────┬─────────┬───────────┐\n");
    printf("\t│ %-6s │ %7s │ %-9s │    ", "id", "povezav", "rank");
    printf(  "│ %-6s │ %7s │ %-9s │    ", "id", "povezav", "rank");
    printf(  "│ %-6s │ %7s │ %-9s │\n", "id", "povezav", "rank");
    printf("\t├────────┼─────────┼───────────┤    ├────────┼─────────┼───────────┤    ├────────┼─────────┼───────────┤\n");
    for (uint32_t i = 0; i < ndisplay; i++) {
        printf("\t│ %6u │ %7lu │ %8.3e │    ", ranked_seq[i].id, ranked_seq[i].links_in.size(), ranked_seq[i].rank);
        printf(  "│ %6u │ %7lu │ %8.3e │    ", ranked_omp[i].id, ranked_omp[i].links_in.size(), ranked_omp[i].rank);
        printf(  "│ %6u │ %7u │ %8.3e │\n"   , ranked_ocl[i].id, ranked_ocl[i].nlinks_in, ranked_ocl[i].rank);
    }
    printf("\t└────────┴─────────┴───────────┘    └────────┴─────────┴───────────┘    └────────┴─────────┴───────────┘\n");
    printf("\t                    (sekvencno)                            (OpenMP)                            (OpenCL)\n");
    printf("\n");

    printf("10 STRANI Z NAJNIŽJIM RANGOM\n");
    printf("\t┌────────┬─────────┬───────────┐    ┌────────┬─────────┬───────────┐    ┌────────┬─────────┬───────────┐\n");
    printf("\t│ %-6s │ %7s │ %-9s │    ", "id", "povezav", "rank");
    printf(  "│ %-6s │ %7s │ %-9s │    ", "id", "povezav", "rank");
    printf(  "│ %-6s │ %7s │ %-9s │\n", "id", "povezav", "rank");
    printf("\t├────────┼─────────┼───────────┤    ├────────┼─────────┼───────────┤    ├────────┼─────────┼───────────┤\n");
    for (uint32_t i = pages.nnodes; i > pages.nnodes-ndisplay; i--) {
        printf("\t│ %6u │ %7lu │ %8.3e │    ", ranked_seq[i-1].id, ranked_seq[i-1].links_in.size(), ranked_seq[i-1].rank);
        printf(  "│ %6u │ %7lu │ %8.3e │    ", ranked_omp[i-1].id, ranked_omp[i-1].links_in.size(), ranked_omp[i-1].rank);
        printf(  "│ %6u │ %7u │ %8.3e │\n"   , ranked_ocl[i-1].id, ranked_ocl[i-1].nlinks_in, ranked_ocl[i-1].rank);
    }
    printf("\t└────────┴─────────┴───────────┘    └────────┴─────────┴───────────┘    └────────┴─────────┴───────────┘\n");
    printf("\t                    (sekvencno)                            (OpenMP)                            (OpenCL)\n");
    printf("\n");
}