#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdarg>
#include <omp.h>
#include <float.h>
#include "Graph.hpp"
#include "Timer.hpp"

using namespace std;

bool comp(const Node &a, const Node &b){ return a.rank > b.rank; }

uint32_t eq_omp(vector<Node> seq, vector<Node> omp)
{
    uint32_t eq = 0;
    for (size_t i = 0; i < seq.size(); i++) {
        if (seq[i].id == omp[i].id) eq++;
    }
    return eq;
}

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

    printf("OSNOVNI PODATKI\n");
    // izpiši osnovne informacije o grafu
    cout << "\tŠtevilo strani    : " << pages.nnodes << ",\n";
    cout << "\tštevilo povezav   : " << pages.nedges << ",\n";
    cout << "\tštevilo ponorov   : " << pages.nsinks << ",\n";
    cout << "\tnajvečji id       : " << pages.max_id << ".\n";
    cout << '\n';

    rank_t sum_seq = 0, sum_omp = 0;
    float time_seq, time_omp;
    uint32_t iter_seq, iter_omp;

    /* RANGIRAJ STRANI */

    printf("IZVAJANJE\n");
    // sekvenčni algoritem
    printf("\t┌───────────┬─────────┬───────────┬───────────┬────────────┐\n");
    printf("\t│ %-9s │ %-7s │ %-9s │ %-9s │ %-10s │\n", "nacin", "niti/WG", "iteracije", "cas [s]", "pohitritev");
    printf("\t├───────────┼─────────┼───────────┼───────────┼────────────┤\n");
    flush(cout);
    
    {
        TIMER(time_seq)
        iter_seq = pages.rank();
    }
    printf("\t│ %-9s │ %7u │ %9u │ %9.5f │ %10.5f │\n", "sekvencno", 1, iter_seq, time_seq, time_seq / time_seq);
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
        printf("\t│ %-9s │ %7u │ %9u │ %9.5f │ %10.5f │\n", "OpenMP", nthreads, iter_omp, time_omp, time_seq / time_omp);
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

    printf("\t└───────────┴─────────┴───────────┴───────────┴────────────┘\n\n");

    // rangi se morajo sešteti v 1
    printf("SEŠTEVKI RANGOV\n");
    printf("\tRang strani je verjetnost - vsota verjetnosti mora biti 1.\n");
    printf("\t┌───────────┬─────────────────┐\n");
    printf("\t│ %-9s │ %-15s │\n", "nacin", "sestevek");
    printf("\t├───────────┼─────────────────┤\n");
    printf("\t│ %-9s │ %15.13lf │\n", "sekvencno", sum_seq);
    printf("\t│ %-9s │ %15.13lf │\n", "OpenMP"   , sum_omp);
    printf("\t└───────────┴─────────────────┘\n");
    printf("\n");

    int32_t ndisplay = min(ranked_seq.size(), 10UL);

    printf("10 STRANI Z NAVIŠJIM RANGOM IN 10 Z NAJNIŽJIM\n");
    printf("\t┌──────────┬─────────────┬───────────┐        ┌──────────┬─────────────┬───────────┐\n");
    printf("\t│ %-8s │ %-11s │ %-9s │        ", "id", "st. povezav", "rang");
    printf(  "│ %-8s │ %-11s │ %-9s │\n",       "id", "st. povezav", "rang");
    printf("\t├──────────┼─────────────┼───────────┤        ├──────────┼─────────────┼───────────┤\n");
    for (uint32_t i = 0; i < ndisplay; i++) {
        uint32_t j = pages.nnodes - 1 - i;
        printf("\t│ %8u │ %11lu │ %8.3e │        ", ranked_omp[i].id, ranked_omp[i].links_in.size(), ranked_omp[i].rank);
        printf(  "│ %8u │ %11lu │ %8.3e │\n"      , ranked_omp[j].id, ranked_omp[j].links_in.size(), ranked_omp[j].rank);
    }
    printf("\t└──────────┴─────────────┴───────────┘        └──────────┴─────────────┴───────────┘\n");
    printf("\t                           (najvisji)                                    (najnizji) \n");
    printf("\n");

    printf("UJEMANJE VRSTNEGA REGA\n");
    printf("\tUjemanje sekvenčno - OpenMP: %u %%,\n", 100 * eq_omp(ranked_seq, ranked_omp) / pages.nnodes);
}