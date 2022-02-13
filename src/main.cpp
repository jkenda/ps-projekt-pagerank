#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdarg>
#include <omp.h>
#include <float.h>
#include "Graph.hpp"
#include "Graph4CL.hpp"
#include "Timer.hpp"

using namespace std;

bool comp(const Node &a, const Node &b){ return a.rank > b.rank; }
bool comp4cl(const pair<id_t, rank_t> &a, const pair<id_t, rank_t> &b){ return a.second > b.second; }

uint32_t eq_omp(vector<Node> seq, vector<Node> omp)
{
    uint32_t eq = 0;
    for (size_t i = 0; i < seq.size(); i++) {
        if (seq[i].id == omp[i].id) eq++;
    }
    return eq;
}

uint32_t eq_ocl(vector<Node> seq, vector<std::pair<id_t, rank_t>> ocl)
{
    uint32_t eq = 0;
    for (size_t i = 0; i < seq.size(); i++) {
        if (seq[i].id == ocl[i].first) eq++;
    }
    return eq;

}

void find_optimal_wg_size(Graph4CL *pages4cl, uint32_t lower, uint32_t upper, float time_seq, float min)
{
    float time_lower, time_upper, time_mid;
    uint32_t iter_lower, iter_mid, iter_upper;
    float time_min; uint32_t wg_size_min, iter_min;
    
    const uint32_t mid = (lower + upper) / 2;

    {
        TIMER(time_lower)
        iter_lower = Graph4CL_rank(pages4cl, lower);
    }
    {
        TIMER(time_upper)
        iter_mid = Graph4CL_rank(pages4cl, upper);
    }
    {
        TIMER(time_mid)
        iter_upper = Graph4CL_rank(pages4cl, mid);
    }

    if (time_lower < time_mid && time_lower < time_upper) {
        time_min = time_lower;
        wg_size_min = lower;
        iter_min = iter_lower;
    }
    else if (time_mid < time_upper) {
        time_min = time_mid;
        wg_size_min = mid;
        iter_min = iter_mid;
    }
    else {
        time_min = time_upper;
        wg_size_min = upper;
        iter_min = iter_upper;
    }

    if (time_min < min) {
        printf("\t│ %-9s │ %7u │ %9u │ %9.5f │ %10.5f │\n", "OpenCL", wg_size_min, iter_min, time_min, time_seq / time_min);
        flush(cout);
        min = time_min;
    }

    if (upper - lower <= 2) return;

    if (time_lower <= time_upper && time_mid <= time_upper) {
        // lower and mid are the fastest
        find_optimal_wg_size(pages4cl, lower, mid, time_seq, min);
    }
    else if (time_mid <= time_lower && time_upper <= time_lower) {
        // mid and upper are the fastest
        find_optimal_wg_size(pages4cl, mid, upper, time_seq, min);
    }
    else {
        // lower and upper are the fastest
        find_optimal_wg_size(pages4cl, (lower + mid) / 2, (mid + upper) / 2, time_seq, min);
    }

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
    cout << time_read << " s\n"; flush(cout);

    // sestavi graph za OpenCL iz obstoječega grafa
    cout << "\tgradim strukturo za OpenCL... "; flush(cout);
    Graph4CL pages4cl(pages);
    cout << "zgrajeno.\n\n"; flush(cout);

    printf("OSNOVNI PODATKI\n");
    // izpiši osnovne informacije o grafu
    cout << "\tŠtevilo strani    : " << pages.nnodes << ",\n";
    cout << "\tštevilo povezav   : " << pages.nedges << ",\n";
    cout << "\tštevilo ponorov   : " << pages.nsinks << ",\n";
    cout << "\tnajvečji id       : " << pages.max_id << ".\n";
    cout << "\tvelikost podatkov : " << pages4cl.data_size() << " MB.\n";
    cout << '\n';

    rank_t sum_seq = 0; 
    rank_t sum_omp = 0; 
    rank_t sum_ocl = 0;
    float time_seq;
    float time_omp; 
    float time_ocl;
    uint32_t iter_seq; 
    uint32_t iter_omp; 
    uint32_t iter_ocl;

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

    // OpenCL
    find_optimal_wg_size(&pages4cl, 1, 256, time_seq, FLT_MAX);

    printf("\t└───────────┴─────────┴───────────┴───────────┴────────────┘\n\n");

    // seštej range strani
    for (uint32_t i = 0; i < pages4cl.nnodes; i++) {
        sum_ocl += pages4cl.ranks[i];
    }

    // sortiraj range strani po velikosti
    vector<pair<id_t, rank_t>> ranked_ocl;
    for (uint32_t i = 0; i < pages4cl.nnodes; i++) {
        ranked_ocl.emplace_back(make_pair(pages4cl.ids[i], pages4cl.ranks[i]));
    }
    sort(ranked_ocl.begin(), ranked_ocl.end(), comp4cl);

    cleanup(&pages4cl);

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
    printf("\tujemanje sekvenčno - OpenCL: %u %%.\n", 100 * eq_ocl(ranked_seq, ranked_ocl) / pages.nnodes);
    printf("\tujemanje OpenMP    - OpenCL: %u %%.\n", 100 * eq_ocl(ranked_omp, ranked_ocl) / pages.nnodes);
}