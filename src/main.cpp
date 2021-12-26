#include <iostream>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <iomanip>
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

    // instanciraj graf
    Graph pages;

    // preberi strukturo iz datoteke
    cout << "berem ... "; flush(cout);
    {
        TIMER("")
        pages.read(filename);
    }

    // sestavi graph za OpenCL iz obstoječega graph-a
    cout << "gradim strukturo za OpenCL... "; flush(cout);
    Graph4CL pages4cl(pages);
    cout << "zgrajeno.\n\n";

    // izpiši osnovne informacije o grafu
    cout << "Število strani  : " << pages.nnodes << '\n';
    cout << "Število povezav : " << pages.nedges << '\n';
    cout << "Število ponorov : " << pages.nsinks << '\n';
    cout << "Največji id     : " << pages.max_id << '\n';
    cout << '\n';

    rank_t sum_seq = 0, sum_omp = 0, sum_ocl = 0;
    float time_seq, time_omp, time_ocl;
    uint32_t iter_seq, iter_omp, iter_ocl;

    /* RANGIRAJ STRANI */

    // sekvenčni algoritem
    printf("┌───────────┬───────────┬───────────┐\n");
    printf("│ %-9s │ %-9s │ %-9s │\n", "nacin", "cas [s]", "iteracije");
    printf("├───────────┼───────────┼───────────┤\n");
    
    time_seq = omp_get_wtime();
    iter_seq = pages.rank();
    time_seq = omp_get_wtime() - time_seq;
    printf("│ %-9s │ % 9.5f │ %9d │\n", "zaporedno", time_seq, iter_seq);

    // seštej range strani
    for (const auto &[id, node] : pages.nodes) {
        sum_seq += node.rank;
    }

    // OpenMP

    time_omp = omp_get_wtime();
    iter_omp = pages.rank_omp();
    time_omp = omp_get_wtime() - time_omp;
    printf("│ %-9s │ % 9.5f │ %9d │\n", "OpenMP", time_omp, iter_omp);

    // seštej range strani
    for (const auto &[id, node] : pages.nodes) {
        sum_omp += node.rank;
    }

    // OpenCL

    time_ocl = omp_get_wtime();
    iter_ocl = Graph4CL_rank(&pages4cl);
    time_ocl = omp_get_wtime() - time_ocl;
    printf("│ %-9s │ % 9.5f │ %9d │\n", "OpenCL", time_ocl, iter_ocl);
    printf("└───────────┴───────────┴───────────┘\n");
    printf("\n");

    // seštej range strani
    for (uint32_t i = 0; i < pages4cl.nnodes; i++) {
        int32_t id = pages4cl.ids[i];
        Node4CL &node = pages4cl.nodes[id]; 
        sum_ocl += node.rank;
    }

    // rangi se morajo sešteti v 1
    printf("seštevki rangov:\n");
    printf("┌───────────┬──────────────┐\n");
    printf("│ %-9s │ %-12s │\n", "nacin", "sestevek");
    printf("├───────────┼──────────────┤\n");
    printf("│ %-9s │ %12.10lf │\n", "zaporedno", sum_seq);
    printf("│ %-9s │ %12.10lf │\n", "OpenMP", sum_omp);
    printf("│ %-9s │ %12.10lf │\n", "OpenCL", sum_ocl);
    printf("└───────────┴──────────────┘\n");
    printf("(rang strani je verjetnost - seštevek vseh verjetnosti mora biti 1)\n");
    printf("\n");

    // sortiraj rank-e strani po velikosti
    std::vector<Node> ranked;
    for (auto &[id, node] : pages.nodes) {
        ranked.emplace_back(node);
    }
    std::sort(ranked.begin(), ranked.end(), comp);

    printf("strani z največjim rangom:\n");
    printf("┌──────────┬───────────┐\n");
    printf("│ %-8s │ %-9s │\n", "id", "rank");
    printf("├──────────┼───────────┤\n");
    for (auto it = ranked.begin(); it != ranked.begin()+10; ++it) {
        printf("│ %8d │ %8.3e │\n", it->id, it->rank);
    }
    printf("└──────────┴───────────┘\n");
    printf("\n");

    printf("strani z najmajšim rangom:\n");
    printf("┌──────────┬───────────┐\n");
    printf("│ %-8s │ %-9s │\n", "id", "rank");
    printf("├──────────┼───────────┤\n");
    for (auto it = ranked.end()-10; it != ranked.end(); ++it) {
        printf("│ %8d │ %8.3e │\n", it->id, it->rank);
    }
    printf("└──────────┴───────────┘\n");

    // // zapisi case in pohitritve v file "time-results.txt"
    // std::ofstream log("time-results.txt", std::ios_base::app | std::ios_base::out);
    // log << std::left << std::setw(16) << std::setfill(' ') << seq_time << "|";
    // log << std::left << std::setw(16) << std::setfill(' ') << time_omp << "|";
    // log << std::left << std::setw(16) << std::setfill(' ') << (seq_time/time_omp) << "|";
    // log << std::left << std::setw(16) << std::setfill(' ') << "-" << "|"; // schedule npr. (dynamic, 10)
    // log << std::left << std::setw(16) << std::setfill(' ') << time_ocl << "|";
    // log << std::left << std::setw(16) << std::setfill(' ') << (seq_time/time_ocl) << "|\n";
}