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

    Graph pages;

    cout << "berem ...\n"; flush(cout);
    {
        TIMER("")
        pages.read(filename);
    }

    // cout << "gradim strukturo za OpenCL... "; flush(cout);
    // Graph4CL pages4cl(pages);
    // cout << "zgrajeno.\n\n";

    cout << "Število strani  : " << pages.nnodes << '\n';
    cout << "Število povezav : " << pages.nedges << '\n';
    cout << "Največji id     : " << pages.max_id << '\n';
    cout << std::endl;

    float sum_seq = 0, sum_omp = 0, sum_ocl = 0;

    double seq_time = omp_get_wtime();
    pages.rank();
    seq_time = omp_get_wtime() - seq_time;
    cout << "seq time: " << seq_time << " s" << endl;

    // {
    //     TIMER("zaporedno : ")
    //     pages.rank();
    // }

    // seštej range strani
    for (const auto &[id, node] : pages.nodes) {
        sum_seq += node.rank;
    }

    double omp_time = omp_get_wtime();
    pages.rank_omp();
    omp_time = omp_get_wtime() - omp_time;
    cout << "omp time: " << omp_time << " s" << endl;

    // {
    //     TIMER("OpenMP    : ")
    //     pages.rank_omp();
    // }

    // seštej range strani
    for (const auto &[id, node] : pages.nodes) {
        sum_omp += node.rank;
    }

    // double opencl_time = omp_get_wtime();
    // Graph4CL_rank(&pages4cl);
    // opencl_time = omp_get_wtime() - opencl_time;

    // {
    //     TIMER("OpenCL    : ")
    //     Graph4CL_rank(&pages4cl);
    // }

    cout << '\n';

    // seštej range strani
    // for (int i = 0; i < pages4cl.nnodes; i++) {
    //     int32_t id = pages4cl.ids[i];
    //     Node4CL &node = pages4cl.nodes[id];
        
    //     sum_ocl += node.rank;
    // }

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

    // zapisi case in pohitritve v file "time-results.txt"
    // std::ofstream log("time-results.txt", std::ios_base::app | std::ios_base::out);
    // log << std::left << std::setw(16) << std::setfill(' ') << seq_time << "|";
    // log << std::left << std::setw(16) << std::setfill(' ') << omp_time << "|";
    // log << std::left << std::setw(16) << std::setfill(' ') << (seq_time/omp_time) << "|";
    // log << std::left << std::setw(16) << std::setfill(' ') << "-" << "|"; // schedule npr. (dynamic, 10)
    // log << std::left << std::setw(16) << std::setfill(' ') << opencl_time << "|";
    // log << std::left << std::setw(16) << std::setfill(' ') << (seq_time/opencl_time) << "|\n";
}