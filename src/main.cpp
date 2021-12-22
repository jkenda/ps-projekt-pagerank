#include <iostream>
#include "Graph.hpp"

int main(int argc, char **argv)
{
    std::cout << "reading file ...\r"; fflush(stdout);
    Graph pages("web-Google.txt");
    std::cout << "file read.        " << std::endl;

    std::cout << "0 -> 11342: " << (pages.has_connection(0, 11342) ? "yes" : "no") << std::endl;
    std::cout << "11342 -> 0: " << (pages.has_connection(11342, 0) ? "yes" : "no") << std::endl;
    std::cout << "11342 -> 1: " << (pages.has_connection(11342, 1) ? "yes" : "no") << std::endl;

    std::cout << pages.nodes[11342].id << ": ";
    for (const Node *l : pages.nodes[11342].links) {
        std::cout << l->id << " ";
    }
    std::cout << std::endl;
    std::cout << pages.max_id << '\n' << pages.max_links << std::endl;
}