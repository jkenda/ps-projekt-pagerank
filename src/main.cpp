#include <iostream>
#include "Matrix.hpp"

int main(int argc, char **argv)
{
    std::cout << "reading file ...\r"; fflush(stdout);
    Matrix pages("web-Google.txt");
    std::cout << "file read.        " << std::endl;

    std::cout << "0 -> 11342: " << (pages.has_connection(0, 11342) ? "yes" : "no") << std::endl;
    std::cout << "11342 -> 0: " << (pages.has_connection(11342, 0) ? "yes" : "no") << std::endl;
    std::cout << "11342 -> 1: " << (pages.has_connection(11342, 1) ? "yes" : "no") << std::endl;
}