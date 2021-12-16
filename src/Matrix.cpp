#include <iostream>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Matrix.hpp"

size_t conn_hash::operator()(const connection& con) const 
{ 
    return con.from ^ con.to; 
}
bool connection::operator==(const connection& other) const 
{ 
    return from == other.from && to == other.to; 
}

Matrix::Matrix(const char *filename)
{
    std::ifstream is(filename, std::ios::in);

    // dobi velikost datoteke
    is.seekg(0, std::ios::end);
    size_t filesize = is.tellg();
    is.seekg(0, std::ios::beg);
    
    // preskoči komentarje in preberi št. povezav
    uint32_t nnodes, nedges;
    std::string line, word;

    std::stringstream words(line);

    do {
        std::getline(is, line, '\n');
        while (words >> word) {
            if (word == "Nodes:") {
                words >> nnodes;
            }
            else if (word == "Edges:") {
                words >> nedges;
            }
        }
    }
    while (line[0] == '#');

    connections.reserve(nnodes);

    uint32_t a, b;

    words >> a; words >> b;
    std::cout << a << ' ' << b << std::endl;
    connections.emplace(a, b);

    // dodaj povezave v množico
    while (is >> a, is >> b) {
        connections.emplace(a, b);
    }
}

bool Matrix::has_connection(const uint32_t a, const uint32_t b) const
{
    return connections.find({a, b}) != connections.end();
}