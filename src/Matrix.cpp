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
    if (!is.is_open()) {
        exit(1);
    }

    uint32_t nnodes, nedges;
    std::string line, word;

    std::stringstream words;
    std::getline(is, line, '\n');

    // preskoči komentarje in preberi št. povezav
    while (line[0] == '#') {
        words.clear();
        words.str(line);

        while (words >> word) {
            if (word == "Nodes:") {
                words >> nnodes;
            }
            else if (word == "Edges:") {
                words >> nedges;
            }
        }

        std::getline(is, line, '\n');
    }

    std::cout << nnodes << std::endl;
    connections.reserve(nnodes);

    uint32_t a, b;

    // dodaj prvo povezavo v množico
    words.clear();
    words.str(line);
    words >> a; words >> b;
    connections.emplace(a, b);

    // največji id vozlišča (velikost tabele)
    uint32_t max_id = std::max(a, b);

    // dodaj povezave v množico
    while (is >> a, is >> b) {
        connections.emplace(a, b);
        max_id = std::max(max_id, a);
        max_id = std::max(max_id, b);
    }
}

bool Matrix::has_connection(const uint32_t a, const uint32_t b) const
{
    return connections.find({a, b}) != connections.end();
}