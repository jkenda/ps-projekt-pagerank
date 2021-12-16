#pragma once
#include <unordered_set>
#include <cinttypes>

/*
    TABELA POVEZAV
    src     ndst    dst
    0       3       1,3,4
    1       0       
    2       1       4
    3       2       0,1
    4       0       
*/

struct connection
{
    uint32_t from;
    uint32_t to;
    connection(uint32_t from, uint32_t to)
    : from(from), to(to)
    {
    }

    bool operator==(const connection& other) const;
};

struct conn_hash
{
    size_t operator()(const connection& con) const;
};

class Matrix
{
    std::unordered_set<connection, conn_hash> connections;

public:
    Matrix(const char *filename);
    bool has_connection(const uint32_t a, const uint32_t b) const;

private:
};