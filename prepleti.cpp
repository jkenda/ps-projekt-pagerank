#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

struct Link
{
    uint32_t src, dst;
};

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3) return 1;

    ifstream is(argv[1], ios::in);
    if (!is.is_open()) {
        exit(1);
    }

    string line, word;
    uint32_t nnodes;

    stringstream words;
    getline(is, line, '\n');

    // preskoči komentarje in preberi št. povezav
    while (line[0] == '#') {
        words.clear();
        words.str(line);

        while (words >> word) {
            if (word == "Nodes:") {
                words >> nnodes;
            }
        }

        getline(is, line, '\n');
    }
    words.clear();
    words.str(line);

    vector<Link> links;

    uint32_t l_id, r_id;
    words >> l_id; words >> r_id;

    // začetni največji id
    uint32_t max_id = max(l_id, r_id);

    links.push_back({ l_id, r_id });

    while (is >> l_id, is >> r_id) {
        links.push_back({ l_id, r_id });
        max_id = max(max_id, max(l_id, r_id));
    }

    ofstream os(argv[2], ios::out | ios::trunc);

    os << "# " << "Nodes: " << nnodes * 3 << '\n';

    for (Link l : links) {
        os << l.src << '\t' << l.dst << '\n';
    }

    for (Link l : links) {
        os << max_id + l.dst + 1 << '\t' << l.src << '\n';
    }

    for (Link l : links) {
        os << max_id + l.src + 1 << '\t' << max_id + l.dst << '\n';
    }

}