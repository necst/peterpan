#include <string>
#include <vector>

#include "peterpan_runner.hpp"

int main(int argc, char* argv[]) {
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc) - 1);

    for (int i = 1; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    return peterpan_runner(args);
}