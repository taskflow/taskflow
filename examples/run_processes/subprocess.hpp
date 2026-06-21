#pragma once

#include <filesystem>
#include <string>


// Run an executable as a child process, block till process is completed.
// Write output of process to stdout/stderr.
// Return exit code of process.
int runBlocking(const std::string &command);
int runBlocking(const std::string &command, const std::filesystem::path &directory);


