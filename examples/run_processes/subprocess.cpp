#include "subprocess.hpp"
#include <iostream>
#include <boost/asio.hpp>
#include <boost/process.hpp>
#include <boost/asio.hpp>
#include <boost/process.hpp>

#include <boost/process.hpp>
#include <boost/process.hpp>

namespace bp = boost::process;


struct CommandAndDir
{
    std::string commandLine;
    std::filesystem::path directory;
};



class SubProcess
{
public:
    SubProcess( const CommandAndDir &command)
        : m_command(command)
        , bufErr(1000)
        , bufOut(1000)
        , apErr(m_ioc)
        , apOut(m_ioc)
        , apIn(m_ioc)
    {
    }

    void run();
    int wait();

private:
    void readStdOut();
    void readStdErr();

    const CommandAndDir &m_command;
    std::vector<char> bufErr;
    std::vector<char> bufOut;

    boost::asio::io_context m_ioc;
    boost::process::async_pipe apErr;
    boost::process::async_pipe apOut;
    boost::process::async_pipe apIn;
    std::unique_ptr<boost::process::child> m_child;
};

void SubProcess::run()
{
    if(m_command.directory == std::filesystem::path{})
    {
        m_child = std::make_unique<bp::child>(m_command.commandLine
                                              , bp::std_err > apErr
                                              , bp::std_out > apOut);
    } else {
        m_child = std::make_unique<bp::child>(m_command.commandLine
                                              , bp::std_err > apErr
                                              , bp::std_out > apOut
                                              , bp::start_dir(m_command.directory.string()));
    }

    readStdOut();
    readStdErr();
}

int SubProcess::wait()
{
    m_ioc.run();
    if(m_child) {
        m_child->wait();
        return m_child->exit_code();
    }
    return EXIT_FAILURE;
}

void SubProcess::readStdOut()
{
    apOut.async_read_some(boost::asio::buffer(bufOut),
                          [this](const boost::system::error_code &ec, std::size_t size){
                              std::cout << std::string{bufOut.cbegin(), bufOut.cbegin() + size} ;
                              if(size != 0) {
                                  readStdOut();
                              }
                          });

}

void SubProcess::readStdErr()
{
    apErr.async_read_some(boost::asio::buffer(bufErr),
                          [this](const boost::system::error_code &ec, std::size_t size){
                              std::cerr << std::string{bufErr.cbegin(), bufErr.cbegin() + size} ;
                              if(size != 0) {
                                  readStdErr();
                              }
                          });
}

int runBlocking(const CommandAndDir &command)
{
    SubProcess subProcess{command};
    subProcess.run();
    const int exitResult = subProcess.wait();
    return exitResult;
}

int runBlocking(const std::string &executable)
{
    CommandAndDir command{executable};
    return runBlocking(command);
}

int runBlocking(const std::string &command, const std::filesystem::path &directory)
{
    CommandAndDir commandAndDir{command, directory};
    return runBlocking(command);
}
