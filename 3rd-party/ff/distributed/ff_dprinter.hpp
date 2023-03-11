#ifndef FF_DPRINTER_H
#define FF_DPRINTER_H

#include <iostream>
namespace ff {
    class prefixbuf : public std::streambuf
    {
        std::string     prefix;
        std::streambuf* sbuf;
        bool            need_prefix = true;
    
        int sync() { return this->sbuf->pubsync();}
        int overflow(int c) {
            if (c != std::char_traits<char>::eof()) {
                if (this->need_prefix
                    && !this->prefix.empty()
                    && (int)(this->prefix.size()) != this->sbuf->sputn(&this->prefix[0], this->prefix.size())) {
                    return std::char_traits<char>::eof();
                }
                this->need_prefix = c == '\n';
            }
            return this->sbuf->sputc(c);
        }
        
    public:
        prefixbuf(std::string const& p, std::streambuf* sbuf) : prefix(std::string("[" + p + "] ")), sbuf(sbuf) {}
        void setPrefix(std::string const& p){this->prefix = std::string("[" + p + "] ");}
    };
    
    class Printer : private virtual prefixbuf, public std::ostream {
    public:
        Printer(std::string const& p, std::ostream& out) : prefixbuf(p, out.rdbuf()), std::ios(static_cast<std::streambuf*>(this)), std::ostream(static_cast<std::streambuf*>(this)) {}
        void setPrefix(std::string const& p){prefixbuf::setPrefix(p);}
    };
    
    template<class CharT, class Traits>
    auto& endl(std::basic_ostream<CharT, Traits>& os){return std::endl(os);}
    
    Printer cout("undefined", std::cout);
}

#endif
