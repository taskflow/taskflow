#if defined(DOCTEST_CONFIG_IMPLEMENT) || !defined(DOCTEST_SINGLE_HEADER)
#ifndef DOCTEST_LIBRARY_IMPLEMENTATION
#define DOCTEST_LIBRARY_IMPLEMENTATION

#ifndef DOCTEST_SINGLE_HEADER
#include "doctest_fwd.h"
#endif // DOCTEST_SINGLE_HEADER

DOCTEST_CLANG_SUPPRESS_WARNING_PUSH
DOCTEST_CLANG_SUPPRESS_WARNING("-Wunknown-pragmas")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wpadded")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wglobal-constructors")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wexit-time-destructors")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wmissing-prototypes")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wsign-conversion")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wshorten-64-to-32")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wmissing-variable-declarations")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wswitch")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wswitch-enum")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wcovered-switch-default")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wmissing-noreturn")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wunused-local-typedef")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wdisabled-macro-expansion")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wmissing-braces")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wmissing-field-initializers")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wc++11-long-long")
#if DOCTEST_CLANG && DOCTEST_CLANG_HAS_WARNING("-Wzero-as-null-pointer-constant")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wzero-as-null-pointer-constant")
#endif // clang - 0 as null

DOCTEST_GCC_SUPPRESS_WARNING_PUSH
DOCTEST_GCC_SUPPRESS_WARNING("-Wunknown-pragmas")
DOCTEST_GCC_SUPPRESS_WARNING("-Wconversion")
DOCTEST_GCC_SUPPRESS_WARNING("-Weffc++")
DOCTEST_GCC_SUPPRESS_WARNING("-Wsign-conversion")
DOCTEST_GCC_SUPPRESS_WARNING("-Wstrict-overflow")
DOCTEST_GCC_SUPPRESS_WARNING("-Wmissing-field-initializers")
DOCTEST_GCC_SUPPRESS_WARNING("-Wmissing-braces")
DOCTEST_GCC_SUPPRESS_WARNING("-Wmissing-declarations")
DOCTEST_GCC_SUPPRESS_WARNING("-Winline")
DOCTEST_GCC_SUPPRESS_WARNING("-Wswitch")
DOCTEST_GCC_SUPPRESS_WARNING("-Wswitch-enum")
DOCTEST_GCC_SUPPRESS_WARNING("-Wswitch-default")
DOCTEST_GCC_SUPPRESS_WARNING("-Wunsafe-loop-optimizations")
DOCTEST_GCC_SUPPRESS_WARNING("-Wlong-long")
DOCTEST_GCC_SUPPRESS_WARNING("-Wold-style-cast")
#if DOCTEST_GCC >= DOCTEST_COMPILER(4, 7, 0)
DOCTEST_GCC_SUPPRESS_WARNING("-Wzero-as-null-pointer-constant")
#endif // GCC 4.7+
#if DOCTEST_GCC >= DOCTEST_COMPILER(4, 8, 0)
DOCTEST_GCC_SUPPRESS_WARNING("-Wunused-local-typedefs")
#endif // GCC 4.8+
#if DOCTEST_GCC >= DOCTEST_COMPILER(5, 4, 0)
DOCTEST_GCC_SUPPRESS_WARNING("-Wuseless-cast")
#endif // GCC 5.4+

DOCTEST_MSVC_SUPPRESS_WARNING_PUSH
DOCTEST_MSVC_SUPPRESS_WARNING(4616) // invalid compiler warning
DOCTEST_MSVC_SUPPRESS_WARNING(4619) // invalid compiler warning
DOCTEST_MSVC_SUPPRESS_WARNING(4996) // The compiler encountered a deprecated declaration
DOCTEST_MSVC_SUPPRESS_WARNING(4267) // 'var' : conversion from 'x' to 'y', possible loss of data
DOCTEST_MSVC_SUPPRESS_WARNING(4706) // assignment within conditional expression
DOCTEST_MSVC_SUPPRESS_WARNING(4512) // 'class' : assignment operator could not be generated
DOCTEST_MSVC_SUPPRESS_WARNING(4127) // conditional expression is constant
DOCTEST_MSVC_SUPPRESS_WARNING(4530) // C++ exception handler used, but unwind semantics not enabled
DOCTEST_MSVC_SUPPRESS_WARNING(4577) // 'noexcept' used with no exception handling mode specified
DOCTEST_MSVC_SUPPRESS_WARNING(4774) // format string expected in argument is not a string literal
DOCTEST_MSVC_SUPPRESS_WARNING(4365) // conversion from 'int' to 'unsigned', signed/unsigned mismatch
DOCTEST_MSVC_SUPPRESS_WARNING(4820) // padding in structs
DOCTEST_MSVC_SUPPRESS_WARNING(4640) // construction of local static object is not thread-safe
DOCTEST_MSVC_SUPPRESS_WARNING(5039) // pointer to potentially throwing function passed to extern C

#if defined(DOCTEST_NO_CPP11_COMPAT)
DOCTEST_CLANG_SUPPRESS_WARNING("-Wc++98-compat")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wc++98-compat-pedantic")
#endif // DOCTEST_NO_CPP11_COMPAT

// snprintf() not in the C++98 standard
#if DOCTEST_MSVC
#define DOCTEST_SNPRINTF _snprintf
#else // MSVC
#define DOCTEST_SNPRINTF std::snprintf
#endif // MSVC

#define DOCTEST_LOG_START()                                                                        \
    do {                                                                                           \
        if(!contextState->hasLoggedCurrentTestStart) {                                             \
            logTestStart(*contextState->currentTest);                                              \
            contextState->hasLoggedCurrentTestStart = true;                                        \
        }                                                                                          \
    } while(false)

DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_BEGIN

// required includes - will go only in one translation unit!
#include <ctime>
#include <cmath>
// borland (Embarcadero) compiler requires math.h and not cmath - https://github.com/onqtam/doctest/pull/37
#ifdef __BORLANDC__
#include <math.h>
#endif // __BORLANDC__
#include <new>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <set>
#include <exception>
#include <stdexcept>
#include <csignal>
#include <cfloat>
#if !DOCTEST_MSVC
#include <stdint.h>
#endif // !MSVC

DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_END

namespace doctest
{
namespace detail
{
    // lowers ascii letters
    char tolower(const char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c; }

    template <typename T>
    T my_max(const T& lhs, const T& rhs) {
        return lhs > rhs ? lhs : rhs;
    }

    // case insensitive strcmp
    int stricmp(char const* a, char const* b) {
        for(;; a++, b++) {
            const int d = tolower(*a) - tolower(*b);
            if(d != 0 || !*a)
                return d;
        }
    }

    void my_memcpy(void* dest, const void* src, unsigned num) {
        const char* csrc  = static_cast<const char*>(src);
        char*       cdest = static_cast<char*>(dest);
        for(unsigned i = 0; i < num; ++i)
            cdest[i] = csrc[i];
    }

    // not using std::strlen() because of valgrind errors when optimizations are turned on
    // 'Invalid read of size 4' when the test suite len (with '\0') is not a multiple of 4
    // for details see http://stackoverflow.com/questions/35671155
    unsigned my_strlen(const char* in) {
        const char* temp = in;
        while(temp && *temp)
            ++temp;
        return unsigned(temp - in);
    }

    template <typename T>
    String fpToString(T value, int precision) {
        std::ostringstream oss;
        oss << std::setprecision(precision) << std::fixed << value;
        std::string d = oss.str();
        size_t      i = d.find_last_not_of('0');
        if(i != std::string::npos && i != d.size() - 1) {
            if(d[i] == '.')
                i++;
            d = d.substr(0, i + 1);
        }
        return d.c_str();
    }

    struct Endianness
    {
        enum Arch
        {
            Big,
            Little
        };

        static Arch which() {
            union _
            {
                int  asInt;
                char asChar[sizeof(int)];
            } u;

            u.asInt = 1;                                            // NOLINT
            return (u.asChar[sizeof(int) - 1] == 1) ? Big : Little; // NOLINT
        }
    };

    String rawMemoryToString(const void* object, unsigned size) {
        // Reverse order for little endian architectures
        int i = 0, end = static_cast<int>(size), inc = 1;
        if(Endianness::which() == Endianness::Little) {
            i   = end - 1;
            end = inc = -1;
        }

        unsigned char const* bytes = static_cast<unsigned char const*>(object);
        std::ostringstream   os;
        os << "0x" << std::setfill('0') << std::hex;
        for(; i != end; i += inc)
            os << std::setw(2) << static_cast<unsigned>(bytes[i]);
        return os.str().c_str();
    }

    std::ostream* createStream() { return new std::ostringstream(); }
    String        getStreamResult(std::ostream* in) {
        return static_cast<std::ostringstream*>(in)->str().c_str(); // NOLINT
    }
    void freeStream(std::ostream* in) { delete in; }

#ifndef DOCTEST_CONFIG_DISABLE

    // this holds both parameters for the command line and runtime data for tests
    struct ContextState : TestAccessibleContextState //!OCLINT too many fields
    {
        // == parameters from the command line

        std::vector<std::vector<String> > filters;

        String   order_by;  // how tests should be ordered
        unsigned rand_seed; // the seed for rand ordering

        unsigned first; // the first (matching) test to be executed
        unsigned last;  // the last (matching) test to be executed

        int  abort_after;           // stop tests after this many failed assertions
        int  subcase_filter_levels; // apply the subcase filters for the first N levels
        bool case_sensitive;        // if filtering should be case sensitive
        bool exit;         // if the program should be exited after the tests are ran/whatever
        bool duration;     // print the time duration of each test case
        bool no_exitcode;  // if the framework should return 0 as the exitcode
        bool no_run;       // to not run the tests at all (can be done with an "*" exclude)
        bool no_version;   // to not print the version of the framework
        bool no_colors;    // if output to the console should be colorized
        bool force_colors; // forces the use of colors even when a tty cannot be detected
        bool no_breaks;    // to not break into the debugger
        bool no_skip;      // don't skip test cases which are marked to be skipped
        bool no_path_in_filenames; // if the path to files should be removed from the output
        bool no_line_numbers;      // if source code line numbers should be omitted from the output
        bool no_skipped_summary;   // don't print "skipped" in the summary !!! UNDOCUMENTED !!!

        bool help;             // to print the help
        bool version;          // to print the version
        bool count;            // if only the count of matching tests is to be retreived
        bool list_test_cases;  // to list all tests matching the filters
        bool list_test_suites; // to list all suites matching the filters

        // == data for the tests being ran

        unsigned        numTestsPassingFilters;
        unsigned        numTestSuitesPassingFilters;
        unsigned        numFailed;
        const TestCase* currentTest;
        bool            hasLoggedCurrentTestStart;
        int             numAssertionsForCurrentTestcase;
        int             numAssertions;
        int             numFailedAssertionsForCurrentTestcase;
        int             numFailedAssertions;
        bool            hasCurrentTestFailed;

        std::vector<IContextScope*> contexts;            // for logging with INFO() and friends
        std::vector<std::string>    exceptionalContexts; // logging from INFO() due to an exception

        // stuff for subcases
        std::set<SubcaseSignature> subcasesPassed;
        std::set<int>              subcasesEnteredLevels;
        std::vector<Subcase>       subcasesStack;
        int                        subcasesCurrentLevel;
        bool                       subcasesHasSkipped;

        void resetRunData() {
            numTestsPassingFilters                = 0;
            numTestSuitesPassingFilters           = 0;
            numFailed                             = 0;
            numAssertions                         = 0;
            numFailedAssertions                   = 0;
            numFailedAssertionsForCurrentTestcase = 0;
        }

        // cppcheck-suppress uninitMemberVar
        ContextState()
                : filters(8) // 8 different filters total
        {
            resetRunData();
        }
    };

    ContextState* contextState = 0;
#endif // DOCTEST_CONFIG_DISABLE
} // namespace detail

void String::copy(const String& other) {
    if(other.isOnStack()) {
        detail::my_memcpy(buf, other.buf, len);
    } else {
        setOnHeap();
        data.size     = other.data.size;
        data.capacity = data.size + 1;
        data.ptr      = new char[data.capacity];
        detail::my_memcpy(data.ptr, other.data.ptr, data.size + 1);
    }
}

String::String(const char* in) {
    unsigned in_len = detail::my_strlen(in);
    if(in_len <= last) {
        detail::my_memcpy(buf, in, in_len + 1);
        setLast(last - in_len);
    } else {
        setOnHeap();
        data.size     = in_len;
        data.capacity = data.size + 1;
        data.ptr      = new char[data.capacity];
        detail::my_memcpy(data.ptr, in, in_len + 1);
    }
}

String& String::operator+=(const String& other) {
    const unsigned my_old_size = size();
    const unsigned other_size  = other.size();
    const unsigned total_size  = my_old_size + other_size;
    if(isOnStack()) {
        if(total_size < len) {
            // append to the current stack space
            detail::my_memcpy(buf + my_old_size, other.c_str(), other_size + 1);
            setLast(last - total_size);
        } else {
            // alloc new chunk
            char* temp = new char[total_size + 1];
            // copy current data to new location before writing in the union
            detail::my_memcpy(temp, buf, my_old_size); // skip the +1 ('\0') for speed
            // update data in union
            setOnHeap();
            data.size     = total_size;
            data.capacity = data.size + 1;
            data.ptr      = temp;
            // transfer the rest of the data
            detail::my_memcpy(data.ptr + my_old_size, other.c_str(), other_size + 1);
        }
    } else {
        if(data.capacity > total_size) {
            // append to the current heap block
            data.size = total_size;
            detail::my_memcpy(data.ptr + my_old_size, other.c_str(), other_size + 1);
        } else {
            // resize
            data.capacity *= 2;
            if(data.capacity <= total_size)
                data.capacity = total_size + 1;
            // alloc new chunk
            char* temp = new char[data.capacity];
            // copy current data to new location before releasing it
            detail::my_memcpy(temp, data.ptr, my_old_size); // skip the +1 ('\0') for speed
            // release old chunk
            delete[] data.ptr;
            // update the rest of the union members
            data.size = total_size;
            data.ptr  = temp;
            // transfer the rest of the data
            detail::my_memcpy(data.ptr + my_old_size, other.c_str(), other_size + 1);
        }
    }

    return *this;
}

#ifdef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
String::String(String&& other) {
    detail::my_memcpy(buf, other.buf, len);
    other.buf[0] = '\0';
    other.setLast();
}

String& String::operator=(String&& other) {
    if(this != &other) {
        if(!isOnStack())
            delete[] data.ptr;
        detail::my_memcpy(buf, other.buf, len);
        other.buf[0] = '\0';
        other.setLast();
    }
    return *this;
}
#endif // DOCTEST_CONFIG_WITH_RVALUE_REFERENCES

int String::compare(const char* other, bool no_case) const {
    if(no_case)
        return detail::stricmp(c_str(), other);
    return std::strcmp(c_str(), other);
}

int String::compare(const String& other, bool no_case) const {
    return compare(other.c_str(), no_case);
}

std::ostream& operator<<(std::ostream& stream, const String& in) {
    stream << in.c_str();
    return stream;
}

Approx::Approx(double value)
        : m_epsilon(static_cast<double>(std::numeric_limits<float>::epsilon()) * 100)
        , m_scale(1.0)
        , m_value(value) {}

bool operator==(double lhs, Approx const& rhs) {
    // Thanks to Richard Harris for his help refining this formula
    return std::fabs(lhs - rhs.m_value) <
           rhs.m_epsilon * (rhs.m_scale + detail::my_max(std::fabs(lhs), std::fabs(rhs.m_value)));
}

String Approx::toString() const { return String("Approx( ") + doctest::toString(m_value) + " )"; }

#ifdef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
String toString(char* in) { return toString(static_cast<const char*>(in)); }
String toString(const char* in) { return String("\"") + (in ? in : "{null string}") + "\""; }
#endif // DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
String toString(bool in) { return in ? "true" : "false"; }
String toString(float in) { return detail::fpToString(in, 5) + "f"; }
String toString(double in) { return detail::fpToString(in, 10); }
String toString(double long in) { return detail::fpToString(in, 15); }

String toString(char in) {
    char buf[64];
    std::sprintf(buf, "%d", in);
    return buf;
}

String toString(char signed in) {
    char buf[64];
    std::sprintf(buf, "%d", in);
    return buf;
}

String toString(char unsigned in) {
    char buf[64];
    std::sprintf(buf, "%ud", in);
    return buf;
}

String toString(int short in) {
    char buf[64];
    std::sprintf(buf, "%d", in);
    return buf;
}

String toString(int short unsigned in) {
    char buf[64];
    std::sprintf(buf, "%u", in);
    return buf;
}

String toString(int in) {
    char buf[64];
    std::sprintf(buf, "%d", in);
    return buf;
}

String toString(int unsigned in) {
    char buf[64];
    std::sprintf(buf, "%u", in);
    return buf;
}

String toString(int long in) {
    char buf[64];
    std::sprintf(buf, "%ld", in);
    return buf;
}

String toString(int long unsigned in) {
    char buf[64];
    std::sprintf(buf, "%lu", in);
    return buf;
}

#ifdef DOCTEST_CONFIG_WITH_LONG_LONG
String toString(int long long in) {
    char buf[64];
    std::sprintf(buf, "%lld", in);
    return buf;
}
String toString(int long long unsigned in) {
    char buf[64];
    std::sprintf(buf, "%llu", in);
    return buf;
}
#endif // DOCTEST_CONFIG_WITH_LONG_LONG

#ifdef DOCTEST_CONFIG_WITH_NULLPTR
String toString(std::nullptr_t) { return "nullptr"; }
#endif // DOCTEST_CONFIG_WITH_NULLPTR

} // namespace doctest

#ifdef DOCTEST_CONFIG_DISABLE
namespace doctest
{
bool isRunningInTest() { return false; }
Context::Context(int, const char* const*) {}
Context::~Context() {}
void Context::applyCommandLine(int, const char* const*) {}
void Context::addFilter(const char*, const char*) {}
void Context::clearFilters() {}
void Context::setOption(const char*, int) {}
void Context::setOption(const char*, const char*) {}
bool Context::shouldExit() { return false; }
int  Context::run() { return 0; }
} // namespace doctest
#else // DOCTEST_CONFIG_DISABLE

#if !defined(DOCTEST_CONFIG_COLORS_NONE)
#if !defined(DOCTEST_CONFIG_COLORS_WINDOWS) && !defined(DOCTEST_CONFIG_COLORS_ANSI)
#ifdef DOCTEST_PLATFORM_WINDOWS
#define DOCTEST_CONFIG_COLORS_WINDOWS
#else // linux
#define DOCTEST_CONFIG_COLORS_ANSI
#endif // platform
#endif // DOCTEST_CONFIG_COLORS_WINDOWS && DOCTEST_CONFIG_COLORS_ANSI
#endif // DOCTEST_CONFIG_COLORS_NONE

#define DOCTEST_PRINTF_COLORED(buffer, color)                                                      \
    do {                                                                                           \
        Color col(color);                                                                          \
        std::printf("%s", buffer);                                                                 \
    } while((void)0, 0)

// the buffer size used for snprintf() calls
#if !defined(DOCTEST_SNPRINTF_BUFFER_LENGTH)
#define DOCTEST_SNPRINTF_BUFFER_LENGTH 1024
#endif // DOCTEST_SNPRINTF_BUFFER_LENGTH

#if DOCTEST_MSVC || defined(__MINGW32__)
#if DOCTEST_MSVC >= DOCTEST_COMPILER(17, 0, 0)
#define DOCTEST_WINDOWS_SAL_IN_OPT _In_opt_
#else // MSVC
#define DOCTEST_WINDOWS_SAL_IN_OPT
#endif // MSVC
extern "C" __declspec(dllimport) void __stdcall OutputDebugStringA(
        DOCTEST_WINDOWS_SAL_IN_OPT const char*);
extern "C" __declspec(dllimport) int __stdcall IsDebuggerPresent();
#endif // MSVC || __MINGW32__

#ifdef DOCTEST_CONFIG_COLORS_ANSI
#include <unistd.h>
#endif // DOCTEST_CONFIG_COLORS_ANSI

#ifdef DOCTEST_PLATFORM_WINDOWS

// defines for a leaner windows.h
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif // WIN32_LEAN_AND_MEAN
#ifndef VC_EXTRA_LEAN
#define VC_EXTRA_LEAN
#endif // VC_EXTRA_LEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX

DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_BEGIN

// not sure what AfxWin.h is for - here I do what Catch does
#ifdef __AFXDLL
#include <AfxWin.h>
#else
#include <windows.h>
#endif
#include <io.h>

DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_END

#else // DOCTEST_PLATFORM_WINDOWS

#include <sys/time.h>

#endif // DOCTEST_PLATFORM_WINDOWS

namespace doctest_detail_test_suite_ns
{
// holds the current test suite
doctest::detail::TestSuite& getCurrentTestSuite() {
    static doctest::detail::TestSuite data;
    return data;
}
} // namespace doctest_detail_test_suite_ns

namespace doctest
{
namespace detail
{
    TestCase::TestCase(funcType test, const char* file, unsigned line, const TestSuite& test_suite,
                       const char* type, int template_id)
            : m_test(test)
            , m_name(0)
            , m_type(type)
            , m_test_suite(test_suite.m_test_suite)
            , m_description(test_suite.m_description)
            , m_skip(test_suite.m_skip)
            , m_may_fail(test_suite.m_may_fail)
            , m_should_fail(test_suite.m_should_fail)
            , m_expected_failures(test_suite.m_expected_failures)
            , m_timeout(test_suite.m_timeout)
            , m_file(file)
            , m_line(line)
            , m_template_id(template_id) {}

    TestCase& TestCase::operator*(const char* in) {
        m_name = in;
        // make a new name with an appended type for templated test case
        if(m_template_id != -1) {
            m_full_name = String(m_name) + m_type;
            // redirect the name to point to the newly constructed full name
            m_name = m_full_name.c_str();
        }
        return *this;
    }

    TestCase& TestCase::operator=(const TestCase& other) {
        m_test              = other.m_test;
        m_full_name         = other.m_full_name;
        m_name              = other.m_name;
        m_type              = other.m_type;
        m_test_suite        = other.m_test_suite;
        m_description       = other.m_description;
        m_skip              = other.m_skip;
        m_may_fail          = other.m_may_fail;
        m_should_fail       = other.m_should_fail;
        m_expected_failures = other.m_expected_failures;
        m_timeout           = other.m_timeout;
        m_file              = other.m_file;
        m_line              = other.m_line;
        m_template_id       = other.m_template_id;

        if(m_template_id != -1)
            m_name = m_full_name.c_str();
        return *this;
    }

    bool TestCase::operator<(const TestCase& other) const {
        if(m_line != other.m_line)
            return m_line < other.m_line;
        const int file_cmp = std::strcmp(m_file, other.m_file);
        if(file_cmp != 0)
            return file_cmp < 0;
        return m_template_id < other.m_template_id;
    }

    const char* getAssertString(assertType::Enum val) {
        DOCTEST_MSVC_SUPPRESS_WARNING_WITH_PUSH(
                4062) // enumerator 'x' in switch of enum 'y' is not handled
        switch(val) { //!OCLINT missing default in switch statements
            // clang-format off
            case assertType::DT_WARN                    : return "WARN";
            case assertType::DT_CHECK                   : return "CHECK";
            case assertType::DT_REQUIRE                 : return "REQUIRE";

            case assertType::DT_WARN_FALSE              : return "WARN_FALSE";
            case assertType::DT_CHECK_FALSE             : return "CHECK_FALSE";
            case assertType::DT_REQUIRE_FALSE           : return "REQUIRE_FALSE";

            case assertType::DT_WARN_THROWS             : return "WARN_THROWS";
            case assertType::DT_CHECK_THROWS            : return "CHECK_THROWS";
            case assertType::DT_REQUIRE_THROWS          : return "REQUIRE_THROWS";

            case assertType::DT_WARN_THROWS_AS          : return "WARN_THROWS_AS";
            case assertType::DT_CHECK_THROWS_AS         : return "CHECK_THROWS_AS";
            case assertType::DT_REQUIRE_THROWS_AS       : return "REQUIRE_THROWS_AS";

            case assertType::DT_WARN_NOTHROW            : return "WARN_NOTHROW";
            case assertType::DT_CHECK_NOTHROW           : return "CHECK_NOTHROW";
            case assertType::DT_REQUIRE_NOTHROW         : return "REQUIRE_NOTHROW";

            case assertType::DT_WARN_EQ                 : return "WARN_EQ";
            case assertType::DT_CHECK_EQ                : return "CHECK_EQ";
            case assertType::DT_REQUIRE_EQ              : return "REQUIRE_EQ";
            case assertType::DT_WARN_NE                 : return "WARN_NE";
            case assertType::DT_CHECK_NE                : return "CHECK_NE";
            case assertType::DT_REQUIRE_NE              : return "REQUIRE_NE";
            case assertType::DT_WARN_GT                 : return "WARN_GT";
            case assertType::DT_CHECK_GT                : return "CHECK_GT";
            case assertType::DT_REQUIRE_GT              : return "REQUIRE_GT";
            case assertType::DT_WARN_LT                 : return "WARN_LT";
            case assertType::DT_CHECK_LT                : return "CHECK_LT";
            case assertType::DT_REQUIRE_LT              : return "REQUIRE_LT";
            case assertType::DT_WARN_GE                 : return "WARN_GE";
            case assertType::DT_CHECK_GE                : return "CHECK_GE";
            case assertType::DT_REQUIRE_GE              : return "REQUIRE_GE";
            case assertType::DT_WARN_LE                 : return "WARN_LE";
            case assertType::DT_CHECK_LE                : return "CHECK_LE";
            case assertType::DT_REQUIRE_LE              : return "REQUIRE_LE";

            case assertType::DT_WARN_UNARY              : return "WARN_UNARY";
            case assertType::DT_CHECK_UNARY             : return "CHECK_UNARY";
            case assertType::DT_REQUIRE_UNARY           : return "REQUIRE_UNARY";
            case assertType::DT_WARN_UNARY_FALSE        : return "WARN_UNARY_FALSE";
            case assertType::DT_CHECK_UNARY_FALSE       : return "CHECK_UNARY_FALSE";
            case assertType::DT_REQUIRE_UNARY_FALSE     : return "REQUIRE_UNARY_FALSE";

            case assertType::DT_FAST_WARN_EQ            : return "FAST_WARN_EQ";
            case assertType::DT_FAST_CHECK_EQ           : return "FAST_CHECK_EQ";
            case assertType::DT_FAST_REQUIRE_EQ         : return "FAST_REQUIRE_EQ";
            case assertType::DT_FAST_WARN_NE            : return "FAST_WARN_NE";
            case assertType::DT_FAST_CHECK_NE           : return "FAST_CHECK_NE";
            case assertType::DT_FAST_REQUIRE_NE         : return "FAST_REQUIRE_NE";
            case assertType::DT_FAST_WARN_GT            : return "FAST_WARN_GT";
            case assertType::DT_FAST_CHECK_GT           : return "FAST_CHECK_GT";
            case assertType::DT_FAST_REQUIRE_GT         : return "FAST_REQUIRE_GT";
            case assertType::DT_FAST_WARN_LT            : return "FAST_WARN_LT";
            case assertType::DT_FAST_CHECK_LT           : return "FAST_CHECK_LT";
            case assertType::DT_FAST_REQUIRE_LT         : return "FAST_REQUIRE_LT";
            case assertType::DT_FAST_WARN_GE            : return "FAST_WARN_GE";
            case assertType::DT_FAST_CHECK_GE           : return "FAST_CHECK_GE";
            case assertType::DT_FAST_REQUIRE_GE         : return "FAST_REQUIRE_GE";
            case assertType::DT_FAST_WARN_LE            : return "FAST_WARN_LE";
            case assertType::DT_FAST_CHECK_LE           : return "FAST_CHECK_LE";
            case assertType::DT_FAST_REQUIRE_LE         : return "FAST_REQUIRE_LE";

            case assertType::DT_FAST_WARN_UNARY         : return "FAST_WARN_UNARY";
            case assertType::DT_FAST_CHECK_UNARY        : return "FAST_CHECK_UNARY";
            case assertType::DT_FAST_REQUIRE_UNARY      : return "FAST_REQUIRE_UNARY";
            case assertType::DT_FAST_WARN_UNARY_FALSE   : return "FAST_WARN_UNARY_FALSE";
            case assertType::DT_FAST_CHECK_UNARY_FALSE  : return "FAST_CHECK_UNARY_FALSE";
            case assertType::DT_FAST_REQUIRE_UNARY_FALSE: return "FAST_REQUIRE_UNARY_FALSE";
                // clang-format on
        }
        DOCTEST_MSVC_SUPPRESS_WARNING_POP
        return "";
    }

    bool checkIfShouldThrow(assertType::Enum assert_type) {
        if(assert_type & assertType::is_require) //!OCLINT bitwise operator in conditional
            return true;

        if((assert_type & assertType::is_check) //!OCLINT bitwise operator in conditional
           && contextState->abort_after > 0 &&
           contextState->numFailedAssertions >= contextState->abort_after)
            return true;

        return false;
    }
    void fastAssertThrowIfFlagSet(int flags) {
        if(flags & assertAction::shouldthrow) //!OCLINT bitwise operator in conditional
            throwException();
    }
    void throwException() {
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
        throw TestFailureException();
#endif // DOCTEST_CONFIG_NO_EXCEPTIONS
    }

    // matching of a string against a wildcard mask (case sensitivity configurable) taken from
    // http://www.emoticode.net/c/simple-wildcard-string-compare-globbing-function.html
    int wildcmp(const char* str, const char* wild, bool caseSensitive) {
        const char* cp = 0;
        const char* mp = 0;

        // rolled my own tolower() to not include more headers
        while((*str) && (*wild != '*')) {
            if((caseSensitive ? (*wild != *str) : (tolower(*wild) != tolower(*str))) &&
               (*wild != '?')) {
                return 0;
            }
            wild++;
            str++;
        }

        while(*str) {
            if(*wild == '*') {
                if(!*++wild) {
                    return 1;
                }
                mp = wild;
                cp = str + 1;
            } else if((caseSensitive ? (*wild == *str) : (tolower(*wild) == tolower(*str))) ||
                      (*wild == '?')) {
                wild++;
                str++;
            } else {
                wild = mp;   //!OCLINT parameter reassignment
                str  = cp++; //!OCLINT parameter reassignment
            }
        }

        while(*wild == '*') {
            wild++;
        }
        return !*wild;
    }

    //// C string hash function (djb2) - taken from http://www.cse.yorku.ca/~oz/hash.html
    //unsigned hashStr(unsigned const char* str) {
    //    unsigned long hash = 5381;
    //    char          c;
    //    while((c = *str++))
    //        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    //    return hash;
    //}

    // checks if the name matches any of the filters (and can be configured what to do when empty)
    bool matchesAny(const char* name, const std::vector<String>& filters, int matchEmpty,
                    bool caseSensitive) {
        if(filters.empty() && matchEmpty)
            return true;
        for(unsigned i = 0; i < filters.size(); ++i)
            if(wildcmp(name, filters[i].c_str(), caseSensitive))
                return true;
        return false;
    }

#ifdef DOCTEST_PLATFORM_WINDOWS

    typedef unsigned long long UInt64;

    UInt64 getCurrentTicks() {
        static UInt64 hz = 0, hzo = 0;
        if(!hz) {
            QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&hz));
            QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&hzo));
        }
        UInt64 t;
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&t));
        return ((t - hzo) * 1000000) / hz;
    }
#else  // DOCTEST_PLATFORM_WINDOWS

    typedef uint64_t UInt64;

    UInt64 getCurrentTicks() {
        timeval t;
        gettimeofday(&t, 0);
        return static_cast<UInt64>(t.tv_sec) * 1000000 + static_cast<UInt64>(t.tv_usec);
    }
#endif // DOCTEST_PLATFORM_WINDOWS

    class Timer
    {
    public:
        Timer()
                : m_ticks(0) {}
        void         start() { m_ticks = getCurrentTicks(); }
        unsigned int getElapsedMicroseconds() const {
            return static_cast<unsigned int>(getCurrentTicks() - m_ticks);
        }
        unsigned int getElapsedMilliseconds() const {
            return static_cast<unsigned int>(getElapsedMicroseconds() / 1000);
        }
        double getElapsedSeconds() const { return getElapsedMicroseconds() / 1000000.0; }

    private:
        UInt64 m_ticks;
    };

    TestAccessibleContextState* getTestsContextState() { return contextState; }

    bool SubcaseSignature::operator<(const SubcaseSignature& other) const {
        if(m_line != other.m_line)
            return m_line < other.m_line;
        if(std::strcmp(m_file, other.m_file) != 0)
            return std::strcmp(m_file, other.m_file) < 0;
        return std::strcmp(m_name, other.m_name) < 0;
    }

    Subcase::Subcase(const char* name, const char* file, int line)
            : m_signature(name, file, line)
            , m_entered(false) {
        ContextState* s = contextState;

        // if we have already completed it
        if(s->subcasesPassed.count(m_signature) != 0)
            return;

        // check subcase filters
        if(s->subcasesCurrentLevel < s->subcase_filter_levels) {
            if(!matchesAny(m_signature.m_name, s->filters[6], 1, s->case_sensitive))
                return;
            if(matchesAny(m_signature.m_name, s->filters[7], 0, s->case_sensitive))
                return;
        }

        // if a Subcase on the same level has already been entered
        if(s->subcasesEnteredLevels.count(s->subcasesCurrentLevel) != 0) {
            s->subcasesHasSkipped = true;
            return;
        }

        s->subcasesStack.push_back(*this);
        if(s->hasLoggedCurrentTestStart)
            logTestEnd();
        s->hasLoggedCurrentTestStart = false;

        s->subcasesEnteredLevels.insert(s->subcasesCurrentLevel++);
        m_entered = true;
    }

    Subcase::Subcase(const Subcase& other)
            : m_signature(other.m_signature.m_name, other.m_signature.m_file,
                          other.m_signature.m_line)
            , m_entered(other.m_entered) {}

    Subcase::~Subcase() {
        if(m_entered) {
            ContextState* s = contextState;

            s->subcasesCurrentLevel--;
            // only mark the subcase as passed if no subcases have been skipped
            if(s->subcasesHasSkipped == false)
                s->subcasesPassed.insert(m_signature);

            if(!s->subcasesStack.empty())
                s->subcasesStack.pop_back();
            if(s->hasLoggedCurrentTestStart)
                logTestEnd();
            s->hasLoggedCurrentTestStart = false;
        }
    }

    Result::~Result() {}

    Result& Result::operator=(const Result& other) {
        m_passed        = other.m_passed;
        m_decomposition = other.m_decomposition;

        return *this;
    }

    // for sorting tests by file/line
    int fileOrderComparator(const void* a, const void* b) {
        const TestCase* lhs = *static_cast<TestCase* const*>(a);
        const TestCase* rhs = *static_cast<TestCase* const*>(b);
#if DOCTEST_MSVC
        // this is needed because MSVC gives different case for drive letters
        // for __FILE__ when evaluated in a header and a source file
        const int res = stricmp(lhs->m_file, rhs->m_file);
#else  // MSVC
        const int res = std::strcmp(lhs->m_file, rhs->m_file);
#endif // MSVC
        if(res != 0)
            return res;
        return static_cast<int>(lhs->m_line - rhs->m_line);
    }

    // for sorting tests by suite/file/line
    int suiteOrderComparator(const void* a, const void* b) {
        const TestCase* lhs = *static_cast<TestCase* const*>(a);
        const TestCase* rhs = *static_cast<TestCase* const*>(b);

        const int res = std::strcmp(lhs->m_test_suite, rhs->m_test_suite);
        if(res != 0)
            return res;
        return fileOrderComparator(a, b);
    }

    // for sorting tests by name/suite/file/line
    int nameOrderComparator(const void* a, const void* b) {
        const TestCase* lhs = *static_cast<TestCase* const*>(a);
        const TestCase* rhs = *static_cast<TestCase* const*>(b);

        const int res_name = std::strcmp(lhs->m_name, rhs->m_name);
        if(res_name != 0)
            return res_name;
        return suiteOrderComparator(a, b);
    }

    // sets the current test suite
    int setTestSuite(const TestSuite& ts) {
        doctest_detail_test_suite_ns::getCurrentTestSuite() = ts;
        return 0;
    }

    // all the registered tests
    std::set<TestCase>& getRegisteredTests() {
        static std::set<TestCase> data;
        return data;
    }

    // used by the macros for registering tests
    int regTest(const TestCase& tc) {
        getRegisteredTests().insert(tc);
        return 0;
    }

    struct Color
    {
        enum Code
        {
            None = 0,
            White,
            Red,
            Green,
            Blue,
            Cyan,
            Yellow,
            Grey,

            Bright = 0x10,

            BrightRed   = Bright | Red,
            BrightGreen = Bright | Green,
            LightGrey   = Bright | Grey,
            BrightWhite = Bright | White
        };
        explicit Color(Code code) { use(code); }
        ~Color() { use(None); }

        static void use(Code code);
        static void init();
    };

#ifdef DOCTEST_CONFIG_COLORS_WINDOWS
    HANDLE g_stdoutHandle;
    WORD   g_originalForegroundAttributes;
    WORD   g_originalBackgroundAttributes;
    bool   g_attrsInitted = false;
#endif // DOCTEST_CONFIG_COLORS_WINDOWS

    void Color::init() {
#ifdef DOCTEST_CONFIG_COLORS_WINDOWS
        if(!g_attrsInitted) {
            g_stdoutHandle = GetStdHandle(STD_OUTPUT_HANDLE);
            g_attrsInitted = true;
            CONSOLE_SCREEN_BUFFER_INFO csbiInfo;
            GetConsoleScreenBufferInfo(g_stdoutHandle, &csbiInfo);
            g_originalForegroundAttributes =
                    csbiInfo.wAttributes &
                    ~(BACKGROUND_GREEN | BACKGROUND_RED | BACKGROUND_BLUE | BACKGROUND_INTENSITY);
            g_originalBackgroundAttributes =
                    csbiInfo.wAttributes &
                    ~(FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
        }
#endif // DOCTEST_CONFIG_COLORS_WINDOWS
    }

    void Color::use(Code
#ifndef DOCTEST_CONFIG_COLORS_NONE
                            code
#endif // DOCTEST_CONFIG_COLORS_NONE
    ) {
        const ContextState* p = contextState;
        if(p->no_colors)
            return;
#ifdef DOCTEST_CONFIG_COLORS_ANSI
        if(isatty(STDOUT_FILENO) == false && p->force_colors == false)
            return;

        const char* col = "";
        // clang-format off
        switch(code) { //!OCLINT missing break in switch statement / unnecessary default statement in covered switch statement
            case Color::Red:         col = "[0;31m"; break;
            case Color::Green:       col = "[0;32m"; break;
            case Color::Blue:        col = "[0;34m"; break;
            case Color::Cyan:        col = "[0;36m"; break;
            case Color::Yellow:      col = "[0;33m"; break;
            case Color::Grey:        col = "[1;30m"; break;
            case Color::LightGrey:   col = "[0;37m"; break;
            case Color::BrightRed:   col = "[1;31m"; break;
            case Color::BrightGreen: col = "[1;32m"; break;
            case Color::BrightWhite: col = "[1;37m"; break;
            case Color::Bright: // invalid
            case Color::None:
            case Color::White:
            default:                 col = "[0m";
        }
        // clang-format on
        std::printf("\033%s", col);
#endif // DOCTEST_CONFIG_COLORS_ANSI

#ifdef DOCTEST_CONFIG_COLORS_WINDOWS
        if(isatty(fileno(stdout)) == false && p->force_colors == false)
            return;

#define DOCTEST_SET_ATTR(x)                                                                        \
    SetConsoleTextAttribute(g_stdoutHandle, x | g_originalBackgroundAttributes)

        // clang-format off
        switch (code) {
            case Color::White:       DOCTEST_SET_ATTR(FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE); break;
            case Color::Red:         DOCTEST_SET_ATTR(FOREGROUND_RED);                                      break;
            case Color::Green:       DOCTEST_SET_ATTR(FOREGROUND_GREEN);                                    break;
            case Color::Blue:        DOCTEST_SET_ATTR(FOREGROUND_BLUE);                                     break;
            case Color::Cyan:        DOCTEST_SET_ATTR(FOREGROUND_BLUE | FOREGROUND_GREEN);                  break;
            case Color::Yellow:      DOCTEST_SET_ATTR(FOREGROUND_RED | FOREGROUND_GREEN);                   break;
            case Color::Grey:        DOCTEST_SET_ATTR(0);                                                   break;
            case Color::LightGrey:   DOCTEST_SET_ATTR(FOREGROUND_INTENSITY);                                break;
            case Color::BrightRed:   DOCTEST_SET_ATTR(FOREGROUND_INTENSITY | FOREGROUND_RED);               break;
            case Color::BrightGreen: DOCTEST_SET_ATTR(FOREGROUND_INTENSITY | FOREGROUND_GREEN);             break;
            case Color::BrightWhite: DOCTEST_SET_ATTR(FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE); break;
            case Color::None:
            case Color::Bright: // invalid
            default:                 DOCTEST_SET_ATTR(g_originalForegroundAttributes);
        }
// clang-format on
#undef DOCTEST_SET_ATTR
#endif // DOCTEST_CONFIG_COLORS_WINDOWS
    }

    std::vector<const IExceptionTranslator*>& getExceptionTranslators() {
        static std::vector<const IExceptionTranslator*> data;
        return data;
    }

    void registerExceptionTranslatorImpl(const IExceptionTranslator* translateFunction) {
        if(std::find(getExceptionTranslators().begin(), getExceptionTranslators().end(),
                     translateFunction) == getExceptionTranslators().end())
            getExceptionTranslators().push_back(translateFunction);
    }

    String translateActiveException() {
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
        String                                    res;
        std::vector<const IExceptionTranslator*>& translators = getExceptionTranslators();
        for(size_t i = 0; i < translators.size(); ++i)
            if(translators[i]->translate(res))
                return res;
        // clang-format off
        try {
            throw;
        } catch(std::exception& ex) {
            return ex.what();
        } catch(std::string& msg) {
            return msg.c_str();
        } catch(const char* msg) {
            return msg;
        } catch(...) {
            return "unknown exception";
        }
// clang-format on
#else  // DOCTEST_CONFIG_NO_EXCEPTIONS
        return "";
#endif // DOCTEST_CONFIG_NO_EXCEPTIONS
    }

    void writeStringToStream(std::ostream* stream, const String& str) { *stream << str; }

#ifdef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
    void toStream(std::ostream* stream, char* in) { *stream << in; }
    void toStream(std::ostream* stream, const char* in) { *stream << in; }
#endif // DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
    void toStream(std::ostream* stream, bool in) {
        *stream << std::boolalpha << in << std::noboolalpha;
    }
    void toStream(std::ostream* stream, float in) { *stream << in; }
    void toStream(std::ostream* stream, double in) { *stream << in; }
    void toStream(std::ostream* stream, double long in) { *stream << in; }

    void toStream(std::ostream* stream, char in) { *stream << in; }
    void toStream(std::ostream* stream, char signed in) { *stream << in; }
    void toStream(std::ostream* stream, char unsigned in) { *stream << in; }
    void toStream(std::ostream* stream, int short in) { *stream << in; }
    void toStream(std::ostream* stream, int short unsigned in) { *stream << in; }
    void toStream(std::ostream* stream, int in) { *stream << in; }
    void toStream(std::ostream* stream, int unsigned in) { *stream << in; }
    void toStream(std::ostream* stream, int long in) { *stream << in; }
    void toStream(std::ostream* stream, int long unsigned in) { *stream << in; }

#ifdef DOCTEST_CONFIG_WITH_LONG_LONG
    void toStream(std::ostream* stream, int long long in) { *stream << in; }
    void toStream(std::ostream* stream, int long long unsigned in) { *stream << in; }
#endif // DOCTEST_CONFIG_WITH_LONG_LONG

    void addToContexts(IContextScope* ptr) { contextState->contexts.push_back(ptr); }
    void popFromContexts() { contextState->contexts.pop_back(); }
    DOCTEST_MSVC_SUPPRESS_WARNING_WITH_PUSH(4996) // std::uncaught_exception is deprecated in C++17
    void useContextIfExceptionOccurred(IContextScope* ptr) {
        if(std::uncaught_exception()) {
            std::ostringstream stream;
            ptr->build(&stream);
            contextState->exceptionalContexts.push_back(stream.str());
        }
    }
    DOCTEST_MSVC_SUPPRESS_WARNING_POP

    void printSummary();

#if !defined(DOCTEST_CONFIG_POSIX_SIGNALS) && !defined(DOCTEST_CONFIG_WINDOWS_SEH)
    void reportFatal(const std::string&) {}
    struct FatalConditionHandler
    {
        void reset() {}
    };
#else // DOCTEST_CONFIG_POSIX_SIGNALS || DOCTEST_CONFIG_WINDOWS_SEH

    void reportFatal(const std::string& message) {
        DOCTEST_LOG_START();

        contextState->numAssertions += contextState->numAssertionsForCurrentTestcase;
        logTestException(message.c_str(), true);
        logTestEnd();
        contextState->numFailed++;

        printSummary();
    }

#ifdef DOCTEST_PLATFORM_WINDOWS

    struct SignalDefs
    {
        DWORD       id;
        const char* name;
    };
    // There is no 1-1 mapping between signals and windows exceptions.
    // Windows can easily distinguish between SO and SigSegV,
    // but SigInt, SigTerm, etc are handled differently.
    SignalDefs signalDefs[] = {
            {EXCEPTION_ILLEGAL_INSTRUCTION, "SIGILL - Illegal instruction signal"},
            {EXCEPTION_STACK_OVERFLOW, "SIGSEGV - Stack overflow"},
            {EXCEPTION_ACCESS_VIOLATION, "SIGSEGV - Segmentation violation signal"},
            {EXCEPTION_INT_DIVIDE_BY_ZERO, "Divide by zero error"},
    };

    struct FatalConditionHandler
    {
        static LONG CALLBACK handleVectoredException(PEXCEPTION_POINTERS ExceptionInfo) {
            for(size_t i = 0; i < sizeof(signalDefs) / sizeof(SignalDefs); ++i) {
                if(ExceptionInfo->ExceptionRecord->ExceptionCode == signalDefs[i].id) {
                    reportFatal(signalDefs[i].name);
                }
            }
            // If its not an exception we care about, pass it along.
            // This stops us from eating debugger breaks etc.
            return EXCEPTION_CONTINUE_SEARCH;
        }

        FatalConditionHandler() {
            isSet = true;
            // 32k seems enough for doctest to handle stack overflow,
            // but the value was found experimentally, so there is no strong guarantee
            guaranteeSize          = 32 * 1024;
            exceptionHandlerHandle = 0;
            // Register as first handler in current chain
            exceptionHandlerHandle = AddVectoredExceptionHandler(1, handleVectoredException);
            // Pass in guarantee size to be filled
            SetThreadStackGuarantee(&guaranteeSize);
        }

        static void reset() {
            if(isSet) {
                // Unregister handler and restore the old guarantee
                RemoveVectoredExceptionHandler(exceptionHandlerHandle);
                SetThreadStackGuarantee(&guaranteeSize);
                exceptionHandlerHandle = 0;
                isSet                  = false;
            }
        }

        ~FatalConditionHandler() { reset(); }

    private:
        static bool  isSet;
        static ULONG guaranteeSize;
        static PVOID exceptionHandlerHandle;
    };

    bool  FatalConditionHandler::isSet                  = false;
    ULONG FatalConditionHandler::guaranteeSize          = 0;
    PVOID FatalConditionHandler::exceptionHandlerHandle = 0;

#else // DOCTEST_PLATFORM_WINDOWS

    struct SignalDefs
    {
        int         id;
        const char* name;
    };
    SignalDefs signalDefs[] = {{SIGINT, "SIGINT - Terminal interrupt signal"},
                               {SIGILL, "SIGILL - Illegal instruction signal"},
                               {SIGFPE, "SIGFPE - Floating point error signal"},
                               {SIGSEGV, "SIGSEGV - Segmentation violation signal"},
                               {SIGTERM, "SIGTERM - Termination request signal"},
                               {SIGABRT, "SIGABRT - Abort (abnormal termination) signal"}};

    struct FatalConditionHandler
    {
        static bool             isSet;
        static struct sigaction oldSigActions[sizeof(signalDefs) / sizeof(SignalDefs)];
        static stack_t          oldSigStack;
        static char             altStackMem[SIGSTKSZ];

        static void handleSignal(int sig) {
            std::string name = "<unknown signal>";
            for(std::size_t i = 0; i < sizeof(signalDefs) / sizeof(SignalDefs); ++i) {
                SignalDefs& def = signalDefs[i];
                if(sig == def.id) {
                    name = def.name;
                    break;
                }
            }
            reset();
            reportFatal(name);
            raise(sig);
        }

        FatalConditionHandler() {
            isSet = true;
            stack_t sigStack;
            sigStack.ss_sp    = altStackMem;
            sigStack.ss_size  = SIGSTKSZ;
            sigStack.ss_flags = 0;
            sigaltstack(&sigStack, &oldSigStack);
            struct sigaction sa = {0};

            sa.sa_handler = handleSignal; // NOLINT
            sa.sa_flags   = SA_ONSTACK;
            for(std::size_t i = 0; i < sizeof(signalDefs) / sizeof(SignalDefs); ++i) {
                sigaction(signalDefs[i].id, &sa, &oldSigActions[i]);
            }
        }

        ~FatalConditionHandler() { reset(); }
        static void reset() {
            if(isSet) {
                // Set signals back to previous values -- hopefully nobody overwrote them in the meantime
                for(std::size_t i = 0; i < sizeof(signalDefs) / sizeof(SignalDefs); ++i) {
                    sigaction(signalDefs[i].id, &oldSigActions[i], 0);
                }
                // Return the old stack
                sigaltstack(&oldSigStack, 0);
                isSet = false;
            }
        }
    };

    bool             FatalConditionHandler::isSet = false;
    struct sigaction FatalConditionHandler::oldSigActions[sizeof(signalDefs) / sizeof(SignalDefs)] =
            {};
    stack_t FatalConditionHandler::oldSigStack           = {};
    char    FatalConditionHandler::altStackMem[SIGSTKSZ] = {};

#endif // DOCTEST_PLATFORM_WINDOWS
#endif // DOCTEST_CONFIG_POSIX_SIGNALS || DOCTEST_CONFIG_WINDOWS_SEH

    // depending on the current options this will remove the path of filenames
    const char* fileForOutput(const char* file) {
        if(contextState->no_path_in_filenames) {
            const char* back    = std::strrchr(file, '\\');
            const char* forward = std::strrchr(file, '/');
            if(back || forward) {
                if(back > forward)
                    forward = back;
                return forward + 1;
            }
        }
        return file;
    }

    // depending on the current options this will substitute the line numbers with 0
    int lineForOutput(int line) {
        if(contextState->no_line_numbers)
            return 0;
        return line;
    }

#ifdef DOCTEST_PLATFORM_MAC
#include <sys/types.h>
#include <unistd.h>
#include <sys/sysctl.h>
    // The following function is taken directly from the following technical note:
    // http://developer.apple.com/library/mac/#qa/qa2004/qa1361.html
    // Returns true if the current process is being debugged (either
    // running under the debugger or has a debugger attached post facto).
    bool isDebuggerActive() {
        int        mib[4];
        kinfo_proc info;
        size_t     size;
        // Initialize the flags so that, if sysctl fails for some bizarre
        // reason, we get a predictable result.
        info.kp_proc.p_flag = 0;
        // Initialize mib, which tells sysctl the info we want, in this case
        // we're looking for information about a specific process ID.
        mib[0] = CTL_KERN;
        mib[1] = KERN_PROC;
        mib[2] = KERN_PROC_PID;
        mib[3] = getpid();
        // Call sysctl.
        size = sizeof(info);
        if(sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &size, 0, 0) != 0) {
            fprintf(stderr, "\n** Call to sysctl failed - unable to determine if debugger is "
                            "active **\n\n");
            return false;
        }
        // We're being debugged if the P_TRACED flag is set.
        return ((info.kp_proc.p_flag & P_TRACED) != 0);
    }
#elif DOCTEST_MSVC || defined(__MINGW32__)
    bool  isDebuggerActive() { return ::IsDebuggerPresent() != 0; }
#else
    bool isDebuggerActive() { return false; }
#endif // Platform

#ifdef DOCTEST_PLATFORM_WINDOWS
    void myOutputDebugString(const String& text) { ::OutputDebugStringA(text.c_str()); }
#else
    // TODO: integration with XCode and other IDEs
    void myOutputDebugString(const String&) {}
#endif // Platform

    const char* getSeparator() {
        return "===============================================================================\n";
    }

    void printToDebugConsole(const String& text) {
        if(isDebuggerActive())
            myOutputDebugString(text.c_str());
    }

    void addFailedAssert(assertType::Enum assert_type) {
        if((assert_type & assertType::is_warn) == 0) { //!OCLINT bitwise operator in conditional
            contextState->numFailedAssertions++;
            contextState->numFailedAssertionsForCurrentTestcase++;
            contextState->hasCurrentTestFailed = true;
        }
    }

    void logTestStart(const TestCase& tc) {
        char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)\n", fileForOutput(tc.m_file),
                         lineForOutput(tc.m_line));

        char ts1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(ts1, DOCTEST_COUNTOF(ts1), "TEST SUITE: ");
        char ts2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(ts2, DOCTEST_COUNTOF(ts2), "%s\n", tc.m_test_suite);
        char n1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(n1, DOCTEST_COUNTOF(n1), "TEST CASE:  ");
        char n2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(n2, DOCTEST_COUNTOF(n2), "%s\n", tc.m_name);
        char d1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(d1, DOCTEST_COUNTOF(d1), "DESCRIPTION: ");
        char d2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(d2, DOCTEST_COUNTOF(d2), "%s\n", tc.m_description);

        // hack for BDD style of macros - to not print "TEST CASE:"
        char scenario[] = "  Scenario:";
        if(std::string(tc.m_name).substr(0, DOCTEST_COUNTOF(scenario) - 1) == scenario)
            n1[0] = '\0';

        DOCTEST_PRINTF_COLORED(getSeparator(), Color::Yellow);
        DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);

        String forDebugConsole;
        if(tc.m_description) {
            DOCTEST_PRINTF_COLORED(d1, Color::Yellow);
            DOCTEST_PRINTF_COLORED(d2, Color::None);
            forDebugConsole += d1;
            forDebugConsole += d2;
        }
        if(tc.m_test_suite && tc.m_test_suite[0] != '\0') {
            DOCTEST_PRINTF_COLORED(ts1, Color::Yellow);
            DOCTEST_PRINTF_COLORED(ts2, Color::None);
            forDebugConsole += ts1;
            forDebugConsole += ts2;
        }
        DOCTEST_PRINTF_COLORED(n1, Color::Yellow);
        DOCTEST_PRINTF_COLORED(n2, Color::None);

        String                subcaseStuff;
        std::vector<Subcase>& subcasesStack = contextState->subcasesStack;
        for(unsigned i = 0; i < subcasesStack.size(); ++i) {
            if(subcasesStack[i].m_signature.m_name[0] != '\0') {
                char subcase[DOCTEST_SNPRINTF_BUFFER_LENGTH];
                DOCTEST_SNPRINTF(subcase, DOCTEST_COUNTOF(loc), "  %s\n",
                                 subcasesStack[i].m_signature.m_name);
                DOCTEST_PRINTF_COLORED(subcase, Color::None);
                subcaseStuff += subcase;
            }
        }

        DOCTEST_PRINTF_COLORED("\n", Color::None);

        printToDebugConsole(String(getSeparator()) + loc + forDebugConsole.c_str() + n1 + n2 +
                            subcaseStuff.c_str() + "\n");
    }

    void logTestEnd() {}

    void logTestException(const String& what, bool crash) {
        char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];

        DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), "TEST CASE FAILED!\n");

        char info1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        char info2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        info1[0] = 0;
        info2[0] = 0;
        DOCTEST_SNPRINTF(info1, DOCTEST_COUNTOF(info1),
                         crash ? "crashed:\n" : "threw exception:\n");
        DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "  %s\n", what.c_str());

        std::string contextStr;

        if(!contextState->exceptionalContexts.empty()) {
            contextStr += "with context:\n";
            for(size_t i = contextState->exceptionalContexts.size(); i > 0; --i) {
                contextStr += "  ";
                contextStr += contextState->exceptionalContexts[i - 1];
                contextStr += "\n";
            }
        }

        DOCTEST_PRINTF_COLORED(msg, Color::Red);
        DOCTEST_PRINTF_COLORED(info1, Color::None);
        DOCTEST_PRINTF_COLORED(info2, Color::Cyan);
        DOCTEST_PRINTF_COLORED(contextStr.c_str(), Color::None);
        DOCTEST_PRINTF_COLORED("\n", Color::None);

        printToDebugConsole(String(msg) + info1 + info2 + contextStr.c_str() + "\n");
    }

    String logContext() {
        std::ostringstream           stream;
        std::vector<IContextScope*>& contexts = contextState->contexts;
        if(!contexts.empty())
            stream << "with context:\n";
        for(size_t i = 0; i < contexts.size(); ++i) {
            stream << "  ";
            contexts[i]->build(&stream);
            stream << "\n";
        }
        return stream.str().c_str();
    }

    const char* getFailString(assertType::Enum assert_type) {
        if(assert_type & assertType::is_warn) //!OCLINT bitwise operator in conditional
            return "WARNING";
        if(assert_type & assertType::is_check) //!OCLINT bitwise operator in conditional
            return "ERROR";
        if(assert_type & assertType::is_require) //!OCLINT bitwise operator in conditional
            return "FATAL ERROR";
        return "";
    }

    void logAssert(bool passed, const char* decomposition, bool threw, const String& exception,
                   const char* expr, assertType::Enum assert_type, const char* file, int line) {
        char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)", fileForOutput(file),
                         lineForOutput(line));

        char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), " %s!\n",
                         passed ? "PASSED" : getFailString(assert_type));

        char info1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(info1, DOCTEST_COUNTOF(info1), "  %s( %s )\n",
                         getAssertString(assert_type), expr);

        char info2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        char info3[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        info2[0] = 0;
        info3[0] = 0;
        if(threw) {
            DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "threw exception:\n");
            DOCTEST_SNPRINTF(info3, DOCTEST_COUNTOF(info3), "  %s\n", exception.c_str());
        } else {
            DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "with expansion:\n");
            DOCTEST_SNPRINTF(info3, DOCTEST_COUNTOF(info3), "  %s( %s )\n",
                             getAssertString(assert_type), decomposition);
        }

        const bool isWarn = assert_type & assertType::is_warn;
        DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);
        DOCTEST_PRINTF_COLORED(msg,
                               passed ? Color::BrightGreen : isWarn ? Color::Yellow : Color::Red);
        DOCTEST_PRINTF_COLORED(info1, Color::Cyan);
        DOCTEST_PRINTF_COLORED(info2, Color::None);
        DOCTEST_PRINTF_COLORED(info3, Color::Cyan);
        String context = logContext();
        DOCTEST_PRINTF_COLORED(context.c_str(), Color::None);
        DOCTEST_PRINTF_COLORED("\n", Color::None);

        printToDebugConsole(String(loc) + msg + info1 + info2 + info3 + context.c_str() + "\n");
    }

    void logAssertThrows(bool threw, const char* expr, assertType::Enum assert_type,
                         const char* file, int line) {
        char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)", fileForOutput(file),
                         lineForOutput(line));

        char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), " %s!\n",
                         threw ? "PASSED" : getFailString(assert_type));

        char info1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(info1, DOCTEST_COUNTOF(info1), "  %s( %s )\n",
                         getAssertString(assert_type), expr);

        char info2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        info2[0] = 0;

        if(!threw)
            DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "didn't throw at all\n");

        const bool isWarn = assert_type & assertType::is_warn;
        DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);
        DOCTEST_PRINTF_COLORED(msg,
                               threw ? Color::BrightGreen : isWarn ? Color::Yellow : Color::Red);
        DOCTEST_PRINTF_COLORED(info1, Color::Cyan);
        DOCTEST_PRINTF_COLORED(info2, Color::None);
        String context = logContext();
        DOCTEST_PRINTF_COLORED(context.c_str(), Color::None);
        DOCTEST_PRINTF_COLORED("\n", Color::None);

        printToDebugConsole(String(loc) + msg + info1 + info2 + context.c_str() + "\n");
    }

    void logAssertThrowsAs(bool threw, bool threw_as, const char* as, const String& exception,
                           const char* expr, assertType::Enum assert_type, const char* file,
                           int line) {
        char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)", fileForOutput(file),
                         lineForOutput(line));

        char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), " %s!\n",
                         threw_as ? "PASSED" : getFailString(assert_type));

        char info1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(info1, DOCTEST_COUNTOF(info1), "  %s( %s, %s )\n",
                         getAssertString(assert_type), expr, as);

        char info2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        char info3[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        info2[0] = 0;
        info3[0] = 0;

        if(!threw) { //!OCLINT inverted logic
            DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "didn't throw at all\n");
        } else if(!threw_as) {
            DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "threw a different exception:\n");
            DOCTEST_SNPRINTF(info3, DOCTEST_COUNTOF(info3), "  %s\n", exception.c_str());
        }

        const bool isWarn = assert_type & assertType::is_warn;
        DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);
        DOCTEST_PRINTF_COLORED(msg,
                               threw_as ? Color::BrightGreen : isWarn ? Color::Yellow : Color::Red);
        DOCTEST_PRINTF_COLORED(info1, Color::Cyan);
        DOCTEST_PRINTF_COLORED(info2, Color::None);
        DOCTEST_PRINTF_COLORED(info3, Color::Cyan);
        String context = logContext();
        DOCTEST_PRINTF_COLORED(context.c_str(), Color::None);
        DOCTEST_PRINTF_COLORED("\n", Color::None);

        printToDebugConsole(String(loc) + msg + info1 + info2 + info3 + context.c_str() + "\n");
    }

    void logAssertNothrow(bool threw, const String& exception, const char* expr,
                          assertType::Enum assert_type, const char* file, int line) {
        char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)", fileForOutput(file),
                         lineForOutput(line));

        char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), " %s!\n",
                         threw ? getFailString(assert_type) : "PASSED");

        char info1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(info1, DOCTEST_COUNTOF(info1), "  %s( %s )\n",
                         getAssertString(assert_type), expr);

        char info2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        char info3[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        info2[0] = 0;
        info3[0] = 0;
        if(threw) {
            DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "threw exception:\n");
            DOCTEST_SNPRINTF(info3, DOCTEST_COUNTOF(info3), "  %s\n", exception.c_str());
        }

        const bool isWarn = assert_type & assertType::is_warn;
        DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);
        DOCTEST_PRINTF_COLORED(msg,
                               threw ? isWarn ? Color::Yellow : Color::Red : Color::BrightGreen);
        DOCTEST_PRINTF_COLORED(info1, Color::Cyan);
        DOCTEST_PRINTF_COLORED(info2, Color::None);
        DOCTEST_PRINTF_COLORED(info3, Color::Cyan);
        String context = logContext();
        DOCTEST_PRINTF_COLORED(context.c_str(), Color::None);
        DOCTEST_PRINTF_COLORED("\n", Color::None);

        printToDebugConsole(String(loc) + msg + info1 + info2 + info3 + context.c_str() + "\n");
    }

    ResultBuilder::ResultBuilder(assertType::Enum assert_type, const char* file, int line,
                                 const char* expr, const char* exception_type)
            : m_assert_type(assert_type)
            , m_file(file)
            , m_line(line)
            , m_expr(expr)
            , m_exception_type(exception_type)
            , m_threw(false)
            , m_threw_as(false)
            , m_failed(false) {
#if DOCTEST_MSVC
        if(m_expr[0] == ' ') // this happens when variadic macros are disabled under MSVC
            ++m_expr;
#endif // MSVC
    }

    ResultBuilder::~ResultBuilder() {}

    void ResultBuilder::unexpectedExceptionOccurred() {
        m_threw = true;

        m_exception = translateActiveException();
    }

    bool ResultBuilder::log() {
        if((m_assert_type & assertType::is_warn) == 0) //!OCLINT bitwise operator in conditional
            contextState->numAssertionsForCurrentTestcase++;

        if(m_assert_type & assertType::is_throws) { //!OCLINT bitwise operator in conditional
            m_failed = !m_threw;
        } else if(m_assert_type & //!OCLINT bitwise operator in conditional
                  assertType::is_throws_as) {
            m_failed = !m_threw_as;
        } else if(m_assert_type & //!OCLINT bitwise operator in conditional
                  assertType::is_nothrow) {
            m_failed = m_threw;
        } else {
            m_failed = m_result;
        }

        if(m_failed || contextState->success) {
            DOCTEST_LOG_START();

            if(m_assert_type & assertType::is_throws) { //!OCLINT bitwise operator in conditional
                logAssertThrows(m_threw, m_expr, m_assert_type, m_file, m_line);
            } else if(m_assert_type & //!OCLINT bitwise operator in conditional
                      assertType::is_throws_as) {
                logAssertThrowsAs(m_threw, m_threw_as, m_exception_type, m_exception, m_expr,
                                  m_assert_type, m_file, m_line);
            } else if(m_assert_type & //!OCLINT bitwise operator in conditional
                      assertType::is_nothrow) {
                logAssertNothrow(m_threw, m_exception, m_expr, m_assert_type, m_file, m_line);
            } else {
                logAssert(m_result.m_passed, m_result.m_decomposition.c_str(), m_threw, m_exception,
                          m_expr, m_assert_type, m_file, m_line);
            }
        }

        if(m_failed)
            addFailedAssert(m_assert_type);

        return m_failed && isDebuggerActive() && !contextState->no_breaks; // break into debugger
    }

    void ResultBuilder::react() const {
        if(m_failed && checkIfShouldThrow(m_assert_type))
            throwException();
    }

    MessageBuilder::MessageBuilder(const char* file, int line, assertType::Enum severity)
            : m_stream(createStream())
            , m_file(file)
            , m_line(line)
            , m_severity(severity) {}

    bool MessageBuilder::log() {
        DOCTEST_LOG_START();

        const bool isWarn = m_severity & assertType::is_warn;

        // warn is just a message in this context so we dont treat it as an assert
        if(!isWarn) {
            contextState->numAssertionsForCurrentTestcase++;
            addFailedAssert(m_severity);
        }

        char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)", fileForOutput(m_file),
                         lineForOutput(m_line));
        char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
        DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), " %s!\n",
                         isWarn ? "MESSAGE" : getFailString(m_severity));

        DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);
        DOCTEST_PRINTF_COLORED(msg, isWarn ? Color::Yellow : Color::Red);

        String info = getStreamResult(m_stream);
        if(info.size()) {
            DOCTEST_PRINTF_COLORED("  ", Color::None);
            DOCTEST_PRINTF_COLORED(info.c_str(), Color::None);
            DOCTEST_PRINTF_COLORED("\n", Color::None);
        }
        String context = logContext();
        DOCTEST_PRINTF_COLORED(context.c_str(), Color::None);
        DOCTEST_PRINTF_COLORED("\n", Color::None);

        printToDebugConsole(String(loc) + msg + "  " + info.c_str() + "\n" + context.c_str() +
                            "\n");

        return isDebuggerActive() && !contextState->no_breaks && !isWarn; // break into debugger
    }

    void MessageBuilder::react() {
        if(m_severity & assertType::is_require) //!OCLINT bitwise operator in conditional
            throwException();
    }

    MessageBuilder::~MessageBuilder() { freeStream(m_stream); }

    // the implementation of parseFlag()
    bool parseFlagImpl(int argc, const char* const* argv, const char* pattern) {
        for(int i = argc - 1; i >= 0; --i) {
            const char* temp = std::strstr(argv[i], pattern);
            if(temp && my_strlen(temp) == my_strlen(pattern)) {
                // eliminate strings in which the chars before the option are not '-'
                bool noBadCharsFound = true; //!OCLINT prefer early exits and continue
                while(temp != argv[i]) {
                    if(*--temp != '-') {
                        noBadCharsFound = false;
                        break;
                    }
                }
                if(noBadCharsFound && argv[i][0] == '-')
                    return true;
            }
        }
        return false;
    }

    // locates a flag on the command line
    bool parseFlag(int argc, const char* const* argv, const char* pattern) {
#ifndef DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
        if(!parseFlagImpl(argc, argv, pattern))
            return parseFlagImpl(argc, argv, pattern + 3); // 3 for "dt-"
        return true;
#else  // DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
        return parseFlagImpl(argc, argv, pattern);
#endif // DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
    }

    // the implementation of parseOption()
    bool parseOptionImpl(int argc, const char* const* argv, const char* pattern, String& res) {
        for(int i = argc - 1; i >= 0; --i) {
            const char* temp = std::strstr(argv[i], pattern);
            if(temp) { //!OCLINT prefer early exits and continue
                // eliminate matches in which the chars before the option are not '-'
                bool        noBadCharsFound = true;
                const char* curr            = argv[i];
                while(curr != temp) {
                    if(*curr++ != '-') {
                        noBadCharsFound = false;
                        break;
                    }
                }
                if(noBadCharsFound && argv[i][0] == '-') {
                    temp += my_strlen(pattern);
                    const unsigned len = my_strlen(temp);
                    if(len) {
                        res = temp;
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // parses an option and returns the string after the '=' character
    bool parseOption(int argc, const char* const* argv, const char* pattern, String& res,
                     const String& defaultVal = String()) {
        res = defaultVal;
#ifndef DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
        if(!parseOptionImpl(argc, argv, pattern, res))
            return parseOptionImpl(argc, argv, pattern + 3, res); // 3 for "dt-"
        return true;
#else  // DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
        return parseOptionImpl(argc, argv, pattern, res);
#endif // DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
    }

    // parses a comma separated list of words after a pattern in one of the arguments in argv
    bool parseCommaSepArgs(int argc, const char* const* argv, const char* pattern,
                           std::vector<String>& res) {
        String filtersString;
        if(parseOption(argc, argv, pattern, filtersString)) {
            // tokenize with "," as a separator
            // cppcheck-suppress strtokCalled
            char* pch = std::strtok(filtersString.c_str(), ","); // modifies the string
            while(pch != 0) {
                if(my_strlen(pch))
                    res.push_back(pch);
                // uses the strtok() internal state to go to the next token
                // cppcheck-suppress strtokCalled
                pch = std::strtok(0, ",");
            }
            return true;
        }
        return false;
    }

    enum optionType
    {
        option_bool,
        option_int
    };

    // parses an int/bool option from the command line
    bool parseIntOption(int argc, const char* const* argv, const char* pattern, optionType type,
                        int& res) {
        String parsedValue;
        if(!parseOption(argc, argv, pattern, parsedValue))
            return false;

        if(type == 0) {
            // boolean
            const char positive[][5] = {"1", "true", "on", "yes"};  // 5 - strlen("true") + 1
            const char negative[][6] = {"0", "false", "off", "no"}; // 6 - strlen("false") + 1

            // if the value matches any of the positive/negative possibilities
            for(unsigned i = 0; i < 4; i++) {
                if(parsedValue.compare(positive[i], true) == 0) {
                    res = 1; //!OCLINT parameter reassignment
                    return true;
                }
                if(parsedValue.compare(negative[i], true) == 0) {
                    res = 0; //!OCLINT parameter reassignment
                    return true;
                }
            }
        } else {
            // integer
            int theInt = std::atoi(parsedValue.c_str()); // NOLINT
            if(theInt != 0) {
                res = theInt; //!OCLINT parameter reassignment
                return true;
            }
        }
        return false;
    }

    void printVersion() {
        if(contextState->no_version == false) {
            DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
            std::printf("doctest version is \"%s\"\n", DOCTEST_VERSION_STR);
        }
    }

    void printHelp() {
        printVersion();
        // clang-format off
        DOCTEST_PRINTF_COLORED("[doctest]\n", Color::Cyan);
        DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
        std::printf("boolean values: \"1/on/yes/true\" or \"0/off/no/false\"\n");
        DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
        std::printf("filter  values: \"str1,str2,str3\" (comma separated strings)\n");
        DOCTEST_PRINTF_COLORED("[doctest]\n", Color::Cyan);
        DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
        std::printf("filters use wildcards for matching strings\n");
        DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
        std::printf("something passes a filter if any of the strings in a filter matches\n");
        DOCTEST_PRINTF_COLORED("[doctest]\n", Color::Cyan);
        DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
        std::printf("ALL FLAGS, OPTIONS AND FILTERS ALSO AVAILABLE WITH A \"dt-\" PREFIX!!!\n");
        DOCTEST_PRINTF_COLORED("[doctest]\n", Color::Cyan);
        DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
        std::printf("Query flags - the program quits after them. Available:\n\n");
        std::printf(" -?,   --help, -h                      prints this message\n");
        std::printf(" -v,   --version                       prints the version\n");
        std::printf(" -c,   --count                         prints the number of matching tests\n");
        std::printf(" -ltc, --list-test-cases               lists all matching tests by name\n");
        std::printf(" -lts, --list-test-suites              lists all matching test suites\n\n");
        // ========================================================================================= << 79
        DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
        std::printf("The available <int>/<string> options/filters are:\n\n");
        std::printf(" -tc,  --test-case=<filters>           filters     tests by their name\n");
        std::printf(" -tce, --test-case-exclude=<filters>   filters OUT tests by their name\n");
        std::printf(" -sf,  --source-file=<filters>         filters     tests by their file\n");
        std::printf(" -sfe, --source-file-exclude=<filters> filters OUT tests by their file\n");
        std::printf(" -ts,  --test-suite=<filters>          filters     tests by their test suite\n");
        std::printf(" -tse, --test-suite-exclude=<filters>  filters OUT tests by their test suite\n");
        std::printf(" -sc,  --subcase=<filters>             filters     subcases by their name\n");
        std::printf(" -sce, --subcase-exclude=<filters>     filters OUT subcases by their name\n");
        std::printf(" -ob,  --order-by=<string>             how the tests should be ordered\n");
        std::printf("                                       <string> - by [file/suite/name/rand]\n");
        std::printf(" -rs,  --rand-seed=<int>               seed for random ordering\n");
        std::printf(" -f,   --first=<int>                   the first test passing the filters to\n");
        std::printf("                                       execute - for range-based execution\n");
        std::printf(" -l,   --last=<int>                    the last test passing the filters to\n");
        std::printf("                                       execute - for range-based execution\n");
        std::printf(" -aa,  --abort-after=<int>             stop after <int> failed assertions\n");
        std::printf(" -scfl,--subcase-filter-levels=<int>   apply filters for the first <int> levels\n");
        DOCTEST_PRINTF_COLORED("\n[doctest] ", Color::Cyan);
        std::printf("Bool options - can be used like flags and true is assumed. Available:\n\n");
        std::printf(" -s,   --success=<bool>                include successful assertions in output\n");
        std::printf(" -cs,  --case-sensitive=<bool>         filters being treated as case sensitive\n");
        std::printf(" -e,   --exit=<bool>                   exits after the tests finish\n");
        std::printf(" -d,   --duration=<bool>               prints the time duration of each test\n");
        std::printf(" -nt,  --no-throw=<bool>               skips exceptions-related assert checks\n");
        std::printf(" -ne,  --no-exitcode=<bool>            returns (or exits) always with success\n");
        std::printf(" -nr,  --no-run=<bool>                 skips all runtime doctest operations\n");
        std::printf(" -nv,  --no-version=<bool>             omit the framework version in the output\n");
        std::printf(" -nc,  --no-colors=<bool>              disables colors in output\n");
        std::printf(" -fc,  --force-colors=<bool>           use colors even when not in a tty\n");
        std::printf(" -nb,  --no-breaks=<bool>              disables breakpoints in debuggers\n");
        std::printf(" -ns,  --no-skip=<bool>                don't skip test cases marked as skip\n");
        std::printf(" -npf, --no-path-filenames=<bool>      only filenames and no paths in output\n");
        std::printf(" -nln, --no-line-numbers=<bool>        0 instead of real line numbers in output\n");
        // ========================================================================================= << 79
        // clang-format on

        DOCTEST_PRINTF_COLORED("\n[doctest] ", Color::Cyan);
        std::printf("for more information visit the project documentation\n\n");
    }

    void printSummary() {
        const ContextState* p = contextState;

        DOCTEST_PRINTF_COLORED(getSeparator(), Color::Yellow);
        if(p->count || p->list_test_cases) {
            DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
            std::printf("unskipped test cases passing the current filters: %u\n",
                        p->numTestsPassingFilters);
        } else if(p->list_test_suites) {
            DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
            std::printf("unskipped test cases passing the current filters: %u\n",
                        p->numTestsPassingFilters);
            DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
            std::printf("test suites with unskipped test cases passing the current filters: %u\n",
                        p->numTestSuitesPassingFilters);
        } else {
            const bool anythingFailed = p->numFailed > 0 || p->numFailedAssertions > 0;

            char buff[DOCTEST_SNPRINTF_BUFFER_LENGTH];

            DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);

            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "test cases: %6u",
                             p->numTestsPassingFilters);
            DOCTEST_PRINTF_COLORED(buff, Color::None);
            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " | ");
            DOCTEST_PRINTF_COLORED(buff, Color::None);
            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "%6u passed",
                             p->numTestsPassingFilters - p->numFailed);
            DOCTEST_PRINTF_COLORED(buff, (p->numTestsPassingFilters == 0 || anythingFailed) ?
                                                 Color::None :
                                                 Color::Green);
            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " | ");
            DOCTEST_PRINTF_COLORED(buff, Color::None);
            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "%6u failed", p->numFailed);
            DOCTEST_PRINTF_COLORED(buff, p->numFailed > 0 ? Color::Red : Color::None);

            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " | ");
            DOCTEST_PRINTF_COLORED(buff, Color::None);
            if(p->no_skipped_summary == false) {
                const int numSkipped = static_cast<unsigned>(getRegisteredTests().size()) -
                                       p->numTestsPassingFilters;
                DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "%6d skipped", numSkipped);
                DOCTEST_PRINTF_COLORED(buff, numSkipped == 0 ? Color::None : Color::Yellow);
            }
            DOCTEST_PRINTF_COLORED("\n", Color::None);

            DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);

            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "assertions: %6d", p->numAssertions);
            DOCTEST_PRINTF_COLORED(buff, Color::None);
            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " | ");
            DOCTEST_PRINTF_COLORED(buff, Color::None);
            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "%6d passed",
                             p->numAssertions - p->numFailedAssertions);
            DOCTEST_PRINTF_COLORED(buff, (p->numAssertions == 0 || anythingFailed) ? Color::None :
                                                                                     Color::Green);
            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " | ");
            DOCTEST_PRINTF_COLORED(buff, Color::None);
            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "%6d failed", p->numFailedAssertions);
            DOCTEST_PRINTF_COLORED(buff, p->numFailedAssertions > 0 ? Color::Red : Color::None);

            DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " |\n");
            DOCTEST_PRINTF_COLORED(buff, Color::None);

            DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
            DOCTEST_PRINTF_COLORED("Status: ", Color::None);
            const char* result = (p->numFailed > 0) ? "FAILURE!\n" : "SUCCESS!\n";
            DOCTEST_PRINTF_COLORED(result, p->numFailed > 0 ? Color::Red : Color::Green);
        }

        // remove any coloring
        DOCTEST_PRINTF_COLORED("", Color::None);
    }
} // namespace detail

bool isRunningInTest() { return detail::contextState != 0; }

Context::Context(int argc, const char* const* argv)
        : p(new detail::ContextState) {
    parseArgs(argc, argv, true);
}

Context::~Context() { delete p; }

void Context::applyCommandLine(int argc, const char* const* argv) { parseArgs(argc, argv); }

// parses args
void Context::parseArgs(int argc, const char* const* argv, bool withDefaults) {
    using namespace detail;

    // clang-format off
    parseCommaSepArgs(argc, argv, "dt-source-file=",        p->filters[0]);
    parseCommaSepArgs(argc, argv, "dt-sf=",                 p->filters[0]);
    parseCommaSepArgs(argc, argv, "dt-source-file-exclude=",p->filters[1]);
    parseCommaSepArgs(argc, argv, "dt-sfe=",                p->filters[1]);
    parseCommaSepArgs(argc, argv, "dt-test-suite=",         p->filters[2]);
    parseCommaSepArgs(argc, argv, "dt-ts=",                 p->filters[2]);
    parseCommaSepArgs(argc, argv, "dt-test-suite-exclude=", p->filters[3]);
    parseCommaSepArgs(argc, argv, "dt-tse=",                p->filters[3]);
    parseCommaSepArgs(argc, argv, "dt-test-case=",          p->filters[4]);
    parseCommaSepArgs(argc, argv, "dt-tc=",                 p->filters[4]);
    parseCommaSepArgs(argc, argv, "dt-test-case-exclude=",  p->filters[5]);
    parseCommaSepArgs(argc, argv, "dt-tce=",                p->filters[5]);
    parseCommaSepArgs(argc, argv, "dt-subcase=",            p->filters[6]);
    parseCommaSepArgs(argc, argv, "dt-sc=",                 p->filters[6]);
    parseCommaSepArgs(argc, argv, "dt-subcase-exclude=",    p->filters[7]);
    parseCommaSepArgs(argc, argv, "dt-sce=",                p->filters[7]);
    // clang-format on

    int    intRes = 0;
    String strRes;

#define DOCTEST_PARSE_AS_BOOL_OR_FLAG(name, sname, var, default)                                   \
    if(parseIntOption(argc, argv, name "=", option_bool, intRes) ||                                \
       parseIntOption(argc, argv, sname "=", option_bool, intRes))                                 \
        p->var = !!intRes;                                                                         \
    else if(parseFlag(argc, argv, name) || parseFlag(argc, argv, sname))                           \
        p->var = true;                                                                             \
    else if(withDefaults)                                                                          \
    p->var = default

#define DOCTEST_PARSE_INT_OPTION(name, sname, var, default)                                        \
    if(parseIntOption(argc, argv, name "=", option_int, intRes) ||                                 \
       parseIntOption(argc, argv, sname "=", option_int, intRes))                                  \
        p->var = intRes;                                                                           \
    else if(withDefaults)                                                                          \
    p->var = default

#define DOCTEST_PARSE_STR_OPTION(name, sname, var, default)                                        \
    if(parseOption(argc, argv, name "=", strRes, default) ||                                       \
       parseOption(argc, argv, sname "=", strRes, default) || withDefaults)                        \
    p->var = strRes

    // clang-format off
    DOCTEST_PARSE_STR_OPTION("dt-order-by", "dt-ob", order_by, "file");
    DOCTEST_PARSE_INT_OPTION("dt-rand-seed", "dt-rs", rand_seed, 0);

    DOCTEST_PARSE_INT_OPTION("dt-first", "dt-f", first, 1);
    DOCTEST_PARSE_INT_OPTION("dt-last", "dt-l", last, 0);

    DOCTEST_PARSE_INT_OPTION("dt-abort-after", "dt-aa", abort_after, 0);
    DOCTEST_PARSE_INT_OPTION("dt-subcase-filter-levels", "dt-scfl", subcase_filter_levels, 2000000000);

    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-success", "dt-s", success, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-case-sensitive", "dt-cs", case_sensitive, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-exit", "dt-e", exit, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-duration", "dt-d", duration, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-no-throw", "dt-nt", no_throw, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-no-exitcode", "dt-ne", no_exitcode, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-no-run", "dt-nr", no_run, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-no-version", "dt-nv", no_version, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-no-colors", "dt-nc", no_colors, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-force-colors", "dt-fc", force_colors, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-no-breaks", "dt-nb", no_breaks, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-no-skip", "dt-ns", no_skip, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-no-path-filenames", "dt-npf", no_path_in_filenames, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-no-line-numbers", "dt-nln", no_line_numbers, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("dt-no-skipped-summary", "dt-nss", no_skipped_summary, false);
    // clang-format on

#undef DOCTEST_PARSE_STR_OPTION
#undef DOCTEST_PARSE_INT_OPTION
#undef DOCTEST_PARSE_AS_BOOL_OR_FLAG

    if(withDefaults) {
        p->help             = false;
        p->version          = false;
        p->count            = false;
        p->list_test_cases  = false;
        p->list_test_suites = false;
    }
    if(parseFlag(argc, argv, "dt-help") || parseFlag(argc, argv, "dt-h") ||
       parseFlag(argc, argv, "dt-?")) {
        p->help = true;
        p->exit = true;
    }
    if(parseFlag(argc, argv, "dt-version") || parseFlag(argc, argv, "dt-v")) {
        p->version = true;
        p->exit    = true;
    }
    if(parseFlag(argc, argv, "dt-count") || parseFlag(argc, argv, "dt-c")) {
        p->count = true;
        p->exit  = true;
    }
    if(parseFlag(argc, argv, "dt-list-test-cases") || parseFlag(argc, argv, "dt-ltc")) {
        p->list_test_cases = true;
        p->exit            = true;
    }
    if(parseFlag(argc, argv, "dt-list-test-suites") || parseFlag(argc, argv, "dt-lts")) {
        p->list_test_suites = true;
        p->exit             = true;
    }
}

// allows the user to add procedurally to the filters from the command line
void Context::addFilter(const char* filter, const char* value) { setOption(filter, value); }

// allows the user to clear all filters from the command line
void Context::clearFilters() {
    for(unsigned i = 0; i < p->filters.size(); ++i)
        p->filters[i].clear();
}

// allows the user to override procedurally the int/bool options from the command line
void Context::setOption(const char* option, int value) {
    setOption(option, toString(value).c_str());
}

// allows the user to override procedurally the string options from the command line
void Context::setOption(const char* option, const char* value) {
    String      argv   = String("-") + option + "=" + value;
    const char* lvalue = argv.c_str();
    parseArgs(1, &lvalue);
}

// users should query this in their main() and exit the program if true
bool Context::shouldExit() { return p->exit; }

// the main function that does all the filtering and test running
int Context::run() {
    using namespace detail;

    Color::init();

    contextState = p;
    p->resetRunData();

    // handle version, help and no_run
    if(p->no_run || p->version || p->help) {
        if(p->version)
            printVersion();
        if(p->help)
            printHelp();

        contextState = 0;

        return EXIT_SUCCESS;
    }

    printVersion();
    DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
    std::printf("run with \"--help\" for options\n");

    unsigned i = 0; // counter used for loops - here for VC6

    std::set<TestCase>& registeredTests = getRegisteredTests();

    std::vector<const TestCase*> testArray;
    for(std::set<TestCase>::iterator it = registeredTests.begin(); it != registeredTests.end();
        ++it)
        testArray.push_back(&(*it));

    // sort the collected records
    if(!testArray.empty()) {
        if(p->order_by.compare("file", true) == 0) {
            std::qsort(&testArray[0], testArray.size(), sizeof(TestCase*), fileOrderComparator);
        } else if(p->order_by.compare("suite", true) == 0) {
            std::qsort(&testArray[0], testArray.size(), sizeof(TestCase*), suiteOrderComparator);
        } else if(p->order_by.compare("name", true) == 0) {
            std::qsort(&testArray[0], testArray.size(), sizeof(TestCase*), nameOrderComparator);
        } else if(p->order_by.compare("rand", true) == 0) {
            std::srand(p->rand_seed);

            // random_shuffle implementation
            const TestCase** first = &testArray[0];
            for(i = testArray.size() - 1; i > 0; --i) {
                int idxToSwap = std::rand() % (i + 1); // NOLINT

                const TestCase* temp = first[i];

                first[i]         = first[idxToSwap];
                first[idxToSwap] = temp;
            }
        }
    }

    if(p->list_test_cases) {
        DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
        std::printf("listing all test case names\n");
        DOCTEST_PRINTF_COLORED(getSeparator(), Color::Yellow);
    }

    std::set<String> testSuitesPassingFilters;
    if(p->list_test_suites) {
        DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
        std::printf("listing all test suites\n");
        DOCTEST_PRINTF_COLORED(getSeparator(), Color::Yellow);
    }

    // invoke the registered functions if they match the filter criteria (or just count them)
    for(i = 0; i < testArray.size(); i++) {
        const TestCase& data = *testArray[i];

        if(data.m_skip && !p->no_skip)
            continue;

        if(!matchesAny(data.m_file, p->filters[0], 1, p->case_sensitive))
            continue;
        if(matchesAny(data.m_file, p->filters[1], 0, p->case_sensitive))
            continue;
        if(!matchesAny(data.m_test_suite, p->filters[2], 1, p->case_sensitive))
            continue;
        if(matchesAny(data.m_test_suite, p->filters[3], 0, p->case_sensitive))
            continue;
        if(!matchesAny(data.m_name, p->filters[4], 1, p->case_sensitive))
            continue;
        if(matchesAny(data.m_name, p->filters[5], 0, p->case_sensitive))
            continue;

        p->numTestsPassingFilters++;

        // do not execute the test if we are to only count the number of filter passing tests
        if(p->count)
            continue;

        // print the name of the test and don't execute it
        if(p->list_test_cases) {
            std::printf("%s\n", data.m_name);
            continue;
        }

        // print the name of the test suite if not done already and don't execute it
        if(p->list_test_suites) {
            if((testSuitesPassingFilters.count(data.m_test_suite) == 0) &&
               data.m_test_suite[0] != '\0') {
                std::printf("%s\n", data.m_test_suite);
                testSuitesPassingFilters.insert(data.m_test_suite);
                p->numTestSuitesPassingFilters++;
            }
            continue;
        }

        // skip the test if it is not in the execution range
        if((p->last < p->numTestsPassingFilters && p->first <= p->last) ||
           (p->first > p->numTestsPassingFilters))
            continue;

        // execute the test if it passes all the filtering
        {
            p->currentTest = &data;

            bool failed                              = false;
            p->hasLoggedCurrentTestStart             = false;
            p->numFailedAssertionsForCurrentTestcase = 0;
            p->subcasesPassed.clear();
            double duration = 0;
            Timer  timer;
            timer.start();
            do {
                // if the start has been logged from a previous iteration of this loop
                if(p->hasLoggedCurrentTestStart)
                    logTestEnd();
                p->hasLoggedCurrentTestStart = false;

                // if logging successful tests - force the start log
                if(p->success)
                    DOCTEST_LOG_START();

                // reset the assertion state
                p->numAssertionsForCurrentTestcase = 0;
                p->hasCurrentTestFailed            = false;

                // reset some of the fields for subcases (except for the set of fully passed ones)
                p->subcasesHasSkipped   = false;
                p->subcasesCurrentLevel = 0;
                p->subcasesEnteredLevels.clear();

                // reset stuff for logging with INFO()
                p->exceptionalContexts.clear();

// execute the test
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
                try {
#endif // DOCTEST_CONFIG_NO_EXCEPTIONS
                    FatalConditionHandler fatalConditionHandler; // Handle signals
                    data.m_test();
                    fatalConditionHandler.reset();
                    if(contextState->hasCurrentTestFailed)
                        failed = true;
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
                } catch(const TestFailureException&) { failed = true; } catch(...) {
                    DOCTEST_LOG_START();
                    logTestException(translateActiveException());
                    failed = true;
                }
#endif // DOCTEST_CONFIG_NO_EXCEPTIONS

                p->numAssertions += p->numAssertionsForCurrentTestcase;

                // exit this loop if enough assertions have failed
                if(p->abort_after > 0 && p->numFailedAssertions >= p->abort_after) {
                    p->subcasesHasSkipped = false;
                    DOCTEST_PRINTF_COLORED("Aborting - too many failed asserts!\n", Color::Red);
                }

            } while(p->subcasesHasSkipped == true);

            duration = timer.getElapsedSeconds();

            if(Approx(p->currentTest->m_timeout).epsilon(DBL_EPSILON) != 0 &&
               Approx(duration).epsilon(DBL_EPSILON) > p->currentTest->m_timeout) {
                failed = true;
                DOCTEST_LOG_START();
                char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
                DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg),
                                 "Test case exceeded time limit of %.6f!\n",
                                 p->currentTest->m_timeout);
                DOCTEST_PRINTF_COLORED(msg, Color::Red);
            }

            if(p->duration) {
                char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
                DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), "%.6f s: %s\n", duration,
                                 p->currentTest->m_name);
                DOCTEST_PRINTF_COLORED(msg, Color::None);
            }

            if(data.m_should_fail) {
                DOCTEST_LOG_START();
                if(failed) {
                    failed = false;
                    DOCTEST_PRINTF_COLORED("Failed as expected so marking it as not failed\n",
                                           Color::Yellow);
                } else {
                    failed = true;
                    DOCTEST_PRINTF_COLORED("Should have failed but didn't! Marking it as failed!\n",
                                           Color::Red);
                }
            } else if(failed && data.m_may_fail) {
                DOCTEST_LOG_START();
                failed = false;
                DOCTEST_PRINTF_COLORED("Allowed to fail so marking it as not failed\n",
                                       Color::Yellow);
            } else if(data.m_expected_failures > 0) {
                DOCTEST_LOG_START();
                char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
                if(p->numFailedAssertionsForCurrentTestcase == data.m_expected_failures) {
                    failed = false;
                    DOCTEST_SNPRINTF(
                            msg, DOCTEST_COUNTOF(msg),
                            "Failed exactly %d times as expected so marking it as not failed!\n",
                            data.m_expected_failures);
                    DOCTEST_PRINTF_COLORED(msg, Color::Yellow);
                } else {
                    failed = true;
                    DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg),
                                     "Didn't fail exactly %d times so marking it as failed!\n",
                                     data.m_expected_failures);
                    DOCTEST_PRINTF_COLORED(msg, Color::Red);
                }
            }

            if(p->hasLoggedCurrentTestStart)
                logTestEnd();

            if(failed) // if any subcase has failed - the whole test case has failed
                p->numFailed++;

            // stop executing tests if enough assertions have failed
            if(p->abort_after > 0 && p->numFailedAssertions >= p->abort_after)
                break;
        }
    }

    printSummary();

    contextState = 0;

    if(p->numFailed && !p->no_exitcode)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
} // namespace doctest

#endif // DOCTEST_CONFIG_DISABLE

#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
int main(int argc, char** argv) { return doctest::Context(argc, argv).run(); }
#endif // DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

DOCTEST_CLANG_SUPPRESS_WARNING_POP
DOCTEST_MSVC_SUPPRESS_WARNING_POP
DOCTEST_GCC_SUPPRESS_WARNING_POP

#endif // DOCTEST_LIBRARY_IMPLEMENTATION
#endif // DOCTEST_CONFIG_IMPLEMENT
