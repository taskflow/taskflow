#if defined(DOCTEST_CONFIG_IMPLEMENT) || !defined(DOCTEST_SINGLE_HEADER)
#ifndef DOCTEST_LIBRARY_IMPLEMENTATION
#define DOCTEST_LIBRARY_IMPLEMENTATION

#ifndef DOCTEST_SINGLE_HEADER
#include "doctest_fwd.h"
#endif // DOCTEST_SINGLE_HEADER

DOCTEST_CLANG_SUPPRESS_WARNING_PUSH
DOCTEST_CLANG_SUPPRESS_WARNING("-Wunknown-pragmas")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wpadded")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wweak-vtables")
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
DOCTEST_CLANG_SUPPRESS_WARNING("-Wc++98-compat")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wc++98-compat-pedantic")

DOCTEST_GCC_SUPPRESS_WARNING_PUSH
DOCTEST_GCC_SUPPRESS_WARNING("-Wunknown-pragmas")
DOCTEST_GCC_SUPPRESS_WARNING("-Wpragmas")
DOCTEST_GCC_SUPPRESS_WARNING("-Wconversion")
DOCTEST_GCC_SUPPRESS_WARNING("-Weffc++")
DOCTEST_GCC_SUPPRESS_WARNING("-Wsign-conversion")
DOCTEST_GCC_SUPPRESS_WARNING("-Wstrict-overflow")
DOCTEST_GCC_SUPPRESS_WARNING("-Wstrict-aliasing")
DOCTEST_GCC_SUPPRESS_WARNING("-Wmissing-field-initializers")
DOCTEST_GCC_SUPPRESS_WARNING("-Wmissing-braces")
DOCTEST_GCC_SUPPRESS_WARNING("-Wmissing-declarations")
DOCTEST_GCC_SUPPRESS_WARNING("-Winline")
DOCTEST_GCC_SUPPRESS_WARNING("-Wswitch")
DOCTEST_GCC_SUPPRESS_WARNING("-Wswitch-enum")
DOCTEST_GCC_SUPPRESS_WARNING("-Wswitch-default")
DOCTEST_GCC_SUPPRESS_WARNING("-Wunsafe-loop-optimizations")
DOCTEST_GCC_SUPPRESS_WARNING("-Wold-style-cast")
DOCTEST_GCC_SUPPRESS_WARNING("-Wunused-local-typedefs")
DOCTEST_GCC_SUPPRESS_WARNING("-Wuseless-cast")

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
DOCTEST_MSVC_SUPPRESS_WARNING(5045) // Spectre mitigation stuff
DOCTEST_MSVC_SUPPRESS_WARNING(4626) // assignment operator was implicitly defined as deleted
DOCTEST_MSVC_SUPPRESS_WARNING(5027) // move assignment operator was implicitly defined as deleted
DOCTEST_MSVC_SUPPRESS_WARNING(5026) // move constructor was implicitly defined as deleted
DOCTEST_MSVC_SUPPRESS_WARNING(4625) // copy constructor was implicitly defined as deleted
DOCTEST_MSVC_SUPPRESS_WARNING(4800) // forcing value to bool 'true' or 'false' (performance warning)
// static analysis
DOCTEST_MSVC_SUPPRESS_WARNING(26439) // This kind of function may not throw. Declare it 'noexcept'
DOCTEST_MSVC_SUPPRESS_WARNING(26495) // Always initialize a member variable
DOCTEST_MSVC_SUPPRESS_WARNING(26451) // Arithmetic overflow ...
DOCTEST_MSVC_SUPPRESS_WARNING(26444) // Avoid unnamed objects with custom construction and dtor...

DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_BEGIN

// required includes - will go only in one translation unit!
#include <ctime>
#include <cmath>
#include <climits>
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
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <atomic>
#include <mutex>
#include <set>
#include <map>
#include <exception>
#include <stdexcept>
#ifdef DOCTEST_CONFIG_POSIX_SIGNALS
#include <csignal>
#endif // DOCTEST_CONFIG_POSIX_SIGNALS
#include <cfloat>
#include <cctype>
#include <cstdint>

#ifdef DOCTEST_PLATFORM_MAC
#include <sys/types.h>
#include <unistd.h>
#include <sys/sysctl.h>
#endif // DOCTEST_PLATFORM_MAC

DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_END

// counts the number of elements in a C array
#define DOCTEST_COUNTOF(x) (sizeof(x) / sizeof(x[0]))

#ifdef DOCTEST_CONFIG_DISABLE
#define DOCTEST_BRANCH_ON_DISABLED(if_disabled, if_not_disabled) if_disabled
#else // DOCTEST_CONFIG_DISABLE
#define DOCTEST_BRANCH_ON_DISABLED(if_disabled, if_not_disabled) if_not_disabled
#endif // DOCTEST_CONFIG_DISABLE

#ifndef DOCTEST_CONFIG_OPTIONS_PREFIX
#define DOCTEST_CONFIG_OPTIONS_PREFIX "dt-"
#endif

#ifndef DOCTEST_THREAD_LOCAL
#define DOCTEST_THREAD_LOCAL thread_local
#endif

#ifdef DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
#define DOCTEST_OPTIONS_PREFIX_DISPLAY DOCTEST_CONFIG_OPTIONS_PREFIX
#else
#define DOCTEST_OPTIONS_PREFIX_DISPLAY ""
#endif

namespace doctest {

bool is_running_in_test = false;

namespace {
    using namespace detail;
    // case insensitive strcmp
    int stricmp(const char* a, const char* b) {
        for(;; a++, b++) {
            const int d = tolower(*a) - tolower(*b);
            if(d != 0 || !*a)
                return d;
        }
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
} // namespace

namespace detail {
    void my_memcpy(void* dest, const void* src, unsigned num) { memcpy(dest, src, num); }

    String rawMemoryToString(const void* object, unsigned size) {
        // Reverse order for little endian architectures
        int i = 0, end = static_cast<int>(size), inc = 1;
        if(Endianness::which() == Endianness::Little) {
            i   = end - 1;
            end = inc = -1;
        }

        unsigned const char* bytes = static_cast<unsigned const char*>(object);
        std::ostringstream   oss;
        oss << "0x" << std::setfill('0') << std::hex;
        for(; i != end; i += inc)
            oss << std::setw(2) << static_cast<unsigned>(bytes[i]);
        return oss.str().c_str();
    }

    DOCTEST_THREAD_LOCAL std::ostringstream g_oss;

    std::ostream* getTlsOss() {
        g_oss.clear(); // there shouldn't be anything worth clearing in the flags
        g_oss.str(""); // the slow way of resetting a string stream
        //g_oss.seekp(0); // optimal reset - as seen here: https://stackoverflow.com/a/624291/3162383
        return &g_oss;
    }

    String getTlsOssResult() {
        //g_oss << std::ends; // needed - as shown here: https://stackoverflow.com/a/624291/3162383
        return g_oss.str().c_str();
    }

#ifndef DOCTEST_CONFIG_DISABLE
    // this holds both parameters from the command line and runtime data for tests
    struct ContextState : ContextOptions, TestRunStats, CurrentTestCaseStats
    {
        std::atomic<int> numAssertsForCurrentTestCase_atomic;
        std::atomic<int> numAssertsFailedForCurrentTestCase_atomic;

        std::vector<std::vector<String> > filters = decltype(filters)(9); // 9 different filters

        std::vector<IReporter*> reporters_currently_used;

        const TestCase* currentTest = nullptr;

        assert_handler ah = nullptr;

        std::vector<String> stringifiedContexts; // logging from INFO() due to an exception

        // stuff for subcases
        std::set<SubcaseSignature> subcasesPassed;
        std::set<int>              subcasesEnteredLevels;
        int                        subcasesCurrentLevel;

        void resetRunData() {
            numTestCases                = 0;
            numTestCasesPassingFilters  = 0;
            numTestSuitesPassingFilters = 0;
            numTestCasesFailed          = 0;
            numAsserts                  = 0;
            numAssertsFailed            = 0;
        }
    };

    ContextState*             g_cs = nullptr;
    DOCTEST_THREAD_LOCAL bool g_no_colors; // used to avoid locks for the debug output

#endif // DOCTEST_CONFIG_DISABLE
} // namespace detail

void String::setOnHeap() { *reinterpret_cast<unsigned char*>(&buf[last]) = 128; }
void String::setLast(unsigned in) { buf[last] = char(in); }

void String::copy(const String& other) {
    if(other.isOnStack()) {
        memcpy(buf, other.buf, len);
    } else {
        setOnHeap();
        data.size     = other.data.size;
        data.capacity = data.size + 1;
        data.ptr      = new char[data.capacity];
        memcpy(data.ptr, other.data.ptr, data.size + 1);
    }
}

String::String() {
    buf[0] = '\0';
    setLast();
}

String::~String() {
    if(!isOnStack())
        delete[] data.ptr;
}

String::String(const char* in)
        : String(in, strlen(in)) {}

String::String(const char* in, unsigned in_size) {
    if(in_size <= last) {
        memcpy(buf, in, in_size + 1);
        setLast(last - in_size);
    } else {
        setOnHeap();
        data.size     = in_size;
        data.capacity = data.size + 1;
        data.ptr      = new char[data.capacity];
        memcpy(data.ptr, in, in_size + 1);
    }
}

String::String(const String& other) { copy(other); }

String& String::operator=(const String& other) {
    if(this != &other) {
        if(!isOnStack())
            delete[] data.ptr;

        copy(other);
    }

    return *this;
}

String& String::operator+=(const String& other) {
    const unsigned my_old_size = size();
    const unsigned other_size  = other.size();
    const unsigned total_size  = my_old_size + other_size;
    if(isOnStack()) {
        if(total_size < len) {
            // append to the current stack space
            memcpy(buf + my_old_size, other.c_str(), other_size + 1);
            setLast(last - total_size);
        } else {
            // alloc new chunk
            char* temp = new char[total_size + 1];
            // copy current data to new location before writing in the union
            memcpy(temp, buf, my_old_size); // skip the +1 ('\0') for speed
            // update data in union
            setOnHeap();
            data.size     = total_size;
            data.capacity = data.size + 1;
            data.ptr      = temp;
            // transfer the rest of the data
            memcpy(data.ptr + my_old_size, other.c_str(), other_size + 1);
        }
    } else {
        if(data.capacity > total_size) {
            // append to the current heap block
            data.size = total_size;
            memcpy(data.ptr + my_old_size, other.c_str(), other_size + 1);
        } else {
            // resize
            data.capacity *= 2;
            if(data.capacity <= total_size)
                data.capacity = total_size + 1;
            // alloc new chunk
            char* temp = new char[data.capacity];
            // copy current data to new location before releasing it
            memcpy(temp, data.ptr, my_old_size); // skip the +1 ('\0') for speed
            // release old chunk
            delete[] data.ptr;
            // update the rest of the union members
            data.size = total_size;
            data.ptr  = temp;
            // transfer the rest of the data
            memcpy(data.ptr + my_old_size, other.c_str(), other_size + 1);
        }
    }

    return *this;
}

String String::operator+(const String& other) const { return String(*this) += other; }

String::String(String&& other) {
    memcpy(buf, other.buf, len);
    other.buf[0] = '\0';
    other.setLast();
}

String& String::operator=(String&& other) {
    if(this != &other) {
        if(!isOnStack())
            delete[] data.ptr;
        memcpy(buf, other.buf, len);
        other.buf[0] = '\0';
        other.setLast();
    }
    return *this;
}

char String::operator[](unsigned i) const {
    return const_cast<String*>(this)->operator[](i); // NOLINT
}

char& String::operator[](unsigned i) {
    if(isOnStack())
        return reinterpret_cast<char*>(buf)[i];
    return data.ptr[i];
}

DOCTEST_GCC_SUPPRESS_WARNING_WITH_PUSH("-Wmaybe-uninitialized")
unsigned String::size() const {
    if(isOnStack())
        return last - (unsigned(buf[last]) & 31); // using "last" would work only if "len" is 32
    return data.size;
}
DOCTEST_GCC_SUPPRESS_WARNING_POP

unsigned String::capacity() const {
    if(isOnStack())
        return len;
    return data.capacity;
}

int String::compare(const char* other, bool no_case) const {
    if(no_case)
        return stricmp(c_str(), other);
    return std::strcmp(c_str(), other);
}

int String::compare(const String& other, bool no_case) const {
    return compare(other.c_str(), no_case);
}

// clang-format off
bool operator==(const String& lhs, const String& rhs) { return lhs.compare(rhs) == 0; }
bool operator!=(const String& lhs, const String& rhs) { return lhs.compare(rhs) != 0; }
bool operator< (const String& lhs, const String& rhs) { return lhs.compare(rhs) < 0; }
bool operator> (const String& lhs, const String& rhs) { return lhs.compare(rhs) > 0; }
bool operator<=(const String& lhs, const String& rhs) { return (lhs != rhs) ? lhs.compare(rhs) < 0 : true; }
bool operator>=(const String& lhs, const String& rhs) { return (lhs != rhs) ? lhs.compare(rhs) > 0 : true; }
// clang-format on

std::ostream& operator<<(std::ostream& s, const String& in) { return s << in.c_str(); }

namespace {
    void color_to_stream(std::ostream&, Color::Enum) DOCTEST_BRANCH_ON_DISABLED({}, ;)
} // namespace

namespace Color {
    std::ostream& operator<<(std::ostream& s, Color::Enum code) {
        color_to_stream(s, code);
        return s;
    }
} // namespace Color

// clang-format off
const char* assertString(assertType::Enum at) {
    DOCTEST_MSVC_SUPPRESS_WARNING_WITH_PUSH(4062) // enum 'x' in switch of enum 'y' is not handled
    switch(at) {  //!OCLINT missing default in switch statements
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

        case assertType::DT_WARN_THROWS_WITH        : return "WARN_THROWS_WITH";
        case assertType::DT_CHECK_THROWS_WITH       : return "CHECK_THROWS_WITH";
        case assertType::DT_REQUIRE_THROWS_WITH     : return "REQUIRE_THROWS_WITH";

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
    }
    DOCTEST_MSVC_SUPPRESS_WARNING_POP
    return "";
}
// clang-format on

const char* failureString(assertType::Enum at) {
    if(at & assertType::is_warn) //!OCLINT bitwise operator in conditional
        return "WARNING: ";
    if(at & assertType::is_check) //!OCLINT bitwise operator in conditional
        return "ERROR: ";
    if(at & assertType::is_require) //!OCLINT bitwise operator in conditional
        return "FATAL ERROR: ";
    return "";
}

DOCTEST_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wnull-dereference")
DOCTEST_GCC_SUPPRESS_WARNING_WITH_PUSH("-Wnull-dereference")
// depending on the current options this will remove the path of filenames
const char* removePathFromFilename(const char* file) {
    if(getContextOptions()->no_path_in_filenames) {
        auto back    = std::strrchr(file, '\\');
        auto forward = std::strrchr(file, '/');
        if(back || forward) {
            if(back > forward)
                forward = back;
            return forward + 1;
        }
    }
    return file;
}
DOCTEST_CLANG_SUPPRESS_WARNING_POP
DOCTEST_GCC_SUPPRESS_WARNING_POP

DOCTEST_DEFINE_DEFAULTS(TestCaseData);
DOCTEST_DEFINE_COPIES(TestCaseData);

DOCTEST_DEFINE_DEFAULTS(AssertData);

DOCTEST_DEFINE_DEFAULTS(MessageData);

SubcaseSignature::SubcaseSignature(const char* name, const char* file, int line)
        : m_name(name)
        , m_file(file)
        , m_line(line) {}

DOCTEST_DEFINE_DEFAULTS(SubcaseSignature);
DOCTEST_DEFINE_COPIES(SubcaseSignature);

bool SubcaseSignature::operator<(const SubcaseSignature& other) const {
    if(m_line != other.m_line)
        return m_line < other.m_line;
    if(std::strcmp(m_file, other.m_file) != 0)
        return std::strcmp(m_file, other.m_file) < 0;
    return std::strcmp(m_name, other.m_name) < 0;
}

IContextScope::IContextScope()  = default;
IContextScope::~IContextScope() = default;

DOCTEST_DEFINE_DEFAULTS(ContextOptions);

#ifdef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
String toString(char* in) { return toString(static_cast<const char*>(in)); }
String toString(const char* in) { return String("\"") + (in ? in : "{null string}") + "\""; }
#endif // DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
String toString(bool in) { return in ? "true" : "false"; }
String toString(float in) { return fpToString(in, 5) + "f"; }
String toString(double in) { return fpToString(in, 10); }
String toString(double long in) { return fpToString(in, 15); }

#define DOCTEST_TO_STRING_OVERLOAD(type, fmt)                                                      \
    String toString(type in) {                                                                     \
        char buf[64];                                                                              \
        std::sprintf(buf, fmt, in);                                                                \
        return buf;                                                                                \
    }

DOCTEST_TO_STRING_OVERLOAD(char, "%d")
DOCTEST_TO_STRING_OVERLOAD(char signed, "%d")
DOCTEST_TO_STRING_OVERLOAD(char unsigned, "%u")
DOCTEST_TO_STRING_OVERLOAD(int short, "%d")
DOCTEST_TO_STRING_OVERLOAD(int short unsigned, "%u")
DOCTEST_TO_STRING_OVERLOAD(int, "%d")
DOCTEST_TO_STRING_OVERLOAD(unsigned, "%u")
DOCTEST_TO_STRING_OVERLOAD(int long, "%ld")
DOCTEST_TO_STRING_OVERLOAD(int long unsigned, "%lu")
DOCTEST_TO_STRING_OVERLOAD(int long long, "%lld")
DOCTEST_TO_STRING_OVERLOAD(int long long unsigned, "%llu")

String toString(std::nullptr_t) { return "NULL"; }

Approx::Approx(double value)
        : m_epsilon(static_cast<double>(std::numeric_limits<float>::epsilon()) * 100)
        , m_scale(1.0)
        , m_value(value) {}

DOCTEST_DEFINE_COPIES(Approx);

Approx Approx::operator()(double value) const {
    Approx approx(value);
    approx.epsilon(m_epsilon);
    approx.scale(m_scale);
    return approx;
}

Approx& Approx::epsilon(double newEpsilon) {
    m_epsilon = newEpsilon;
    return *this;
}
Approx& Approx::scale(double newScale) {
    m_scale = newScale;
    return *this;
}

bool operator==(double lhs, const Approx& rhs) {
    // Thanks to Richard Harris for his help refining this formula
    return std::fabs(lhs - rhs.m_value) <
           rhs.m_epsilon * (rhs.m_scale + std::max(std::fabs(lhs), std::fabs(rhs.m_value)));
}
bool operator==(const Approx& lhs, double rhs) { return operator==(rhs, lhs); }
bool operator!=(double lhs, const Approx& rhs) { return !operator==(lhs, rhs); }
bool operator!=(const Approx& lhs, double rhs) { return !operator==(rhs, lhs); }
bool operator<=(double lhs, const Approx& rhs) { return lhs < rhs.m_value || lhs == rhs; }
bool operator<=(const Approx& lhs, double rhs) { return lhs.m_value < rhs || lhs == rhs; }
bool operator>=(double lhs, const Approx& rhs) { return lhs > rhs.m_value || lhs == rhs; }
bool operator>=(const Approx& lhs, double rhs) { return lhs.m_value > rhs || lhs == rhs; }
bool operator<(double lhs, const Approx& rhs) { return lhs < rhs.m_value && lhs != rhs; }
bool operator<(const Approx& lhs, double rhs) { return lhs.m_value < rhs && lhs != rhs; }
bool operator>(double lhs, const Approx& rhs) { return lhs > rhs.m_value && lhs != rhs; }
bool operator>(const Approx& lhs, double rhs) { return lhs.m_value > rhs && lhs != rhs; }

String toString(const Approx& in) {
    return String("Approx( ") + doctest::toString(in.m_value) + " )";
}
const ContextOptions* getContextOptions() { return DOCTEST_BRANCH_ON_DISABLED(nullptr, g_cs); }

} // namespace doctest

#ifdef DOCTEST_CONFIG_DISABLE
namespace doctest {
Context::Context(int, const char* const*) {}
Context::~Context() = default;
void Context::applyCommandLine(int, const char* const*) {}
void Context::addFilter(const char*, const char*) {}
void Context::clearFilters() {}
void Context::setOption(const char*, int) {}
void Context::setOption(const char*, const char*) {}
bool Context::shouldExit() { return false; }
void Context::setAsDefaultForAssertsOutOfTestCases() {}
void Context::setAssertHandler(detail::assert_handler) {}
int  Context::run() { return 0; }

DOCTEST_DEFINE_DEFAULTS(CurrentTestCaseStats);

DOCTEST_DEFINE_DEFAULTS(TestRunStats);

IReporter::~IReporter() = default;

int                         IReporter::get_num_active_contexts() { return 0; }
const IContextScope* const* IReporter::get_active_contexts() { return nullptr; }
int                         IReporter::get_num_stringified_contexts() { return 0; }
const String*               IReporter::get_stringified_contexts() { return nullptr; }

int registerReporter(const char*, int, IReporter*) { return 0; }

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

#if DOCTEST_MSVC || defined(__MINGW32__)
#if DOCTEST_MSVC
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
#include <Windows.h>
#endif
#include <io.h>

DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_END

#else // DOCTEST_PLATFORM_WINDOWS

#include <sys/time.h>

#endif // DOCTEST_PLATFORM_WINDOWS

namespace doctest_detail_test_suite_ns {
// holds the current test suite
doctest::detail::TestSuite& getCurrentTestSuite() {
    static doctest::detail::TestSuite data;
    return data;
}
} // namespace doctest_detail_test_suite_ns

namespace doctest {
namespace {
    using namespace detail;
    typedef std::map<std::pair<int, String>, IReporter*> reporterMap;
    reporterMap&                                         getReporters() {
        static reporterMap data;
        return data;
    }
} // namespace
namespace detail {
#define DOCTEST_ITERATE_THROUGH_REPORTERS(function, ...)                                           \
    for(auto& curr_rep : g_cs->reporters_currently_used)                                           \
    curr_rep->function(__VA_ARGS__)

    DOCTEST_DEFINE_DEFAULTS(TestFailureException);
    DOCTEST_DEFINE_COPIES(TestFailureException);
    bool checkIfShouldThrow(assertType::Enum at) {
        if(at & assertType::is_require) //!OCLINT bitwise operator in conditional
            return true;

        if((at & assertType::is_check) //!OCLINT bitwise operator in conditional
           && getContextOptions()->abort_after > 0 &&
           (g_cs->numAssertsFailed + g_cs->numAssertsFailedForCurrentTestCase_atomic) >=
                   getContextOptions()->abort_after)
            return true;

        return false;
    }

    void throwException() {
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
        throw TestFailureException();
#endif // DOCTEST_CONFIG_NO_EXCEPTIONS
    }
} // namespace detail

namespace {
    using namespace detail;
    // matching of a string against a wildcard mask (case sensitivity configurable) taken from
    // https://www.codeproject.com/Articles/1088/Wildcard-string-compare-globbing
    int wildcmp(const char* str, const char* wild, bool caseSensitive) {
        const char* cp = nullptr;
        const char* mp = nullptr;

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
    bool matchesAny(const char* name, const std::vector<String>& filters, bool matchEmpty,
                    bool caseSensitive) {
        if(filters.empty() && matchEmpty)
            return true;
        for(auto& curr : filters)
            if(wildcmp(name, curr.c_str(), caseSensitive))
                return true;
        return false;
    }

    typedef uint64_t UInt64;

#ifdef DOCTEST_CONFIG_GETCURRENTTICKS
    UInt64           getCurrentTicks() { return DOCTEST_CONFIG_GETCURRENTTICKS(); }
#elif defined(DOCTEST_PLATFORM_WINDOWS)
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
    UInt64 getCurrentTicks() {
        timeval t;
        gettimeofday(&t, nullptr);
        return static_cast<UInt64>(t.tv_sec) * 1000000 + static_cast<UInt64>(t.tv_usec);
    }
#endif // DOCTEST_PLATFORM_WINDOWS

    struct Timer
    {
        void         start() { m_ticks = getCurrentTicks(); }
        unsigned int getElapsedMicroseconds() const {
            return static_cast<unsigned int>(getCurrentTicks() - m_ticks);
        }
        //unsigned int getElapsedMilliseconds() const {
        //    return static_cast<unsigned int>(getElapsedMicroseconds() / 1000);
        //}
        double getElapsedSeconds() const { return getElapsedMicroseconds() / 1000000.0; }

    private:
        UInt64 m_ticks = 0;
    };

    Timer g_timer;
} // namespace
namespace detail {

    Subcase::Subcase(const char* name, const char* file, int line)
            : m_signature(name, file, line) {
        ContextState* s = g_cs;

        // if we have already completed it
        if(s->subcasesPassed.count(m_signature) != 0)
            return;

        // check subcase filters
        if(s->subcasesCurrentLevel < s->subcase_filter_levels) {
            if(!matchesAny(m_signature.m_name, s->filters[6], true, s->case_sensitive))
                return;
            if(matchesAny(m_signature.m_name, s->filters[7], false, s->case_sensitive))
                return;
        }

        // if a Subcase on the same level has already been entered
        if(s->subcasesEnteredLevels.count(s->subcasesCurrentLevel) != 0) {
            s->should_reenter = true;
            return;
        }

        s->subcasesEnteredLevels.insert(s->subcasesCurrentLevel++);
        m_entered = true;

        DOCTEST_ITERATE_THROUGH_REPORTERS(subcase_start, m_signature);
    }

    Subcase::~Subcase() {
        if(m_entered) {
            ContextState* s = g_cs;

            s->subcasesCurrentLevel--;
            // only mark the subcase as passed if no subcases have been skipped
            if(s->should_reenter == false)
                s->subcasesPassed.insert(m_signature);

            DOCTEST_ITERATE_THROUGH_REPORTERS(subcase_end, m_signature);
        }
    }

    Subcase::operator bool() const { return m_entered; }

    Result::Result(bool passed, const String& decomposition)
            : m_passed(passed)
            , m_decomp(decomposition) {}

    DOCTEST_DEFINE_DEFAULTS(Result);
    DOCTEST_DEFINE_COPIES(Result);

    ExpressionDecomposer::ExpressionDecomposer(assertType::Enum at)
            : m_at(at) {}

    DOCTEST_DEFINE_DEFAULTS(ExpressionDecomposer);

    DOCTEST_DEFINE_DEFAULTS(TestSuite);
    DOCTEST_DEFINE_COPIES(TestSuite);

    TestSuite& TestSuite::operator*(const char* in) {
        m_test_suite = in;
        // clear state
        m_description       = nullptr;
        m_skip              = false;
        m_may_fail          = false;
        m_should_fail       = false;
        m_expected_failures = 0;
        m_timeout           = 0;
        return *this;
    }

    TestCase::TestCase(funcType test, const char* file, unsigned line, const TestSuite& test_suite,
                       const char* type, int template_id) {
        m_file              = file;
        m_line              = line;
        m_name              = nullptr;
        m_test_suite        = test_suite.m_test_suite;
        m_description       = test_suite.m_description;
        m_skip              = test_suite.m_skip;
        m_may_fail          = test_suite.m_may_fail;
        m_should_fail       = test_suite.m_should_fail;
        m_expected_failures = test_suite.m_expected_failures;
        m_timeout           = test_suite.m_timeout;

        m_test        = test;
        m_type        = type;
        m_template_id = template_id;
    }

    DOCTEST_DEFINE_DEFAULTS(TestCase);

    TestCase::TestCase(const TestCase& other)
            : TestCaseData() {
        *this = other;
    }

    DOCTEST_MSVC_SUPPRESS_WARNING_WITH_PUSH(26434) // hides a non-virtual function
    DOCTEST_MSVC_SUPPRESS_WARNING(26437)           // Do not slice
    TestCase& TestCase::operator=(const TestCase& other) {
        static_cast<TestCaseData&>(*this) = static_cast<const TestCaseData&>(other);

        m_test        = other.m_test;
        m_type        = other.m_type;
        m_template_id = other.m_template_id;
        m_full_name   = other.m_full_name;

        if(m_template_id != -1)
            m_name = m_full_name.c_str();
        return *this;
    }
    DOCTEST_MSVC_SUPPRESS_WARNING_POP

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

    bool TestCase::operator<(const TestCase& other) const {
        if(m_line != other.m_line)
            return m_line < other.m_line;
        const int file_cmp = std::strcmp(m_file, other.m_file);
        if(file_cmp != 0)
            return file_cmp < 0;
        return m_template_id < other.m_template_id;
    }
} // namespace detail
namespace {
    using namespace detail;
    // for sorting tests by file/line
    int fileOrderComparator(const void* a, const void* b) {
        auto lhs = *static_cast<TestCase* const*>(a);
        auto rhs = *static_cast<TestCase* const*>(b);
#if DOCTEST_MSVC
        // this is needed because MSVC gives different case for drive letters
        // for __FILE__ when evaluated in a header and a source file
        const int res = stricmp(lhs->m_file, rhs->m_file);
#else  // MSVC
        const int res = std::strcmp(lhs->m_file, rhs->m_file);
#endif // MSVC
        if(res != 0)
            return res;
        return static_cast<int>(lhs->m_line) - static_cast<int>(rhs->m_line);
    }

    // for sorting tests by suite/file/line
    int suiteOrderComparator(const void* a, const void* b) {
        auto lhs = *static_cast<TestCase* const*>(a);
        auto rhs = *static_cast<TestCase* const*>(b);

        const int res = std::strcmp(lhs->m_test_suite, rhs->m_test_suite);
        if(res != 0)
            return res;
        return fileOrderComparator(a, b);
    }

    // for sorting tests by name/suite/file/line
    int nameOrderComparator(const void* a, const void* b) {
        auto lhs = *static_cast<TestCase* const*>(a);
        auto rhs = *static_cast<TestCase* const*>(b);

        const int res_name = std::strcmp(lhs->m_name, rhs->m_name);
        if(res_name != 0)
            return res_name;
        return suiteOrderComparator(a, b);
    }

    // all the registered tests
    std::set<TestCase>& getRegisteredTests() {
        static std::set<TestCase> data;
        return data;
    }

#ifdef DOCTEST_CONFIG_COLORS_WINDOWS
    HANDLE g_stdoutHandle;
    WORD   g_origFgAttrs;
    WORD   g_origBgAttrs;
    bool   g_attrsInitted = false;

    int colors_init() {
        if(!g_attrsInitted) {
            g_stdoutHandle = GetStdHandle(STD_OUTPUT_HANDLE);
            g_attrsInitted = true;
            CONSOLE_SCREEN_BUFFER_INFO csbiInfo;
            GetConsoleScreenBufferInfo(g_stdoutHandle, &csbiInfo);
            g_origFgAttrs = csbiInfo.wAttributes & ~(BACKGROUND_GREEN | BACKGROUND_RED |
                                                     BACKGROUND_BLUE | BACKGROUND_INTENSITY);
            g_origBgAttrs = csbiInfo.wAttributes & ~(FOREGROUND_GREEN | FOREGROUND_RED |
                                                     FOREGROUND_BLUE | FOREGROUND_INTENSITY);
        }
        return 0;
    }

    int dumy_init_console_colors = colors_init();
#endif // DOCTEST_CONFIG_COLORS_WINDOWS

    DOCTEST_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated-declarations")
    void color_to_stream(std::ostream& s, Color::Enum code) {
        ((void)s);    // for DOCTEST_CONFIG_COLORS_NONE or DOCTEST_CONFIG_COLORS_WINDOWS
        ((void)code); // for DOCTEST_CONFIG_COLORS_NONE
#ifdef DOCTEST_CONFIG_COLORS_ANSI
        if(g_no_colors ||
           (isatty(STDOUT_FILENO) == false && getContextOptions()->force_colors == false))
            return;

        auto col = "";
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
        s << "\033" << col;
#endif // DOCTEST_CONFIG_COLORS_ANSI

#ifdef DOCTEST_CONFIG_COLORS_WINDOWS
        if(g_no_colors ||
           (isatty(fileno(stdout)) == false && getContextOptions()->force_colors == false))
            return;

#define DOCTEST_SET_ATTR(x) SetConsoleTextAttribute(g_stdoutHandle, x | g_origBgAttrs)

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
            default:                 DOCTEST_SET_ATTR(g_origFgAttrs);
        }
            // clang-format on
#endif // DOCTEST_CONFIG_COLORS_WINDOWS
    }
    DOCTEST_CLANG_SUPPRESS_WARNING_POP

    std::vector<const IExceptionTranslator*>& getExceptionTranslators() {
        static std::vector<const IExceptionTranslator*> data;
        return data;
    }

    String translateActiveException() {
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
        String res;
        auto&  translators = getExceptionTranslators();
        for(auto& curr : translators)
            if(curr->translate(res))
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
} // namespace

namespace detail {
    // used by the macros for registering tests
    int regTest(const TestCase& tc) {
        getRegisteredTests().insert(tc);
        return 0;
    }

    // sets the current test suite
    int setTestSuite(const TestSuite& ts) {
        doctest_detail_test_suite_ns::getCurrentTestSuite() = ts;
        return 0;
    }

#ifdef DOCTEST_PLATFORM_MAC
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
        if(sysctl(mib, DOCTEST_COUNTOF(mib), &info, &size, 0, 0) != 0) {
            fprintf(stderr, "\n** Call to sysctl failed - unable to determine if debugger is "
                            "active **\n\n");
            return false;
        }
        // We're being debugged if the P_TRACED flag is set.
        return ((info.kp_proc.p_flag & P_TRACED) != 0);
    }
#elif DOCTEST_MSVC || defined(__MINGW32__)
    bool isDebuggerActive() { return ::IsDebuggerPresent() != 0; }
#else
    bool isDebuggerActive() { return false; }
#endif // Platform

    void registerExceptionTranslatorImpl(const IExceptionTranslator* et) {
        if(std::find(getExceptionTranslators().begin(), getExceptionTranslators().end(), et) ==
           getExceptionTranslators().end())
            getExceptionTranslators().push_back(et);
    }

    void writeStringToStream(std::ostream* s, const String& str) { *s << str; }

#ifdef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
    void toStream(std::ostream* s, char* in) { *s << in; }
    void toStream(std::ostream* s, const char* in) { *s << in; }
#endif // DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
    void toStream(std::ostream* s, bool in) { *s << std::boolalpha << in << std::noboolalpha; }
    void toStream(std::ostream* s, float in) { *s << in; }
    void toStream(std::ostream* s, double in) { *s << in; }
    void toStream(std::ostream* s, double long in) { *s << in; }

    void toStream(std::ostream* s, char in) { *s << in; }
    void toStream(std::ostream* s, char signed in) { *s << in; }
    void toStream(std::ostream* s, char unsigned in) { *s << in; }
    void toStream(std::ostream* s, int short in) { *s << in; }
    void toStream(std::ostream* s, int short unsigned in) { *s << in; }
    void toStream(std::ostream* s, int in) { *s << in; }
    void toStream(std::ostream* s, int unsigned in) { *s << in; }
    void toStream(std::ostream* s, int long in) { *s << in; }
    void toStream(std::ostream* s, int long unsigned in) { *s << in; }
    void toStream(std::ostream* s, int long long in) { *s << in; }
    void toStream(std::ostream* s, int long long unsigned in) { *s << in; }

    DOCTEST_THREAD_LOCAL std::vector<IContextScope*> g_infoContexts; // for logging with INFO()

    ContextBuilder::ICapture::ICapture()  = default;
    ContextBuilder::ICapture::~ICapture() = default;

    ContextBuilder::Chunk::Chunk()  = default;
    ContextBuilder::Chunk::~Chunk() = default;

    ContextBuilder::Node::Node()  = default;
    ContextBuilder::Node::~Node() = default;

    // steal the contents of the other - acting as a move constructor...
    ContextBuilder::ContextBuilder(ContextBuilder& other)
            : numCaptures(other.numCaptures)
            , head(other.head)
            , tail(other.tail) {
        other.numCaptures = 0;
        other.head        = nullptr;
        other.tail        = nullptr;
        memcpy(stackChunks, other.stackChunks,
               unsigned(int(sizeof(Chunk)) * DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK));
    }

    DOCTEST_GCC_SUPPRESS_WARNING_WITH_PUSH("-Wcast-align")
    void ContextBuilder::stringify(std::ostream* s) const {
        int curr = 0;
        // iterate over small buffer
        while(curr < numCaptures && curr < DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK)
            reinterpret_cast<const ICapture*>(stackChunks[curr++].buf)->toStream(s);
        // iterate over list
        auto curr_elem = head;
        while(curr < numCaptures) {
            reinterpret_cast<const ICapture*>(curr_elem->chunk.buf)->toStream(s);
            curr_elem = curr_elem->next;
            ++curr;
        }
    }
    DOCTEST_GCC_SUPPRESS_WARNING_POP

    ContextBuilder::ContextBuilder() = default;

    ContextBuilder::~ContextBuilder() {
        // free the linked list - the ones on the stack are left as-is
        // no destructors are called at all - there is no need
        while(head) {
            auto next = head->next;
            delete head;
            head = next;
        }
    }

    ContextScope::ContextScope(ContextBuilder& temp)
            : contextBuilder(temp) {
        g_infoContexts.push_back(this);
    }

    DOCTEST_MSVC_SUPPRESS_WARNING_WITH_PUSH(4996) // std::uncaught_exception is deprecated in C++17
    DOCTEST_GCC_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated-declarations")
    DOCTEST_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated-declarations")
    ContextScope::~ContextScope() {
        if(std::uncaught_exception()) {
            std::ostringstream s;
            this->stringify(&s);
            g_cs->stringifiedContexts.push_back(s.str().c_str());
        }
        g_infoContexts.pop_back();
    }
    DOCTEST_CLANG_SUPPRESS_WARNING_POP
    DOCTEST_GCC_SUPPRESS_WARNING_POP
    DOCTEST_MSVC_SUPPRESS_WARNING_POP

    void ContextScope::stringify(std::ostream* s) const { contextBuilder.stringify(s); }
} // namespace detail
namespace {
    using namespace detail;
#if !defined(DOCTEST_CONFIG_POSIX_SIGNALS) && !defined(DOCTEST_CONFIG_WINDOWS_SEH)
    struct FatalConditionHandler
    {
        void reset() {}
    };
#else // DOCTEST_CONFIG_POSIX_SIGNALS || DOCTEST_CONFIG_WINDOWS_SEH

    void reportFatal(const std::string&);

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
            for(size_t i = 0; i < DOCTEST_COUNTOF(signalDefs); ++i) {
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
            exceptionHandlerHandle = nullptr;
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
                exceptionHandlerHandle = nullptr;
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
    PVOID FatalConditionHandler::exceptionHandlerHandle = nullptr;

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
        static struct sigaction oldSigActions[DOCTEST_COUNTOF(signalDefs)];
        static stack_t          oldSigStack;
        static char             altStackMem[4 * SIGSTKSZ];

        static void handleSignal(int sig) {
            const char* name = "<unknown signal>";
            for(std::size_t i = 0; i < DOCTEST_COUNTOF(signalDefs); ++i) {
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
            sigStack.ss_size  = sizeof(altStackMem);
            sigStack.ss_flags = 0;
            sigaltstack(&sigStack, &oldSigStack);
            struct sigaction sa = {};
            sa.sa_handler       = handleSignal; // NOLINT
            sa.sa_flags         = SA_ONSTACK;
            for(std::size_t i = 0; i < DOCTEST_COUNTOF(signalDefs); ++i) {
                sigaction(signalDefs[i].id, &sa, &oldSigActions[i]);
            }
        }

        ~FatalConditionHandler() { reset(); }
        static void reset() {
            if(isSet) {
                // Set signals back to previous values -- hopefully nobody overwrote them in the meantime
                for(std::size_t i = 0; i < DOCTEST_COUNTOF(signalDefs); ++i) {
                    sigaction(signalDefs[i].id, &oldSigActions[i], nullptr);
                }
                // Return the old stack
                sigaltstack(&oldSigStack, nullptr);
                isSet = false;
            }
        }
    };

    bool             FatalConditionHandler::isSet                                      = false;
    struct sigaction FatalConditionHandler::oldSigActions[DOCTEST_COUNTOF(signalDefs)] = {};
    stack_t          FatalConditionHandler::oldSigStack                                = {};
    char             FatalConditionHandler::altStackMem[]                              = {};

#endif // DOCTEST_PLATFORM_WINDOWS
#endif // DOCTEST_CONFIG_POSIX_SIGNALS || DOCTEST_CONFIG_WINDOWS_SEH

} // namespace

namespace {
    using namespace detail;

#ifdef DOCTEST_PLATFORM_WINDOWS
#define DOCTEST_OUTPUT_DEBUG_STRING(text) ::OutputDebugStringA(text)
#else
    // TODO: integration with XCode and other IDEs
#define DOCTEST_OUTPUT_DEBUG_STRING(text)
#endif // Platform

    void addAssert(assertType::Enum at) {
        if((at & assertType::is_warn) == 0) //!OCLINT bitwise operator in conditional
            g_cs->numAssertsForCurrentTestCase_atomic++;
    }

    void addFailedAssert(assertType::Enum at) {
        if((at & assertType::is_warn) == 0) //!OCLINT bitwise operator in conditional
            g_cs->numAssertsFailedForCurrentTestCase_atomic++;
    }

#if defined(DOCTEST_CONFIG_POSIX_SIGNALS) || defined(DOCTEST_CONFIG_WINDOWS_SEH)
    void reportFatal(const std::string& message) {
        g_cs->seconds_so_far += g_timer.getElapsedSeconds();
        g_cs->failure_flags |= TestCaseFailureReason::Crash;
        g_cs->error_string   = message.c_str();
        g_cs->should_reenter = false;

        // TODO: end all currently opened subcases...?

        DOCTEST_ITERATE_THROUGH_REPORTERS(test_case_end, *g_cs);

        g_cs->numTestCasesFailed++;

        DOCTEST_ITERATE_THROUGH_REPORTERS(test_run_end, *g_cs);
    }
#endif // DOCTEST_CONFIG_POSIX_SIGNALS || DOCTEST_CONFIG_WINDOWS_SEH
} // namespace
namespace detail {

    ResultBuilder::ResultBuilder(assertType::Enum at, const char* file, int line, const char* expr,
                                 const char* exception_type) {
        m_test_case      = g_cs->currentTest;
        m_at             = at;
        m_file           = file;
        m_line           = line;
        m_expr           = expr;
        m_failed         = true;
        m_threw          = false;
        m_threw_as       = false;
        m_exception_type = exception_type;
#if DOCTEST_MSVC
        if(m_expr[0] == ' ') // this happens when variadic macros are disabled under MSVC
            ++m_expr;
#endif // MSVC
    }

    DOCTEST_DEFINE_DEFAULTS(ResultBuilder);

    void ResultBuilder::setResult(const Result& res) {
        m_decomp = res.m_decomp;
        m_failed = !res.m_passed;
    }

    void ResultBuilder::translateException() {
        m_threw     = true;
        m_exception = translateActiveException();
    }

    bool ResultBuilder::log() {
        if(m_at & assertType::is_throws) { //!OCLINT bitwise operator in conditional
            m_failed = !m_threw;
        } else if(m_at & assertType::is_throws_as) { //!OCLINT bitwise operator in conditional
            m_failed = !m_threw_as;
        } else if(m_at & assertType::is_throws_with) { //!OCLINT bitwise operator in conditional
            m_failed = m_exception != m_exception_type;
        } else if(m_at & assertType::is_nothrow) { //!OCLINT bitwise operator in conditional
            m_failed = m_threw;
        }

        if(m_exception.size())
            m_exception = String("\"") + m_exception + "\"";

        if(is_running_in_test) {
            addAssert(m_at);
            DOCTEST_ITERATE_THROUGH_REPORTERS(log_assert, *this);

            if(m_failed)
                addFailedAssert(m_at);
        } else if(m_failed) {
            failed_out_of_a_testing_context(*this);
        }

        return m_failed && isDebuggerActive() &&
               !getContextOptions()->no_breaks; // break into debugger
    }

    void ResultBuilder::react() const {
        if(m_failed && checkIfShouldThrow(m_at))
            throwException();
    }

    void failed_out_of_a_testing_context(const AssertData& ad) {
        if(g_cs->ah)
            g_cs->ah(ad);
        else
            std::abort();
    }

    void decomp_assert(assertType::Enum at, const char* file, int line, const char* expr,
                       Result result) {
        bool failed = !result.m_passed;

        // ###################################################################################
        // IF THE DEBUGGER BREAKS HERE - GO 1 LEVEL UP IN THE CALLSTACK FOR THE FAILING ASSERT
        // THIS IS THE EFFECT OF HAVING 'DOCTEST_CONFIG_SUPER_FAST_ASSERTS' DEFINED
        // ###################################################################################
        DOCTEST_ASSERT_OUT_OF_TESTS(result.m_decomp);
        DOCTEST_ASSERT_IN_TESTS(result.m_decomp);
    }

    MessageBuilder::MessageBuilder(const char* file, int line, assertType::Enum severity) {
        m_stream   = getTlsOss();
        m_file     = file;
        m_line     = line;
        m_severity = severity;
    }

    IExceptionTranslator::IExceptionTranslator()  = default;
    IExceptionTranslator::~IExceptionTranslator() = default;

    bool MessageBuilder::log() {
        m_string = getTlsOssResult();
        DOCTEST_ITERATE_THROUGH_REPORTERS(log_message, *this);

        const bool isWarn = m_severity & assertType::is_warn;

        // warn is just a message in this context so we dont treat it as an assert
        if(!isWarn) {
            addAssert(m_severity);
            addFailedAssert(m_severity);
        }

        return isDebuggerActive() && !getContextOptions()->no_breaks && !isWarn; // break
    }

    void MessageBuilder::react() {
        if(m_severity & assertType::is_require) //!OCLINT bitwise operator in conditional
            throwException();
    }

    MessageBuilder::~MessageBuilder() = default;
} // namespace detail
namespace {
    std::mutex g_mutex;

    using namespace detail;
    struct ConsoleReporter : public IReporter
    {
        std::ostream&                 s;
        bool                          hasLoggedCurrentTestStart;
        std::vector<SubcaseSignature> subcasesStack;

        // caching pointers to objects of these types - safe to do
        const ContextOptions* opt;
        const TestCaseData*   tc;

        ConsoleReporter(std::ostream& in)
                : s(in) {}

        // =========================================================================================
        // WHAT FOLLOWS ARE HELPERS USED BY THE OVERRIDES OF THE VIRTUAL METHODS OF THE INTERFACE
        // =========================================================================================

        void separator_to_stream() {
            s << Color::Yellow
              << "============================================================================="
                 "=="
                 "\n";
        }

        void file_line_to_stream(const char* file, int line, const char* tail = "") {
            s << Color::LightGrey << removePathFromFilename(file)
              << (opt->gnu_file_line ? ":" : "(")
              << (opt->no_line_numbers ? 0 : line) // 0 or the real num depending on the option
              << (opt->gnu_file_line ? ":" : "):") << tail;
        }

        const char* getSuccessOrFailString(bool success, assertType::Enum at,
                                           const char* success_str) {
            if(success)
                return success_str;
            return failureString(at);
        }

        Color::Enum getSuccessOrFailColor(bool success, assertType::Enum at) {
            return success ? Color::BrightGreen :
                             (at & assertType::is_warn) ? Color::Yellow : Color::Red;
        }

        void successOrFailColoredStringToStream(bool success, assertType::Enum at,
                                                const char* success_str = "SUCCESS: ") {
            s << getSuccessOrFailColor(success, at)
              << getSuccessOrFailString(success, at, success_str);
        }

        void log_contexts() {
            int num_contexts = get_num_active_contexts();
            if(num_contexts) {
                auto contexts = get_active_contexts();

                s << Color::None << "  logged: ";
                for(int i = 0; i < num_contexts; ++i) {
                    s << (i == 0 ? "" : "          ");
                    contexts[i]->stringify(&s);
                    s << "\n";
                }
            }

            s << "\n";
        }

        void logTestStart() {
            if(hasLoggedCurrentTestStart)
                return;

            separator_to_stream();
            file_line_to_stream(tc->m_file, tc->m_line, "\n");
            if(tc->m_description)
                s << Color::Yellow << "DESCRIPTION: " << Color::None << tc->m_description << "\n";
            if(tc->m_test_suite && tc->m_test_suite[0] != '\0')
                s << Color::Yellow << "TEST SUITE: " << Color::None << tc->m_test_suite << "\n";
            if(strncmp(tc->m_name, "  Scenario:", 11) != 0)
                s << Color::None << "TEST CASE:  ";
            s << Color::None << tc->m_name << "\n";

            for(auto& curr : subcasesStack)
                if(curr.m_name[0] != '\0')
                    s << "  " << curr.m_name << "\n";

            s << "\n";

            hasLoggedCurrentTestStart = true;
        }

        // =========================================================================================
        // WHAT FOLLOWS ARE OVERRIDES OF THE VIRTUAL METHODS OF THE REPORTER INTERFACE
        // =========================================================================================

        void test_run_start(const ContextOptions& o) override { opt = &o; }

        void test_run_end(const TestRunStats& p) override {
            separator_to_stream();

            const bool anythingFailed = p.numTestCasesFailed > 0 || p.numAssertsFailed > 0;
            s << Color::Cyan << "[doctest] " << Color::None << "test cases: " << std::setw(6)
              << p.numTestCasesPassingFilters << " | "
              << ((p.numTestCasesPassingFilters == 0 || anythingFailed) ? Color::None :
                                                                          Color::Green)
              << std::setw(6) << p.numTestCasesPassingFilters - p.numTestCasesFailed << " passed"
              << Color::None << " | " << (p.numTestCasesFailed > 0 ? Color::Red : Color::None)
              << std::setw(6) << p.numTestCasesFailed << " failed" << Color::None << " | ";
            if(opt->no_skipped_summary == false) {
                const int numSkipped = p.numTestCases - p.numTestCasesPassingFilters;
                s << (numSkipped == 0 ? Color::None : Color::Yellow) << std::setw(6) << numSkipped
                  << " skipped" << Color::None;
            }
            s << "\n";
            s << Color::Cyan << "[doctest] " << Color::None << "assertions: " << std::setw(6)
              << p.numAsserts << " | "
              << ((p.numAsserts == 0 || anythingFailed) ? Color::None : Color::Green)
              << std::setw(6) << (p.numAsserts - p.numAssertsFailed) << " passed" << Color::None
              << " | " << (p.numAssertsFailed > 0 ? Color::Red : Color::None) << std::setw(6)
              << p.numAssertsFailed << " failed" << Color::None << " |\n";
            s << Color::Cyan << "[doctest] " << Color::None
              << "Status: " << (p.numTestCasesFailed > 0 ? Color::Red : Color::Green)
              << ((p.numTestCasesFailed > 0) ? "FAILURE!\n" : "SUCCESS!\n") << Color::None;
        }

        void test_case_start(const TestCaseData& in) override {
            hasLoggedCurrentTestStart = false;
            tc                        = &in;
        }

        void test_case_end(const CurrentTestCaseStats& st) override {
            // log the preamble of the test case only if there is something
            // else to print - something other than that an assert has failed
            if(opt->duration ||
               (st.failure_flags && st.failure_flags != TestCaseFailureReason::AssertFailure))
                logTestStart();

            // report test case exceptions and crashes
            bool crashed = st.failure_flags & TestCaseFailureReason::Crash;
            if(crashed || (st.failure_flags & TestCaseFailureReason::Exception)) {
                file_line_to_stream(tc->m_file, tc->m_line, " ");
                successOrFailColoredStringToStream(false, crashed ? assertType::is_require :
                                                                    assertType::is_check);
                s << Color::Red << (crashed ? "test case CRASHED: " : "test case THREW exception: ")
                  << Color::Cyan << st.error_string << "\n";

                int num_stringified_contexts = get_num_stringified_contexts();
                if(num_stringified_contexts) {
                    auto stringified_contexts = get_stringified_contexts();
                    s << Color::None << "  logged: ";
                    for(int i = num_stringified_contexts - 1; i >= 0; --i) {
                        s << (i == num_stringified_contexts - 1 ? "" : "          ")
                          << stringified_contexts[i] << "\n";
                    }
                }
                s << "\n";
            }

            // means the test case will be re-entered because there are untraversed (but discovered) subcases
            if(st.should_reenter)
                return;

            if(opt->duration)
                s << Color::None << std::setprecision(6) << std::fixed << st.seconds_so_far
                  << " s: " << tc->m_name << "\n";

            if(st.failure_flags & TestCaseFailureReason::Timeout)
                s << Color::Red << "Test case exceeded time limit of " << std::setprecision(6)
                  << std::fixed << tc->m_timeout << "!\n";

            if(st.failure_flags & TestCaseFailureReason::ShouldHaveFailedButDidnt) {
                s << Color::Red << "Should have failed but didn't! Marking it as failed!\n";
            } else if(st.failure_flags & TestCaseFailureReason::ShouldHaveFailedAndDid) {
                s << Color::Yellow << "Failed as expected so marking it as not failed\n";
            } else if(st.failure_flags & TestCaseFailureReason::CouldHaveFailedAndDid) {
                s << Color::Yellow << "Allowed to fail so marking it as not failed\n";
            } else if(st.failure_flags & TestCaseFailureReason::DidntFailExactlyNumTimes) {
                s << Color::Red << "Didn't fail exactly " << tc->m_expected_failures
                  << " times so marking it as failed!\n";
            } else if(st.failure_flags & TestCaseFailureReason::FailedExactlyNumTimes) {
                s << Color::Yellow << "Failed exactly " << tc->m_expected_failures
                  << " times as expected so marking it as not failed!\n";
            }
            if(st.failure_flags & TestCaseFailureReason::TooManyFailedAsserts) {
                s << Color::Red << "Aborting - too many failed asserts!\n";
            }
            s << Color::None;
        }

        void subcase_start(const SubcaseSignature& subc) override {
            subcasesStack.push_back(subc);
            hasLoggedCurrentTestStart = false;
        }

        void subcase_end(const SubcaseSignature& /*subc*/) override {
            subcasesStack.pop_back();
            hasLoggedCurrentTestStart = false;
        }

        void log_assert(const AssertData& rb) override {
            if(!rb.m_failed && !opt->success)
                return;

            std::lock_guard<std::mutex> lock(g_mutex);

            logTestStart();

            file_line_to_stream(rb.m_file, rb.m_line, " ");
            successOrFailColoredStringToStream(!rb.m_failed, rb.m_at);
            if((rb.m_at & (assertType::is_throws_as | assertType::is_throws_with)) ==
               0) //!OCLINT bitwise operator in conditional
                s << Color::Cyan << assertString(rb.m_at) << "( " << rb.m_expr << " ) "
                  << Color::None;

            if(rb.m_at & assertType::is_throws) { //!OCLINT bitwise operator in conditional
                s << (rb.m_threw ? "threw as expected!" : "did NOT throw at all!") << "\n";
            } else if(rb.m_at &
                      assertType::is_throws_as) { //!OCLINT bitwise operator in conditional
                s << Color::Cyan << assertString(rb.m_at) << "( " << rb.m_expr << ", "
                  << rb.m_exception_type << " ) " << Color::None
                  << (rb.m_threw ? (rb.m_threw_as ? "threw as expected!" :
                                                    "threw a DIFFERENT exception: ") :
                                   "did NOT throw at all!")
                  << Color::Cyan << rb.m_exception << "\n";
            } else if(rb.m_at &
                      assertType::is_throws_with) { //!OCLINT bitwise operator in conditional
                s << Color::Cyan << assertString(rb.m_at) << "( " << rb.m_expr << ", \""
                  << rb.m_exception_type << "\" ) " << Color::None
                  << (rb.m_threw ? (!rb.m_failed ? "threw as expected!" :
                                                   "threw a DIFFERENT exception: ") :
                                   "did NOT throw at all!")
                  << Color::Cyan << rb.m_exception << "\n";
            } else if(rb.m_at & assertType::is_nothrow) { //!OCLINT bitwise operator in conditional
                s << (rb.m_threw ? "THREW exception: " : "didn't throw!") << Color::Cyan
                  << rb.m_exception << "\n";
            } else {
                s << (rb.m_threw ? "THREW exception: " :
                                   (!rb.m_failed ? "is correct!\n" : "is NOT correct!\n"));
                if(rb.m_threw)
                    s << rb.m_exception << "\n";
                else
                    s << "  values: " << assertString(rb.m_at) << "( " << rb.m_decomp << " )\n";
            }

            log_contexts();
        }

        void log_message(const MessageData& mb) override {
            std::lock_guard<std::mutex> lock(g_mutex);

            logTestStart();

            file_line_to_stream(mb.m_file, mb.m_line, " ");
            s << getSuccessOrFailColor(false, mb.m_severity)
              << getSuccessOrFailString(mb.m_severity & assertType::is_warn, mb.m_severity,
                                        "MESSAGE: ");
            s << Color::None << mb.m_string << "\n";
            log_contexts();
        }

        void test_case_skipped(const TestCaseData&) override {}
    };

    struct Whitespace
    {
        int nrSpaces;
        explicit Whitespace(int nr)
                : nrSpaces(nr) {}
    };

    std::ostream& operator<<(std::ostream& out, const Whitespace& ws) {
        if(ws.nrSpaces != 0)
            out << std::setw(ws.nrSpaces) << ' ';
        return out;
    }

    // extension of the console reporter - with a bunch of helpers for the stdout stream redirection
    struct ConsoleReporterWithHelpers : public ConsoleReporter
    {
        ConsoleReporterWithHelpers(std::ostream& in)
                : ConsoleReporter(in) {}

        void printVersion() {
            if(getContextOptions()->no_version == false)
                s << Color::Cyan << "[doctest] " << Color::None << "doctest version is \""
                  << DOCTEST_VERSION_STR << "\"\n";
        }

        void printIntro() {
            printVersion();
            s << Color::Cyan << "[doctest] " << Color::None
              << "run with \"--" DOCTEST_OPTIONS_PREFIX_DISPLAY "help\" for options\n";
        }

        void printHelp() {
            int sizePrefixDisplay = static_cast<int>(strlen(DOCTEST_OPTIONS_PREFIX_DISPLAY));
            printVersion();
            // clang-format off
            s << Color::Cyan << "[doctest]\n" << Color::None;
            s << Color::Cyan << "[doctest] " << Color::None;
            s << "boolean values: \"1/on/yes/true\" or \"0/off/no/false\"\n";
            s << Color::Cyan << "[doctest] " << Color::None;
            s << "filter  values: \"str1,str2,str3\" (comma separated strings)\n";
            s << Color::Cyan << "[doctest]\n" << Color::None;
            s << Color::Cyan << "[doctest] " << Color::None;
            s << "filters use wildcards for matching strings\n";
            s << Color::Cyan << "[doctest] " << Color::None;
            s << "something passes a filter if any of the strings in a filter matches\n";
#ifndef DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
            s << Color::Cyan << "[doctest]\n" << Color::None;
            s << Color::Cyan << "[doctest] " << Color::None;
            s << "ALL FLAGS, OPTIONS AND FILTERS ALSO AVAILABLE WITH A \"" DOCTEST_CONFIG_OPTIONS_PREFIX "\" PREFIX!!!\n";
#endif
            s << Color::Cyan << "[doctest]\n" << Color::None;
            s << Color::Cyan << "[doctest] " << Color::None;
            s << "Query flags - the program quits after them. Available:\n\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "?,   --" DOCTEST_OPTIONS_PREFIX_DISPLAY "help, -" DOCTEST_OPTIONS_PREFIX_DISPLAY "h                      "
              << Whitespace(sizePrefixDisplay*0) <<  "prints this message\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "v,   --" DOCTEST_OPTIONS_PREFIX_DISPLAY "version                       "
              << Whitespace(sizePrefixDisplay*1) << "prints the version\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "c,   --" DOCTEST_OPTIONS_PREFIX_DISPLAY "count                         "
              << Whitespace(sizePrefixDisplay*1) << "prints the number of matching tests\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "ltc, --" DOCTEST_OPTIONS_PREFIX_DISPLAY "list-test-cases               "
              << Whitespace(sizePrefixDisplay*1) << "lists all matching tests by name\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "lts, --" DOCTEST_OPTIONS_PREFIX_DISPLAY "list-test-suites              "
              << Whitespace(sizePrefixDisplay*1) << "lists all matching test suites\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "lr,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "list-reporters                "
              << Whitespace(sizePrefixDisplay*1) << "lists all registered reporters\n\n";
            // ================================================================================== << 79
            s << Color::Cyan << "[doctest] " << Color::None;
            s << "The available <int>/<string> options/filters are:\n\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "tc,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "test-case=<filters>           "
              << Whitespace(sizePrefixDisplay*1) << "filters     tests by their name\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "tce, --" DOCTEST_OPTIONS_PREFIX_DISPLAY "test-case-exclude=<filters>   "
              << Whitespace(sizePrefixDisplay*1) << "filters OUT tests by their name\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "sf,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "source-file=<filters>         "
              << Whitespace(sizePrefixDisplay*1) << "filters     tests by their file\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "sfe, --" DOCTEST_OPTIONS_PREFIX_DISPLAY "source-file-exclude=<filters> "
              << Whitespace(sizePrefixDisplay*1) << "filters OUT tests by their file\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "ts,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "test-suite=<filters>          "
              << Whitespace(sizePrefixDisplay*1) << "filters     tests by their test suite\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "tse, --" DOCTEST_OPTIONS_PREFIX_DISPLAY "test-suite-exclude=<filters>  "
              << Whitespace(sizePrefixDisplay*1) << "filters OUT tests by their test suite\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "sc,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "subcase=<filters>             "
              << Whitespace(sizePrefixDisplay*1) << "filters     subcases by their name\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "sce, --" DOCTEST_OPTIONS_PREFIX_DISPLAY "subcase-exclude=<filters>     "
              << Whitespace(sizePrefixDisplay*1) << "filters OUT subcases by their name\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "r,   --" DOCTEST_OPTIONS_PREFIX_DISPLAY "reporters=<filters>           "
              << Whitespace(sizePrefixDisplay*1) << "reporters to use (console is default)\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "ob,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "order-by=<string>             "
              << Whitespace(sizePrefixDisplay*1) << "how the tests should be ordered\n";
            s << Whitespace(sizePrefixDisplay*3) << "                                       <string> - by [file/suite/name/rand]\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "rs,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "rand-seed=<int>               "
              << Whitespace(sizePrefixDisplay*1) << "seed for random ordering\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "f,   --" DOCTEST_OPTIONS_PREFIX_DISPLAY "first=<int>                   "
              << Whitespace(sizePrefixDisplay*1) << "the first test passing the filters to\n";
            s << Whitespace(sizePrefixDisplay*3) << "                                       execute - for range-based execution\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "l,   --" DOCTEST_OPTIONS_PREFIX_DISPLAY "last=<int>                    "
              << Whitespace(sizePrefixDisplay*1) << "the last test passing the filters to\n";
            s << Whitespace(sizePrefixDisplay*3) << "                                       execute - for range-based execution\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "aa,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "abort-after=<int>             "
              << Whitespace(sizePrefixDisplay*1) << "stop after <int> failed assertions\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "scfl,--" DOCTEST_OPTIONS_PREFIX_DISPLAY "subcase-filter-levels=<int>   "
              << Whitespace(sizePrefixDisplay*1) << "apply filters for the first <int> levels\n";
            s << Color::Cyan << "\n[doctest] " << Color::None;
            s << "Bool options - can be used like flags and true is assumed. Available:\n\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "s,   --" DOCTEST_OPTIONS_PREFIX_DISPLAY "success=<bool>                "
              << Whitespace(sizePrefixDisplay*1) << "include successful assertions in output\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "cs,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "case-sensitive=<bool>         "
              << Whitespace(sizePrefixDisplay*1) << "filters being treated as case sensitive\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "e,   --" DOCTEST_OPTIONS_PREFIX_DISPLAY "exit=<bool>                   "
              << Whitespace(sizePrefixDisplay*1) << "exits after the tests finish\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "d,   --" DOCTEST_OPTIONS_PREFIX_DISPLAY "duration=<bool>               "
              << Whitespace(sizePrefixDisplay*1) << "prints the time duration of each test\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "nt,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "no-throw=<bool>               "
              << Whitespace(sizePrefixDisplay*1) << "skips exceptions-related assert checks\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "ne,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "no-exitcode=<bool>            "
              << Whitespace(sizePrefixDisplay*1) << "returns (or exits) always with success\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "nr,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "no-run=<bool>                 "
              << Whitespace(sizePrefixDisplay*1) << "skips all runtime doctest operations\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "nv,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "no-version=<bool>             "
              << Whitespace(sizePrefixDisplay*1) << "omit the framework version in the output\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "nc,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "no-colors=<bool>              "
              << Whitespace(sizePrefixDisplay*1) << "disables colors in output\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "fc,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "force-colors=<bool>           "
              << Whitespace(sizePrefixDisplay*1) << "use colors even when not in a tty\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "nb,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "no-breaks=<bool>              "
              << Whitespace(sizePrefixDisplay*1) << "disables breakpoints in debuggers\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "ns,  --" DOCTEST_OPTIONS_PREFIX_DISPLAY "no-skip=<bool>                "
              << Whitespace(sizePrefixDisplay*1) << "don't skip test cases marked as skip\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "gfl, --" DOCTEST_OPTIONS_PREFIX_DISPLAY "gnu-file-line=<bool>          "
              << Whitespace(sizePrefixDisplay*1) << ":n: vs (n): for line numbers in output\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "npf, --" DOCTEST_OPTIONS_PREFIX_DISPLAY "no-path-filenames=<bool>      "
              << Whitespace(sizePrefixDisplay*1) << "only filenames and no paths in output\n";
            s << " -" DOCTEST_OPTIONS_PREFIX_DISPLAY "nln, --" DOCTEST_OPTIONS_PREFIX_DISPLAY "no-line-numbers=<bool>        "
              << Whitespace(sizePrefixDisplay*1) << "0 instead of real line numbers in output\n";
            // ================================================================================== << 79
            // clang-format on

            s << Color::Cyan << "\n[doctest] " << Color::None;
            s << "for more information visit the project documentation\n\n";
        }

        void printRegisteredReporters() {
            printVersion();
            s << Color::Cyan << "[doctest] " << Color::None << "listing all registered reporters\n";
            for(auto& curr : getReporters())
                s << "priority: " << std::setw(5) << curr.first.first
                  << " name: " << curr.first.second << "\n";
        }

        void output_query_results() {
            separator_to_stream();
            if(getContextOptions()->count || getContextOptions()->list_test_cases) {
                s << Color::Cyan << "[doctest] " << Color::None
                  << "unskipped test cases passing the current filters: "
                  << g_cs->numTestCasesPassingFilters << "\n";
            } else if(getContextOptions()->list_test_suites) {
                s << Color::Cyan << "[doctest] " << Color::None
                  << "unskipped test cases passing the current filters: "
                  << g_cs->numTestCasesPassingFilters << "\n";
                s << Color::Cyan << "[doctest] " << Color::None
                  << "test suites with unskipped test cases passing the current filters: "
                  << g_cs->numTestSuitesPassingFilters << "\n";
            }
        }

        void output_query_preamble_test_cases() {
            s << Color::Cyan << "[doctest] " << Color::None << "listing all test case names\n";
            separator_to_stream();
        }

        void output_query_preamble_test_suites() {
            s << Color::Cyan << "[doctest] " << Color::None << "listing all test suites\n";
            separator_to_stream();
        }

        void output_c_string_with_newline(const char* str) { s << Color::None << str << "\n"; }
    };

#ifdef DOCTEST_PLATFORM_WINDOWS
    struct DebugOutputWindowReporter : public ConsoleReporter
    {
        DOCTEST_THREAD_LOCAL static std::ostringstream oss;

        DebugOutputWindowReporter()
                : ConsoleReporter(oss) {}

#define DOCTEST_DEBUG_OUTPUT_WINDOW_REPORTER_OVERRIDE(func, type)                                  \
    void func(type in) override {                                                                  \
        if(isDebuggerActive()) {                                                                   \
            bool with_col = g_no_colors;                                                           \
            g_no_colors   = false;                                                                 \
            ConsoleReporter::func(in);                                                             \
            DOCTEST_OUTPUT_DEBUG_STRING(oss.str().c_str());                                        \
            oss.str("");                                                                           \
            g_no_colors = with_col;                                                                \
        }                                                                                          \
    }

        DOCTEST_DEBUG_OUTPUT_WINDOW_REPORTER_OVERRIDE(test_run_start, const ContextOptions&)
        DOCTEST_DEBUG_OUTPUT_WINDOW_REPORTER_OVERRIDE(test_run_end, const TestRunStats&)
        DOCTEST_DEBUG_OUTPUT_WINDOW_REPORTER_OVERRIDE(test_case_start, const TestCaseData&)
        DOCTEST_DEBUG_OUTPUT_WINDOW_REPORTER_OVERRIDE(test_case_end, const CurrentTestCaseStats&)
        DOCTEST_DEBUG_OUTPUT_WINDOW_REPORTER_OVERRIDE(subcase_start, const SubcaseSignature&)
        DOCTEST_DEBUG_OUTPUT_WINDOW_REPORTER_OVERRIDE(subcase_end, const SubcaseSignature&)
        DOCTEST_DEBUG_OUTPUT_WINDOW_REPORTER_OVERRIDE(log_assert, const AssertData&)
        DOCTEST_DEBUG_OUTPUT_WINDOW_REPORTER_OVERRIDE(log_message, const MessageData&)
        DOCTEST_DEBUG_OUTPUT_WINDOW_REPORTER_OVERRIDE(test_case_skipped, const TestCaseData&)
    };

    DOCTEST_THREAD_LOCAL std::ostringstream DebugOutputWindowReporter::oss;
#endif // DOCTEST_PLATFORM_WINDOWS

    // the implementation of parseFlag()
    bool parseFlagImpl(int argc, const char* const* argv, const char* pattern) {
        for(int i = argc - 1; i >= 0; --i) {
            auto temp = std::strstr(argv[i], pattern);
            if(temp && strlen(temp) == strlen(pattern)) {
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
        // offset (normally 3 for "dt-") to skip prefix
        if(parseFlagImpl(argc, argv, pattern + strlen(DOCTEST_CONFIG_OPTIONS_PREFIX)))
            return true;
#endif // DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
        return parseFlagImpl(argc, argv, pattern);
    }

    // the implementation of parseOption()
    bool parseOptionImpl(int argc, const char* const* argv, const char* pattern, String& res) {
        for(int i = argc - 1; i >= 0; --i) {
            auto temp = std::strstr(argv[i], pattern);
            if(temp) { //!OCLINT prefer early exits and continue
                // eliminate matches in which the chars before the option are not '-'
                bool noBadCharsFound = true;
                auto curr            = argv[i];
                while(curr != temp) {
                    if(*curr++ != '-') {
                        noBadCharsFound = false;
                        break;
                    }
                }
                if(noBadCharsFound && argv[i][0] == '-') {
                    temp += strlen(pattern);
                    const unsigned len = strlen(temp);
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
        // offset (normally 3 for "dt-") to skip prefix
        if(parseOptionImpl(argc, argv, pattern + strlen(DOCTEST_CONFIG_OPTIONS_PREFIX), res))
            return true;
#endif // DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
        return parseOptionImpl(argc, argv, pattern, res);
    }

    // parses a comma separated list of words after a pattern in one of the arguments in argv
    bool parseCommaSepArgs(int argc, const char* const* argv, const char* pattern,
                           std::vector<String>& res) {
        String filtersString;
        if(parseOption(argc, argv, pattern, filtersString)) {
            // tokenize with "," as a separator
            // cppcheck-suppress strtokCalled
            DOCTEST_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated-declarations")
            auto pch = std::strtok(filtersString.c_str(), ","); // modifies the string
            while(pch != nullptr) {
                if(strlen(pch))
                    res.push_back(pch);
                // uses the strtok() internal state to go to the next token
                // cppcheck-suppress strtokCalled
                pch = std::strtok(nullptr, ",");
            }
            DOCTEST_CLANG_SUPPRESS_WARNING_POP
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
            // TODO: change this to use std::stoi or something else! currently it uses undefined behavior - assumes '0' on unsuccessful parse...
            int theInt = std::atoi(parsedValue.c_str()); // NOLINT
            if(theInt != 0) {
                res = theInt; //!OCLINT parameter reassignment
                return true;
            }
        }
        return false;
    }
} // namespace

Context::Context(int argc, const char* const* argv)
        : p(new detail::ContextState) {
    parseArgs(argc, argv, true);
}

Context::~Context() {
    if(g_cs == p)
        g_cs = nullptr;
    delete p;
}

void Context::applyCommandLine(int argc, const char* const* argv) { parseArgs(argc, argv); }

// parses args
void Context::parseArgs(int argc, const char* const* argv, bool withDefaults) {
    using namespace detail;

    // clang-format off
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "source-file=",        p->filters[0]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "sf=",                 p->filters[0]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "source-file-exclude=",p->filters[1]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "sfe=",                p->filters[1]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "test-suite=",         p->filters[2]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "ts=",                 p->filters[2]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "test-suite-exclude=", p->filters[3]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "tse=",                p->filters[3]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "test-case=",          p->filters[4]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "tc=",                 p->filters[4]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "test-case-exclude=",  p->filters[5]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "tce=",                p->filters[5]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "subcase=",            p->filters[6]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "sc=",                 p->filters[6]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "subcase-exclude=",    p->filters[7]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "sce=",                p->filters[7]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "reporters=",          p->filters[8]);
    parseCommaSepArgs(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "r=",                  p->filters[8]);
    // clang-format on

    int    intRes = 0;
    String strRes;

#define DOCTEST_PARSE_AS_BOOL_OR_FLAG(name, sname, var, default)                                   \
    if(parseIntOption(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX name "=", option_bool, intRes) ||  \
       parseIntOption(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX sname "=", option_bool, intRes))   \
        p->var = !!intRes;                                                                         \
    else if(parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX name) ||                           \
            parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX sname))                            \
        p->var = true;                                                                             \
    else if(withDefaults)                                                                          \
    p->var = default

#define DOCTEST_PARSE_INT_OPTION(name, sname, var, default)                                        \
    if(parseIntOption(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX name "=", option_int, intRes) ||   \
       parseIntOption(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX sname "=", option_int, intRes))    \
        p->var = intRes;                                                                           \
    else if(withDefaults)                                                                          \
    p->var = default

#define DOCTEST_PARSE_STR_OPTION(name, sname, var, default)                                        \
    if(parseOption(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX name "=", strRes, default) ||         \
       parseOption(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX sname "=", strRes, default) ||        \
       withDefaults)                                                                               \
    p->var = strRes

    // clang-format off
    DOCTEST_PARSE_STR_OPTION("order-by", "ob", order_by, "file");
    DOCTEST_PARSE_INT_OPTION("rand-seed", "rs", rand_seed, 0);

    DOCTEST_PARSE_INT_OPTION("first", "f", first, 0);
    DOCTEST_PARSE_INT_OPTION("last", "l", last, UINT_MAX);

    DOCTEST_PARSE_INT_OPTION("abort-after", "aa", abort_after, 0);
    DOCTEST_PARSE_INT_OPTION("subcase-filter-levels", "scfl", subcase_filter_levels, 2000000000);

    DOCTEST_PARSE_AS_BOOL_OR_FLAG("success", "s", success, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("case-sensitive", "cs", case_sensitive, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("exit", "e", exit, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("duration", "d", duration, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("no-throw", "nt", no_throw, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("no-exitcode", "ne", no_exitcode, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("no-run", "nr", no_run, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("no-version", "nv", no_version, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("no-colors", "nc", no_colors, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("force-colors", "fc", force_colors, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("no-breaks", "nb", no_breaks, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("no-skip", "ns", no_skip, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("gnu-file-line", "gfl", gnu_file_line, !bool(DOCTEST_MSVC));
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("no-path-filenames", "npf", no_path_in_filenames, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("no-line-numbers", "nln", no_line_numbers, false);
    DOCTEST_PARSE_AS_BOOL_OR_FLAG("no-skipped-summary", "nss", no_skipped_summary, false);
    // clang-format on

    if(withDefaults) {
        p->help             = false;
        p->version          = false;
        p->count            = false;
        p->list_test_cases  = false;
        p->list_test_suites = false;
        p->list_reporters   = false;
    }
    if(parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "help") ||
       parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "h") ||
       parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "?")) {
        p->help = true;
        p->exit = true;
    }
    if(parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "version") ||
       parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "v")) {
        p->version = true;
        p->exit    = true;
    }
    if(parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "count") ||
       parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "c")) {
        p->count = true;
        p->exit  = true;
    }
    if(parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "list-test-cases") ||
       parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "ltc")) {
        p->list_test_cases = true;
        p->exit            = true;
    }
    if(parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "list-test-suites") ||
       parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "lts")) {
        p->list_test_suites = true;
        p->exit             = true;
    }
    if(parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "list-reporters") ||
       parseFlag(argc, argv, DOCTEST_CONFIG_OPTIONS_PREFIX "lr")) {
        p->list_reporters = true;
        p->exit           = true;
    }
}

// allows the user to add procedurally to the filters from the command line
void Context::addFilter(const char* filter, const char* value) { setOption(filter, value); }

// allows the user to clear all filters from the command line
void Context::clearFilters() {
    for(auto& curr : p->filters)
        curr.clear();
}

// allows the user to override procedurally the int/bool options from the command line
void Context::setOption(const char* option, int value) {
    setOption(option, toString(value).c_str());
}

// allows the user to override procedurally the string options from the command line
void Context::setOption(const char* option, const char* value) {
    auto argv   = String("-") + option + "=" + value;
    auto lvalue = argv.c_str();
    parseArgs(1, &lvalue);
}

// users should query this in their main() and exit the program if true
bool Context::shouldExit() { return p->exit; }

void Context::setAsDefaultForAssertsOutOfTestCases() { g_cs = p; }

void Context::setAssertHandler(detail::assert_handler ah) { p->ah = ah; }

// the main function that does all the filtering and test running
int Context::run() {
    using namespace detail;

    auto old_cs        = g_cs;
    g_cs               = p;
    is_running_in_test = true;
    g_no_colors        = p->no_colors;
    p->resetRunData();

    ConsoleReporterWithHelpers con_rep(std::cout);
    registerReporter("console", 0, con_rep);

    // setup default reporter if none is given through the command line
    p->reporters_currently_used.clear();
    if(p->filters[8].empty())
        p->reporters_currently_used.push_back(getReporters()[reporterMap::key_type(0, "console")]);

    // check to see if any of the registered reporters has been selected
    for(auto& curr : getReporters()) {
        if(matchesAny(curr.first.second.c_str(), p->filters[8], false, p->case_sensitive))
            p->reporters_currently_used.push_back(curr.second);
    }

    // always use the debug output window reporter
#ifdef DOCTEST_PLATFORM_WINDOWS
    DebugOutputWindowReporter debug_output_rep;
    if(isDebuggerActive())
        p->reporters_currently_used.push_back(&debug_output_rep);
#endif // DOCTEST_PLATFORM_WINDOWS

    // handle version, help and no_run
    if(p->no_run || p->version || p->help || p->list_reporters) {
        if(p->version)
            con_rep.printVersion();
        if(p->help)
            con_rep.printHelp();
        if(p->list_reporters)
            con_rep.printRegisteredReporters();

        g_cs               = old_cs;
        is_running_in_test = false;

        return EXIT_SUCCESS;
    }

    con_rep.printIntro();

    std::vector<const TestCase*> testArray;
    for(auto& curr : getRegisteredTests())
        testArray.push_back(&curr);
    p->numTestCases = testArray.size();

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
            const auto first = &testArray[0];
            for(size_t i = testArray.size() - 1; i > 0; --i) {
                int idxToSwap = std::rand() % (i + 1); // NOLINT

                const auto temp = first[i];

                first[i]         = first[idxToSwap];
                first[idxToSwap] = temp;
            }
        }
    }

    if(p->list_test_cases)
        con_rep.output_query_preamble_test_cases();

    std::set<String> testSuitesPassingFilt;
    if(p->list_test_suites)
        con_rep.output_query_preamble_test_suites();

    bool query_mode = p->count || p->list_test_cases || p->list_test_suites;

    if(!query_mode)
        DOCTEST_ITERATE_THROUGH_REPORTERS(test_run_start, *g_cs);

    // invoke the registered functions if they match the filter criteria (or just count them)
    for(auto& curr : testArray) {
        const auto tc = *curr;

        bool skip_me = false;
        if(tc.m_skip && !p->no_skip)
            skip_me = true;

        if(!matchesAny(tc.m_file, p->filters[0], true, p->case_sensitive))
            skip_me = true;
        if(matchesAny(tc.m_file, p->filters[1], false, p->case_sensitive))
            skip_me = true;
        if(!matchesAny(tc.m_test_suite, p->filters[2], true, p->case_sensitive))
            skip_me = true;
        if(matchesAny(tc.m_test_suite, p->filters[3], false, p->case_sensitive))
            skip_me = true;
        if(!matchesAny(tc.m_name, p->filters[4], true, p->case_sensitive))
            skip_me = true;
        if(matchesAny(tc.m_name, p->filters[5], false, p->case_sensitive))
            skip_me = true;

        if(!skip_me)
            p->numTestCasesPassingFilters++;

        // skip the test if it is not in the execution range
        if((p->last < p->numTestCasesPassingFilters && p->first <= p->last) ||
           (p->first > p->numTestCasesPassingFilters))
            skip_me = true;

        if(skip_me) {
            if(!query_mode)
                DOCTEST_ITERATE_THROUGH_REPORTERS(test_case_skipped, tc);
            continue;
        }

        // do not execute the test if we are to only count the number of filter passing tests
        if(p->count)
            continue;

        // print the name of the test and don't execute it
        if(p->list_test_cases) {
            con_rep.output_c_string_with_newline(tc.m_name);
            continue;
        }

        // print the name of the test suite if not done already and don't execute it
        if(p->list_test_suites) {
            if((testSuitesPassingFilt.count(tc.m_test_suite) == 0) && tc.m_test_suite[0] != '\0') {
                con_rep.output_c_string_with_newline(tc.m_test_suite);
                testSuitesPassingFilt.insert(tc.m_test_suite);
                p->numTestSuitesPassingFilters++;
            }
            continue;
        }

        // execute the test if it passes all the filtering
        {
            p->currentTest = &tc;

            p->failure_flags  = TestCaseFailureReason::None;
            p->seconds_so_far = 0;
            p->error_string   = "";

            // reset non-atomic counters
            p->numAssertsFailedForCurrentTestCase = 0;
            p->numAssertsForCurrentTestCase       = 0;

            p->subcasesPassed.clear();
            do {
                // reset some of the fields for subcases (except for the set of fully passed ones)
                p->should_reenter       = false;
                p->subcasesCurrentLevel = 0;
                p->subcasesEnteredLevels.clear();

                // reset stuff for logging with INFO()
                p->stringifiedContexts.clear();

                // reset atomic counters
                p->numAssertsFailedForCurrentTestCase_atomic = 0;
                p->numAssertsForCurrentTestCase_atomic       = 0;

                DOCTEST_ITERATE_THROUGH_REPORTERS(test_case_start, tc);

                g_timer.start();

#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
                try {
#endif // DOCTEST_CONFIG_NO_EXCEPTIONS
                    FatalConditionHandler fatalConditionHandler; // Handle signals
                    // execute the test
                    tc.m_test();
                    fatalConditionHandler.reset();
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
                } catch(const TestFailureException&) {
                    p->failure_flags |= TestCaseFailureReason::AssertFailure;
                } catch(...) {
                    p->error_string = translateActiveException();
                    p->failure_flags |= TestCaseFailureReason::Exception;
                }
#endif // DOCTEST_CONFIG_NO_EXCEPTIONS

                p->seconds_so_far += g_timer.getElapsedSeconds();

                // update the non-atomic counters
                p->numAsserts += p->numAssertsForCurrentTestCase_atomic;
                p->numAssertsForCurrentTestCase += p->numAssertsForCurrentTestCase_atomic;
                p->numAssertsFailed += p->numAssertsFailedForCurrentTestCase_atomic;
                p->numAssertsFailedForCurrentTestCase +=
                        p->numAssertsFailedForCurrentTestCase_atomic;

                // exit this loop if enough assertions have failed - even if there are more subcases
                if(p->abort_after > 0 && p->numAssertsFailed >= p->abort_after) {
                    p->should_reenter = false;
                    p->failure_flags |= TestCaseFailureReason::TooManyFailedAsserts;
                }

                // call it from here only if we will continue looping for other subcases and
                // call it again outside of the loop for one final time - with updated flags
                if(p->should_reenter == true) {
                    DOCTEST_ITERATE_THROUGH_REPORTERS(test_case_end, *g_cs);
                    // remove these flags - it is expected that the reporters have handled these issues
                    p->failure_flags &= ~TestCaseFailureReason::Exception;
                    p->failure_flags &= ~TestCaseFailureReason::AssertFailure;
                }
            } while(p->should_reenter == true);

            if(p->numAssertsFailedForCurrentTestCase)
                p->failure_flags |= TestCaseFailureReason::AssertFailure;

            if(Approx(p->currentTest->m_timeout).epsilon(DBL_EPSILON) != 0 &&
               Approx(p->seconds_so_far).epsilon(DBL_EPSILON) > p->currentTest->m_timeout)
                p->failure_flags |= TestCaseFailureReason::Timeout;

            if(tc.m_should_fail) {
                if(p->failure_flags) {
                    p->failure_flags |= TestCaseFailureReason::ShouldHaveFailedAndDid;
                } else {
                    p->failure_flags |= TestCaseFailureReason::ShouldHaveFailedButDidnt;
                }
            } else if(p->failure_flags && tc.m_may_fail) {
                p->failure_flags |= TestCaseFailureReason::CouldHaveFailedAndDid;
            } else if(tc.m_expected_failures > 0) {
                if(p->numAssertsFailedForCurrentTestCase == tc.m_expected_failures) {
                    p->failure_flags |= TestCaseFailureReason::FailedExactlyNumTimes;
                } else {
                    p->failure_flags |= TestCaseFailureReason::DidntFailExactlyNumTimes;
                }
            }

            bool ok_to_fail = (TestCaseFailureReason::ShouldHaveFailedAndDid & p->failure_flags) ||
                              (TestCaseFailureReason::CouldHaveFailedAndDid & p->failure_flags) ||
                              (TestCaseFailureReason::FailedExactlyNumTimes & p->failure_flags);

            // if any subcase has failed - the whole test case has failed
            if(p->failure_flags && !ok_to_fail)
                p->numTestCasesFailed++;

            DOCTEST_ITERATE_THROUGH_REPORTERS(test_case_end, *g_cs);

            p->currentTest = nullptr;

            // stop executing tests if enough assertions have failed
            if(p->abort_after > 0 && p->numAssertsFailed >= p->abort_after)
                break;
        }
    }

    if(!query_mode)
        DOCTEST_ITERATE_THROUGH_REPORTERS(test_run_end, *g_cs);
    else
        con_rep.output_query_results();

    g_cs               = old_cs;
    is_running_in_test = false;

    if(p->numTestCasesFailed && !p->no_exitcode)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}

DOCTEST_DEFINE_DEFAULTS(CurrentTestCaseStats);

DOCTEST_DEFINE_DEFAULTS(TestRunStats);

IReporter::~IReporter() = default;

int IReporter::get_num_active_contexts() { return detail::g_infoContexts.size(); }
const IContextScope* const* IReporter::get_active_contexts() {
    return get_num_active_contexts() ? &detail::g_infoContexts[0] : nullptr;
}

int IReporter::get_num_stringified_contexts() { return detail::g_cs->stringifiedContexts.size(); }
const String* IReporter::get_stringified_contexts() {
    return get_num_stringified_contexts() ? &detail::g_cs->stringifiedContexts[0] : nullptr;
}

int registerReporter(const char* name, int priority, IReporter& r) {
    getReporters().insert(reporterMap::value_type(reporterMap::key_type(priority, name), &r));
    return 0;
}

// see these issues on the reasoning for this:
// - https://github.com/onqtam/doctest/issues/143#issuecomment-414418903
// - https://github.com/onqtam/doctest/issues/126
void DOCTEST_FIX_FOR_MACOS_LIBCPP_IOSFWD_STRING_LINK_ERRORS() { std::cout << std::string(); }

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
