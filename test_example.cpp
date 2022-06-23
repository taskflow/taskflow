#include <string>
#include <variant>
#include <vector>
#include <type_traits>

template <typename T, typename... Ts>
struct filter_duplicates { using type = T; };

template <template <typename...> class C, typename... Ts, typename U, typename... Us>
struct filter_duplicates<C<Ts...>, U, Us...>
    : std::conditional_t<(std::is_same_v<U, Ts> || ...)
                       , filter_duplicates<C<Ts...>, Us...>
                       , filter_duplicates<C<Ts..., U>, Us...>> {};

template <typename T>
struct unique_variant;

template <typename... Ts>
struct unique_variant<std::variant<Ts...>> : filter_duplicates<std::variant<>, Ts...> {};

template <typename T>
using unique_variant_t = typename unique_variant<T>::type;



template <typename Input, typename Output>
class DataPipe {
public:
    using input_type = Input;
    using output_type = Output;
};

template <typename... DPs>
class DataPipeline {
public:
    //I want to exclude the last element in DPs.
    //Because if there is a `void` type in the variant, it will lead to error.
    //pseudocode of my goal: using variant_type = std::variant<typename DPs::output_type[:-1]>;
    // using variant_type = std::variant<std::conditional_t<std::is_void_v<typename DPs::output_type>, std::monostate, typename DPs::output_type>...>;
    using variant_type = unique_variant_t<std::variant<int, float>>;
};

int main()
{
    using dp1 = DataPipe<void, int>;
    using dp2 = DataPipe<int, std::string>;
    using dp3 = DataPipe<std::string, int>;
    using dp4 = DataPipe<int, void>;

    using pipeline_t = DataPipeline<dp1, dp2, dp3, dp4>;

    std::vector<pipeline_t::variant_type> buffer(3);
    std::get<int>(buffer[0]);
    
    return 0;
}