#include <string>
#include <variant>
#include <vector>
#include <type_traits>

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
    using variant_type = std::variant<std::conditional_t<std::is_void_v<typename DPs::output_type>, std::monostate, typename DPs::output_type>...>;
};

int main()
{
    using dp1 = DataPipe<void, int>;
    using dp2 = DataPipe<int, std::string>;
    using dp3 = DataPipe<std::string, float>;
    using dp4 = DataPipe<float, void>;

    using pipeline_t = DataPipeline<dp1, dp2, dp3, dp4>;

    std::vector<pipeline_t::variant_type> buffer(3);
    std::get<int>(buffer[0]);

    return 0;
}

//Hello! I wanna ask a small question, which I didn't find a useful solution on the internet. The question is stated as comments in the code block below.
//Can you give me any suggestions? Many thanks!