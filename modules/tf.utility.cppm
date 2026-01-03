module;

#include <taskflow/utility/iterator.hpp>
#include <taskflow/utility/lazy_string.hpp>
#include <taskflow/utility/math.hpp>
#include <taskflow/utility/mpmc.hpp>
#include <taskflow/utility/object_pool.hpp>
#include <taskflow/utility/os.hpp>
#include <taskflow/utility/serializer.hpp>
#include <taskflow/utility/small_vector.hpp>
#include <taskflow/utility/stream.hpp>
#include <taskflow/utility/traits.hpp>
#include <taskflow/utility/uuid.hpp>

export module tf:utility;

export namespace tf {
    using tf::IndexRange;
    using tf::LazyString;
    using tf::MPMC;
    using tf::ObjectPool;
    using tf::CachelineAligned;
    using tf::is_std_basic_string;
    using tf::is_std_array;
    using tf::is_std_vector;
    using tf::is_std_deque;
    using tf::is_std_list;
    using tf::is_std_forward_list;
    using tf::is_std_map;
    using tf::is_std_unordered_map;
    using tf::is_std_set;
    using tf::is_std_unordered_set;
    using tf::is_std_variant;
    using tf::is_std_optional;
    using tf::is_std_unique_ptr;
    using tf::is_std_shared_ptr;
    using tf::is_std_duration;
    using tf::is_std_time_point;
    using tf::is_std_tuple;
    using tf::SizeTag;
    using tf::MapItem;
    using tf::ExtractType;
    using tf::ExtractType_t;
    using tf::Serializer;
    using tf::Deserializer;
    using tf::IsPod;
    using tf::SmallVectorBase;
    using tf::SmallVectorStorage;
    using tf::SmallVectorTemplateCommon;
    using tf::SmallVectorTemplateBase;
    using tf::SmallVectorImpl;
    using tf::SmallVector;
    using tf::is_pod;
    using tf::NoInit;
    using tf::MoC;
    using tf::get_index;
    using tf::get_index_impl;
    using tf::unwrap_reference;
    using tf::unwrap_reference_t;
    using tf::unwrap_ref_decay;
    using tf::unwrap_ref_decay_t;
    using tf::stateful_iterator;
    using tf::stateful_iterator_t;
    using tf::stateful_index;
    using tf::stateful_index_t;
    using tf::Unroll;
    using tf::filter_duplicates;
    using tf::unique_variant;
    using tf::unique_variant_t;
    using tf::is_std_compare;
    using tf::bool_pack;
    using tf::all_true;
    using tf::all_same;
    using tf::deref_t;
    using tf::UUID;
    
    using tf::is_std_basic_string_v;
    using tf::is_std_array_v;
    using tf::is_std_vector_v;
    using tf::is_std_deque_v;
    using tf::is_std_list_v;
    using tf::is_std_forward_list_v;
    using tf::is_std_map_v;
    using tf::is_std_unordered_map_v;
    using tf::is_std_set_v;
    using tf::is_std_unordered_set_v;
    using tf::is_std_variant_v;
    using tf::is_std_optional_v;
    using tf::is_std_unique_ptr_v;
    using tf::is_std_shared_ptr_v;
    using tf::is_std_duration_v;
    using tf::is_std_time_point_v;
    using tf::is_std_tuple_v;
    using tf::is_default_serializable_v;
    using tf::is_default_deserializable_v;
    using tf::dependent_false_v;
    using tf::is_pod_v;
    using tf::get_index_v;
    using tf::is_std_compare_v;
    using tf::all_same_v;
    using tf::is_random_access_iterator;

    using tf::is_index_range_invalid;
    using tf::distance;
    using tf::next_pow2;
    using tf::is_pow2;
    using tf::floor_log2;
    using tf::static_floor_log2;
    using tf::median_of_three;
    using tf::pseudo_median_of_nine;
    using tf::sort2;
    using tf::sort3;
    using tf::unique_id;
    using tf::atomic_max;
    using tf::atomic_min;
    using tf::seed;
    using tf::coprime;
    using tf::make_coprime_lut;
    using tf::get_env;
    using tf::has_env;
    using tf::pause;
    using tf::spin_until;
    using tf::make_size_tag;
    using tf::make_kv_pair;
    using tf::capacity_in_bytes;
    using tf::ostreamize;
    using tf::stringify;
    using tf::make_moc;
    using tf::visit_tuple;
    using tf::unroll;
    using tf::swap;

    using tf::operator<<;
}

export namespace std {
    using std::hash;
}
