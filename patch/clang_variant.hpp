// <variant> -*- C++ -*-

// Copyright (C) 2016-2018 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

/** @file variant
 *  This is the <variant> C++ Library header.
 */

#ifndef _GLIBCXX_VARIANT
#define _GLIBCXX_VARIANT 1

#pragma GCC system_header

#if __cplusplus >= 201703L

#include <type_traits>
#include <utility>
#include <bits/enable_special_members.h>
#include <bits/functexcept.h>
#include <bits/move.h>
#include <bits/functional_hash.h>
#include <bits/invoke.h>
#include <ext/aligned_buffer.h>
#include <bits/parse_numbers.h>
#include <bits/stl_iterator_base_types.h>
#include <bits/stl_iterator_base_funcs.h>
#include <bits/stl_construct.h>

namespace std _GLIBCXX_VISIBILITY(default)
{
_GLIBCXX_BEGIN_NAMESPACE_VERSION

namespace __detail
{
namespace __variant
{
  template<size_t _Np, typename... _Types>
    struct _Nth_type;

  template<size_t _Np, typename _First, typename... _Rest>
    struct _Nth_type<_Np, _First, _Rest...>
    : _Nth_type<_Np-1, _Rest...> { };

  template<typename _First, typename... _Rest>
    struct _Nth_type<0, _First, _Rest...>
    { using type = _First; };

} // namespace __variant
} // namespace __detail

#define __cpp_lib_variant 201603

  template<typename... _Types> class tuple;
  template<typename... _Types> class variant;
  template <typename> struct hash;

  template<typename _Variant>
    struct variant_size;

  template<typename _Variant>
    struct variant_size<const _Variant> : variant_size<_Variant> {};

  template<typename _Variant>
    struct variant_size<volatile _Variant> : variant_size<_Variant> {};

  template<typename _Variant>
    struct variant_size<const volatile _Variant> : variant_size<_Variant> {};

  template<typename... _Types>
    struct variant_size<variant<_Types...>>
    : std::integral_constant<size_t, sizeof...(_Types)> {};

  template<typename _Variant>
    inline constexpr size_t variant_size_v = variant_size<_Variant>::value;

  template<size_t _Np, typename _Variant>
    struct variant_alternative;

  template<size_t _Np, typename _First, typename... _Rest>
    struct variant_alternative<_Np, variant<_First, _Rest...>>
    : variant_alternative<_Np-1, variant<_Rest...>> {};

  template<typename _First, typename... _Rest>
    struct variant_alternative<0, variant<_First, _Rest...>>
    { using type = _First; };

  template<size_t _Np, typename _Variant>
    using variant_alternative_t =
      typename variant_alternative<_Np, _Variant>::type;

  template<size_t _Np, typename _Variant>
    struct variant_alternative<_Np, const _Variant>
    { using type = add_const_t<variant_alternative_t<_Np, _Variant>>; };

  template<size_t _Np, typename _Variant>
    struct variant_alternative<_Np, volatile _Variant>
    { using type = add_volatile_t<variant_alternative_t<_Np, _Variant>>; };

  template<size_t _Np, typename _Variant>
    struct variant_alternative<_Np, const volatile _Variant>
    { using type = add_cv_t<variant_alternative_t<_Np, _Variant>>; };

  inline constexpr size_t variant_npos = -1;

  template<size_t _Np, typename... _Types>
    constexpr variant_alternative_t<_Np, variant<_Types...>>&
    get(variant<_Types...>&);

  template<size_t _Np, typename... _Types>
    constexpr variant_alternative_t<_Np, variant<_Types...>>&&
    get(variant<_Types...>&&);

  template<size_t _Np, typename... _Types>
    constexpr variant_alternative_t<_Np, variant<_Types...>> const&
    get(const variant<_Types...>&);

  template<size_t _Np, typename... _Types>
    constexpr variant_alternative_t<_Np, variant<_Types...>> const&&
    get(const variant<_Types...>&&);

namespace __detail
{
namespace __variant
{
  // Returns the first apparence of _Tp in _Types.
  // Returns sizeof...(_Types) if _Tp is not in _Types.
  template<typename _Tp, typename... _Types>
    struct __index_of : std::integral_constant<size_t, 0> {};

  template<typename _Tp, typename... _Types>
    inline constexpr size_t __index_of_v = __index_of<_Tp, _Types...>::value;

  template<typename _Tp, typename _First, typename... _Rest>
    struct __index_of<_Tp, _First, _Rest...> :
      std::integral_constant<size_t, is_same_v<_Tp, _First>
	? 0 : __index_of_v<_Tp, _Rest...> + 1> {};

  // _Uninitialized<T> is guaranteed to be a literal type, even if T is not.
  // We have to do this, because [basic.types]p10.5.3 (n4606) is not implemented
  // yet. When it's implemented, _Uninitialized<T> can be changed to the alias
  // to T, therefore equivalent to being removed entirely.
  //
  // Another reason we may not want to remove _Uninitialzied<T> may be that, we
  // want _Uninitialized<T> to be trivially destructible, no matter whether T
  // is; but we will see.
  template<typename _Type, bool = std::is_literal_type_v<_Type>>
    struct _Uninitialized;

  template<typename _Type>
    struct _Uninitialized<_Type, true>
    {
      template<typename... _Args>
      constexpr _Uninitialized(in_place_index_t<0>, _Args&&... __args)
      : _M_storage(std::forward<_Args>(__args)...)
      { }

      constexpr const _Type& _M_get() const &
      { return _M_storage; }

      constexpr _Type& _M_get() &
      { return _M_storage; }

      constexpr const _Type&& _M_get() const &&
      { return std::move(_M_storage); }

      constexpr _Type&& _M_get() &&
      { return std::move(_M_storage); }

      _Type _M_storage;
    };

  template<typename _Type>
    struct _Uninitialized<_Type, false>
    {
      template<typename... _Args>
      constexpr _Uninitialized(in_place_index_t<0>, _Args&&... __args)
      { ::new (&_M_storage) _Type(std::forward<_Args>(__args)...); }

      const _Type& _M_get() const &
      { return *_M_storage._M_ptr(); }

      _Type& _M_get() &
      { return *_M_storage._M_ptr(); }

      const _Type&& _M_get() const &&
      { return std::move(*_M_storage._M_ptr()); }

      _Type&& _M_get() &&
      { return std::move(*_M_storage._M_ptr()); }

      __gnu_cxx::__aligned_membuf<_Type> _M_storage;
    };

  template<typename _Ref>
    _Ref __ref_cast(void* __ptr)
    {
      return static_cast<_Ref>(*static_cast<remove_reference_t<_Ref>*>(__ptr));
    }

  template<typename _Union>
    constexpr decltype(auto) __get(in_place_index_t<0>, _Union&& __u)
    { return std::forward<_Union>(__u)._M_first._M_get(); }

  template<size_t _Np, typename _Union>
    constexpr decltype(auto) __get(in_place_index_t<_Np>, _Union&& __u)
    {
      return __variant::__get(in_place_index<_Np-1>,
			      std::forward<_Union>(__u)._M_rest);
    }

  // Returns the typed storage for __v.
  template<size_t _Np, typename _Variant>
    constexpr decltype(auto) __get(_Variant&& __v)
    {
      return __variant::__get(std::in_place_index<_Np>,
			      std::forward<_Variant>(__v)._M_u);
    }

  // Various functions as "vtable" entries, where those vtables are used by
  // polymorphic operations.
  template<typename _Lhs, typename _Rhs>
    void
    __erased_ctor(void* __lhs, void* __rhs)
    {
      using _Type = remove_reference_t<_Lhs>;
      ::new (__lhs) _Type(__variant::__ref_cast<_Rhs>(__rhs));
    }

  template<typename _Variant, size_t _Np>
    void
    __erased_dtor(_Variant&& __v)
    { std::_Destroy(std::__addressof(__get<_Np>(__v))); }

  template<typename _Lhs, typename _Rhs>
    void
    __erased_assign(void* __lhs, void* __rhs)
    {
      __variant::__ref_cast<_Lhs>(__lhs) = __variant::__ref_cast<_Rhs>(__rhs);
    }

  template<typename _Lhs, typename _Rhs>
    void
    __erased_swap(void* __lhs, void* __rhs)
    {
      using std::swap;
      swap(__variant::__ref_cast<_Lhs>(__lhs),
	   __variant::__ref_cast<_Rhs>(__rhs));
    }

#define _VARIANT_RELATION_FUNCTION_TEMPLATE(__OP, __NAME) \
  template<typename _Variant, size_t _Np> \
    constexpr bool \
    __erased_##__NAME(const _Variant& __lhs, const _Variant& __rhs) \
    { \
      return __get<_Np>(std::forward<_Variant>(__lhs)) \
	  __OP __get<_Np>(std::forward<_Variant>(__rhs)); \
    }

  _VARIANT_RELATION_FUNCTION_TEMPLATE(<, less)
  _VARIANT_RELATION_FUNCTION_TEMPLATE(<=, less_equal)
  _VARIANT_RELATION_FUNCTION_TEMPLATE(==, equal)
  _VARIANT_RELATION_FUNCTION_TEMPLATE(!=, not_equal)
  _VARIANT_RELATION_FUNCTION_TEMPLATE(>=, greater_equal)
  _VARIANT_RELATION_FUNCTION_TEMPLATE(>, greater)

#undef _VARIANT_RELATION_FUNCTION_TEMPLATE

  template<typename _Tp>
    size_t
    __erased_hash(void* __t)
    {
      return std::hash<remove_cv_t<remove_reference_t<_Tp>>>{}(
	  __variant::__ref_cast<_Tp>(__t));
    }

  template<typename... _Types>
    struct _Traits
    {
      static constexpr bool _S_default_ctor =
	  is_default_constructible_v<typename _Nth_type<0, _Types...>::type>;
      static constexpr bool _S_copy_ctor =
	  (is_copy_constructible_v<_Types> && ...);
      static constexpr bool _S_move_ctor =
	  (is_move_constructible_v<_Types> && ...);
      static constexpr bool _S_copy_assign =
	  _S_copy_ctor && _S_move_ctor
	  && (is_copy_assignable_v<_Types> && ...);
      static constexpr bool _S_move_assign =
	  _S_move_ctor
	  && (is_move_assignable_v<_Types> && ...);

      static constexpr bool _S_trivial_dtor =
	  (is_trivially_destructible_v<_Types> && ...);
      static constexpr bool _S_trivial_copy_ctor =
	  (is_trivially_copy_constructible_v<_Types> && ...);
      static constexpr bool _S_trivial_move_ctor =
	  (is_trivially_move_constructible_v<_Types> && ...);
      static constexpr bool _S_trivial_copy_assign =
	  _S_trivial_dtor && (is_trivially_copy_assignable_v<_Types> && ...);
      static constexpr bool _S_trivial_move_assign =
	  _S_trivial_dtor && (is_trivially_move_assignable_v<_Types> && ...);

      // The following nothrow traits are for non-trivial SMFs. Trivial SMFs
      // are always nothrow.
      static constexpr bool _S_nothrow_default_ctor =
	  is_nothrow_default_constructible_v<
	      typename _Nth_type<0, _Types...>::type>;
      static constexpr bool _S_nothrow_copy_ctor = false;
      static constexpr bool _S_nothrow_move_ctor =
	  (is_nothrow_move_constructible_v<_Types> && ...);
      static constexpr bool _S_nothrow_copy_assign = false;
      static constexpr bool _S_nothrow_move_assign =
	  _S_nothrow_move_ctor && (is_nothrow_move_assignable_v<_Types> && ...);
    };

  // Defines members and ctors.
  template<typename... _Types>
    union _Variadic_union { };

  template<typename _First, typename... _Rest>
    union _Variadic_union<_First, _Rest...>
    {
      constexpr _Variadic_union() : _M_rest() { }

      template<typename... _Args>
	constexpr _Variadic_union(in_place_index_t<0>, _Args&&... __args)
	: _M_first(in_place_index<0>, std::forward<_Args>(__args)...)
	{ }

      template<size_t _Np, typename... _Args>
	constexpr _Variadic_union(in_place_index_t<_Np>, _Args&&... __args)
	: _M_rest(in_place_index<_Np-1>, std::forward<_Args>(__args)...)
	{ }

      _Uninitialized<_First> _M_first;
      _Variadic_union<_Rest...> _M_rest;
    };

  // Defines index and the dtor, possibly trivial.
  template<bool __trivially_destructible, typename... _Types>
    struct _Variant_storage;

  template <typename... _Types>
  using __select_index =
    typename __select_int::_Select_int_base<sizeof...(_Types) + 1,
					    unsigned char,
					    unsigned short>::type::value_type;

  template<typename... _Types>
    struct _Variant_storage<false, _Types...>
    {
      template<size_t... __indices>
	static constexpr void (*_S_vtable[])(const _Variant_storage&) =
	    { &__erased_dtor<const _Variant_storage&, __indices>... };

      constexpr _Variant_storage() : _M_index(variant_npos) { }

      template<size_t _Np, typename... _Args>
	constexpr _Variant_storage(in_place_index_t<_Np>, _Args&&... __args)
	: _M_u(in_place_index<_Np>, std::forward<_Args>(__args)...),
	_M_index(_Np)
	{ }

      template<size_t... __indices>
	constexpr void _M_reset_impl(std::index_sequence<__indices...>)
	{
	  if (_M_index != __index_type(variant_npos))
	    _S_vtable<__indices...>[_M_index](*this);
	}

      void _M_reset()
      {
	_M_reset_impl(std::index_sequence_for<_Types...>{});
	_M_index = variant_npos;
      }

      ~_Variant_storage()
      { _M_reset(); }

      void*
      _M_storage() const
      {
	return const_cast<void*>(static_cast<const void*>(
	    std::addressof(_M_u)));
      }

      constexpr bool
      _M_valid() const noexcept
      {
	return this->_M_index != __index_type(variant_npos);
      }

      _Variadic_union<_Types...> _M_u;
      using __index_type = __select_index<_Types...>;
      __index_type _M_index;
    };

  template<typename... _Types>
    struct _Variant_storage<true, _Types...>
    {
      constexpr _Variant_storage() : _M_index(variant_npos) { }

      template<size_t _Np, typename... _Args>
	constexpr _Variant_storage(in_place_index_t<_Np>, _Args&&... __args)
	: _M_u(in_place_index<_Np>, std::forward<_Args>(__args)...),
	_M_index(_Np)
	{ }

      void _M_reset()
      { _M_index = variant_npos; }

      void*
      _M_storage() const
      {
	return const_cast<void*>(static_cast<const void*>(
	    std::addressof(_M_u)));
      }

      constexpr bool
      _M_valid() const noexcept
      {
	return this->_M_index != __index_type(variant_npos);
      }

      _Variadic_union<_Types...> _M_u;
      using __index_type = __select_index<_Types...>;
      __index_type _M_index;
    };

  template<typename... _Types>
    using _Variant_storage_alias =
	_Variant_storage<_Traits<_Types...>::_S_trivial_dtor, _Types...>;

  // The following are (Copy|Move) (ctor|assign) layers for forwarding
  // triviality and handling non-trivial SMF behaviors.

  template<bool, typename... _Types>
    struct _Copy_ctor_base : _Variant_storage_alias<_Types...>
    {
      using _Base = _Variant_storage_alias<_Types...>;
      using _Base::_Base;

      _Copy_ctor_base(const _Copy_ctor_base& __rhs)
	  noexcept(_Traits<_Types...>::_S_nothrow_copy_ctor)
      {
	if (__rhs._M_valid())
	  {
	    static constexpr void (*_S_vtable[])(void*, void*) =
	      { &__erased_ctor<_Types&, const _Types&>... };
	    _S_vtable[__rhs._M_index](this->_M_storage(), __rhs._M_storage());
	    this->_M_index = __rhs._M_index;
	  }
      }

      _Copy_ctor_base(_Copy_ctor_base&&) = default;
      _Copy_ctor_base& operator=(const _Copy_ctor_base&) = default;
      _Copy_ctor_base& operator=(_Copy_ctor_base&&) = default;
    };

  template<typename... _Types>
    struct _Copy_ctor_base<true, _Types...> : _Variant_storage_alias<_Types...>
    {
      using _Base = _Variant_storage_alias<_Types...>;
      using _Base::_Base;
    };

  template<typename... _Types>
    using _Copy_ctor_alias =
	_Copy_ctor_base<_Traits<_Types...>::_S_trivial_copy_ctor, _Types...>;

  template<bool, typename... _Types>
    struct _Move_ctor_base : _Copy_ctor_alias<_Types...>
    {
      using _Base = _Copy_ctor_alias<_Types...>;
      using _Base::_Base;

      _Move_ctor_base(_Move_ctor_base&& __rhs)
	  noexcept(_Traits<_Types...>::_S_nothrow_move_ctor)
      {
	if (__rhs._M_valid())
	  {
	    static constexpr void (*_S_vtable[])(void*, void*) =
	      { &__erased_ctor<_Types&, _Types&&>... };
	    _S_vtable[__rhs._M_index](this->_M_storage(), __rhs._M_storage());
	    this->_M_index = __rhs._M_index;
	  }
      }

      _Move_ctor_base(const _Move_ctor_base&) = default;
      _Move_ctor_base& operator=(const _Move_ctor_base&) = default;
      _Move_ctor_base& operator=(_Move_ctor_base&&) = default;
    };

  template<typename... _Types>
    struct _Move_ctor_base<true, _Types...> : _Copy_ctor_alias<_Types...>
    {
      using _Base = _Copy_ctor_alias<_Types...>;
      using _Base::_Base;
    };

  template<typename... _Types>
    using _Move_ctor_alias =
	_Move_ctor_base<_Traits<_Types...>::_S_trivial_move_ctor, _Types...>;

  template<bool, typename... _Types>
    struct _Copy_assign_base : _Move_ctor_alias<_Types...>
    {
      using _Base = _Move_ctor_alias<_Types...>;
      using _Base::_Base;

      _Copy_assign_base&
      operator=(const _Copy_assign_base& __rhs)
	  noexcept(_Traits<_Types...>::_S_nothrow_copy_assign)
      {
	if (this->_M_index == __rhs._M_index)
	  {
	    if (__rhs._M_valid())
	      {
		static constexpr void (*_S_vtable[])(void*, void*) =
		  { &__erased_assign<_Types&, const _Types&>... };
		_S_vtable[__rhs._M_index](this->_M_storage(), __rhs._M_storage());
	      }
	  }
	else
	  {
	    _Copy_assign_base __tmp(__rhs);
	    this->~_Copy_assign_base();
	    __try
	      {
		::new (this) _Copy_assign_base(std::move(__tmp));
	      }
	    __catch (...)
	      {
		this->_M_index = variant_npos;
		__throw_exception_again;
	      }
	  }
	__glibcxx_assert(this->_M_index == __rhs._M_index);
	return *this;
      }

      _Copy_assign_base(const _Copy_assign_base&) = default;
      _Copy_assign_base(_Copy_assign_base&&) = default;
      _Copy_assign_base& operator=(_Copy_assign_base&&) = default;
    };

  template<typename... _Types>
    struct _Copy_assign_base<true, _Types...> : _Move_ctor_alias<_Types...>
    {
      using _Base = _Move_ctor_alias<_Types...>;
      using _Base::_Base;
    };

  template<typename... _Types>
    using _Copy_assign_alias =
	_Copy_assign_base<_Traits<_Types...>::_S_trivial_copy_assign,
			  _Types...>;

  template<bool, typename... _Types>
    struct _Move_assign_base : _Copy_assign_alias<_Types...>
    {
      using _Base = _Copy_assign_alias<_Types...>;
      using _Base::_Base;

      void _M_destructive_move(_Move_assign_base&& __rhs)
      {
	this->~_Move_assign_base();
	__try
	  {
	    ::new (this) _Move_assign_base(std::move(__rhs));
	  }
	__catch (...)
	  {
	    this->_M_index = variant_npos;
	    __throw_exception_again;
	  }
      }

      _Move_assign_base&
      operator=(_Move_assign_base&& __rhs)
	  noexcept(_Traits<_Types...>::_S_nothrow_move_assign)
      {
	if (this->_M_index == __rhs._M_index)
	  {
	    if (__rhs._M_valid())
	      {
		static constexpr void (*_S_vtable[])(void*, void*) =
		  { &__erased_assign<_Types&, const _Types&>... };
		_S_vtable[__rhs._M_index](this->_M_storage(), __rhs._M_storage());
	      }
	  }
	else
	  {
	    _Move_assign_base __tmp(__rhs);
	    this->~_Move_assign_base();
	    __try
	      {
		::new (this) _Move_assign_base(std::move(__tmp));
	      }
	    __catch (...)
	      {
		this->_M_index = variant_npos;
		__throw_exception_again;
	      }
	  }
	__glibcxx_assert(this->_M_index == __rhs._M_index);
	return *this;
      }

      _Move_assign_base(const _Move_assign_base&) = default;
      _Move_assign_base(_Move_assign_base&&) = default;
      _Move_assign_base& operator=(const _Move_assign_base&) = default;
    };

  template<typename... _Types>
    struct _Move_assign_base<true, _Types...> : _Copy_assign_alias<_Types...>
    {
      using _Base = _Copy_assign_alias<_Types...>;
      using _Base::_Base;
    };

  template<typename... _Types>
    using _Move_assign_alias =
	_Move_assign_base<_Traits<_Types...>::_S_trivial_move_assign,
			  _Types...>;

  template<typename... _Types>
    struct _Variant_base : _Move_assign_alias<_Types...>
    {
      using _Base = _Move_assign_alias<_Types...>;

      constexpr
      _Variant_base()
	  noexcept(_Traits<_Types...>::_S_nothrow_default_ctor)
      : _Variant_base(in_place_index<0>) { }

      template<size_t _Np, typename... _Args>
	constexpr explicit
	_Variant_base(in_place_index_t<_Np> __i, _Args&&... __args)
	: _Base(__i, std::forward<_Args>(__args)...)
	{ }

      _Variant_base(const _Variant_base&) = default;
      _Variant_base(_Variant_base&&) = default;
      _Variant_base& operator=(const _Variant_base&) = default;
      _Variant_base& operator=(_Variant_base&&) = default;
    };

  // For how many times does _Tp appear in _Tuple?
  template<typename _Tp, typename _Tuple>
    struct __tuple_count;

  template<typename _Tp, typename _Tuple>
    inline constexpr size_t __tuple_count_v =
      __tuple_count<_Tp, _Tuple>::value;

  template<typename _Tp, typename... _Types>
    struct __tuple_count<_Tp, tuple<_Types...>>
    : integral_constant<size_t, 0> { };

  template<typename _Tp, typename _First, typename... _Rest>
    struct __tuple_count<_Tp, tuple<_First, _Rest...>>
    : integral_constant<
	size_t,
	__tuple_count_v<_Tp, tuple<_Rest...>> + is_same_v<_Tp, _First>> { };

  // TODO: Reuse this in <tuple> ?
  template<typename _Tp, typename... _Types>
    inline constexpr bool __exactly_once =
      __tuple_count_v<_Tp, tuple<_Types...>> == 1;

  // Takes _Types and create an overloaded _S_fun for each type.
  // If a type appears more than once in _Types, create only one overload.
  template<typename... _Types>
    struct __overload_set
    { static void _S_fun(); };

  template<typename _First, typename... _Rest>
    struct __overload_set<_First, _Rest...> : __overload_set<_Rest...>
    {
      using __overload_set<_Rest...>::_S_fun;
      static integral_constant<size_t, sizeof...(_Rest)> _S_fun(_First);
    };

  template<typename... _Rest>
    struct __overload_set<void, _Rest...> : __overload_set<_Rest...>
    {
      using __overload_set<_Rest...>::_S_fun;
    };

  // Helper for variant(_Tp&&) and variant::operator=(_Tp&&).
  // __accepted_index maps the arbitrary _Tp to an alternative type in _Variant.
  template<typename _Tp, typename _Variant, typename = void>
    struct __accepted_index
    { static constexpr size_t value = variant_npos; };

  template<typename _Tp, typename... _Types>
    struct __accepted_index<
      _Tp, variant<_Types...>,
      decltype(__overload_set<_Types...>::_S_fun(std::declval<_Tp>()),
	       std::declval<void>())>
    {
      static constexpr size_t value = sizeof...(_Types) - 1
	- decltype(__overload_set<_Types...>::
		   _S_fun(std::declval<_Tp>()))::value;
    };

  // Returns the raw storage for __v.
  template<typename _Variant>
    void* __get_storage(_Variant&& __v)
    { return __v._M_storage(); }

  // Used for storing multi-dimensional vtable.
  template<typename _Tp, size_t... _Dimensions>
    struct _Multi_array
    {
      constexpr const _Tp&
      _M_access() const
      { return _M_data; }

      _Tp _M_data;
    };

  template<typename _Tp, size_t __first, size_t... __rest>
    struct _Multi_array<_Tp, __first, __rest...>
    {
      template<typename... _Args>
	constexpr const _Tp&
	_M_access(size_t __first_index, _Args... __rest_indices) const
	{ return _M_arr[__first_index]._M_access(__rest_indices...); }

      _Multi_array<_Tp, __rest...> _M_arr[__first];
    };

  // Creates a multi-dimensional vtable recursively.
  //
  // For example,
  // visit([](auto, auto){},
  //       variant<int, char>(),  // typedef'ed as V1
  //       variant<float, double, long double>())  // typedef'ed as V2
  // will trigger instantiations of:
  // __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&), 2, 3>,
  //                   tuple<V1&&, V2&&>, std::index_sequence<>>
  //   __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&), 3>,
  //                     tuple<V1&&, V2&&>, std::index_sequence<0>>
  //     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
  //                       tuple<V1&&, V2&&>, std::index_sequence<0, 0>>
  //     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
  //                       tuple<V1&&, V2&&>, std::index_sequence<0, 1>>
  //     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
  //                       tuple<V1&&, V2&&>, std::index_sequence<0, 2>>
  //   __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&), 3>,
  //                     tuple<V1&&, V2&&>, std::index_sequence<1>>
  //     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
  //                       tuple<V1&&, V2&&>, std::index_sequence<1, 0>>
  //     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
  //                       tuple<V1&&, V2&&>, std::index_sequence<1, 1>>
  //     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
  //                       tuple<V1&&, V2&&>, std::index_sequence<1, 2>>
  // The returned multi-dimensional vtable can be fast accessed by the visitor
  // using index calculation.
  template<typename _Array_type, typename _Variant_tuple, typename _Index_seq>
    struct __gen_vtable_impl;

  template<typename _Result_type, typename _Visitor, size_t... __dimensions,
	   typename... _Variants, size_t... __indices>
    struct __gen_vtable_impl<
	_Multi_array<_Result_type (*)(_Visitor, _Variants...), __dimensions...>,
	tuple<_Variants...>, std::index_sequence<__indices...>>
    {
      using _Next =
	  remove_reference_t<typename _Nth_type<sizeof...(__indices),
			     _Variants...>::type>;
      using _Array_type =
	  _Multi_array<_Result_type (*)(_Visitor, _Variants...),
		       __dimensions...>;

      static constexpr _Array_type
      _S_apply()
      {
	_Array_type __vtable{};
	_S_apply_all_alts(
	  __vtable, make_index_sequence<variant_size_v<_Next>>());
	return __vtable;
      }

      template<size_t... __var_indices>
	static constexpr void
	_S_apply_all_alts(_Array_type& __vtable,
			  std::index_sequence<__var_indices...>)
	{
	  (_S_apply_single_alt<__var_indices>(
	     __vtable._M_arr[__var_indices]), ...);
	}

      template<size_t __index, typename _Tp>
	static constexpr void
	_S_apply_single_alt(_Tp& __element)
	{
	  using _Alternative = variant_alternative_t<__index, _Next>;
	  __element = __gen_vtable_impl<
	    remove_reference_t<
	      decltype(__element)>, tuple<_Variants...>,
	      std::index_sequence<__indices..., __index>>::_S_apply();
	}
    };

  template<typename _Result_type, typename _Visitor, typename... _Variants,
	   size_t... __indices>
    struct __gen_vtable_impl<
      _Multi_array<_Result_type (*)(_Visitor, _Variants...)>,
		   tuple<_Variants...>, std::index_sequence<__indices...>>
    {
      using _Array_type =
	  _Multi_array<_Result_type (*)(_Visitor&&, _Variants...)>;

      decltype(auto)
      static constexpr __visit_invoke(_Visitor&& __visitor, _Variants... __vars)
      {
	return __invoke(std::forward<_Visitor>(__visitor),
			std::get<__indices>(
			    std::forward<_Variants>(__vars))...);
      }

      static constexpr auto
      _S_apply()
      { return _Array_type{&__visit_invoke}; }
    };

  template<typename _Result_type, typename _Visitor, typename... _Variants>
    struct __gen_vtable
    {
      using _Func_ptr = _Result_type (*)(_Visitor&&, _Variants...);
      using _Array_type =
	  _Multi_array<_Func_ptr,
		       variant_size_v<remove_reference_t<_Variants>>...>;

      static constexpr _Array_type
      _S_apply()
      {
	return __gen_vtable_impl<_Array_type, tuple<_Variants...>,
				 std::index_sequence<>>::_S_apply();
      }

      static constexpr auto _S_vtable = _S_apply();
    };

  template<size_t _Np, typename _Tp>
    struct _Base_dedup : public _Tp { };

  template<typename _Variant, typename __indices>
    struct _Variant_hash_base;

  template<typename... _Types, size_t... __indices>
    struct _Variant_hash_base<variant<_Types...>,
			      std::index_sequence<__indices...>>
    : _Base_dedup<__indices, __poison_hash<remove_const_t<_Types>>>... { };

} // namespace __variant
} // namespace __detail

  template<typename _Tp, typename... _Types>
    inline constexpr bool holds_alternative(const variant<_Types...>& __v)
    noexcept
    {
      static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>,
		    "T should occur for exactly once in alternatives");
      return __v.index() == __detail::__variant::__index_of_v<_Tp, _Types...>;
    }

  template<typename _Tp, typename... _Types>
    constexpr inline _Tp& get(variant<_Types...>& __v)
    {
      static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>,
		    "T should occur for exactly once in alternatives");
      static_assert(!is_void_v<_Tp>, "_Tp should not be void");
      return std::get<__detail::__variant::__index_of_v<_Tp, _Types...>>(__v);
    }

  template<typename _Tp, typename... _Types>
    constexpr inline _Tp&& get(variant<_Types...>&& __v)
    {
      static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>,
		    "T should occur for exactly once in alternatives");
      static_assert(!is_void_v<_Tp>, "_Tp should not be void");
      return std::get<__detail::__variant::__index_of_v<_Tp, _Types...>>(
	std::move(__v));
    }

  template<typename _Tp, typename... _Types>
    constexpr inline const _Tp& get(const variant<_Types...>& __v)
    {
      static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>,
		    "T should occur for exactly once in alternatives");
      static_assert(!is_void_v<_Tp>, "_Tp should not be void");
      return std::get<__detail::__variant::__index_of_v<_Tp, _Types...>>(__v);
    }

  template<typename _Tp, typename... _Types>
    constexpr inline const _Tp&& get(const variant<_Types...>&& __v)
    {
      static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>,
		    "T should occur for exactly once in alternatives");
      static_assert(!is_void_v<_Tp>, "_Tp should not be void");
      return std::get<__detail::__variant::__index_of_v<_Tp, _Types...>>(
	std::move(__v));
    }

  template<size_t _Np, typename... _Types>
    constexpr inline
    add_pointer_t<variant_alternative_t<_Np, variant<_Types...>>>
    get_if(variant<_Types...>* __ptr) noexcept
    {
      using _Alternative_type = variant_alternative_t<_Np, variant<_Types...>>;
      static_assert(_Np < sizeof...(_Types),
		    "The index should be in [0, number of alternatives)");
      static_assert(!is_void_v<_Alternative_type>, "_Tp should not be void");
      if (__ptr && __ptr->index() == _Np)
	return &__detail::__variant::__get<_Np>(*__ptr);
      return nullptr;
    }

  template<size_t _Np, typename... _Types>
    constexpr inline
    add_pointer_t<const variant_alternative_t<_Np, variant<_Types...>>>
    get_if(const variant<_Types...>* __ptr) noexcept
    {
      using _Alternative_type = variant_alternative_t<_Np, variant<_Types...>>;
      static_assert(_Np < sizeof...(_Types),
		    "The index should be in [0, number of alternatives)");
      static_assert(!is_void_v<_Alternative_type>, "_Tp should not be void");
      if (__ptr && __ptr->index() == _Np)
	return &__detail::__variant::__get<_Np>(*__ptr);
      return nullptr;
    }

  template<typename _Tp, typename... _Types>
    constexpr inline add_pointer_t<_Tp>
    get_if(variant<_Types...>* __ptr) noexcept
    {
      static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>,
		    "T should occur for exactly once in alternatives");
      static_assert(!is_void_v<_Tp>, "_Tp should not be void");
      return std::get_if<__detail::__variant::__index_of_v<_Tp, _Types...>>(
	  __ptr);
    }

  template<typename _Tp, typename... _Types>
    constexpr inline add_pointer_t<const _Tp>
    get_if(const variant<_Types...>* __ptr)
    noexcept
    {
      static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>,
		    "T should occur for exactly once in alternatives");
      static_assert(!is_void_v<_Tp>, "_Tp should not be void");
      return std::get_if<__detail::__variant::__index_of_v<_Tp, _Types...>>(
	  __ptr);
    }

  struct monostate { };

#define _VARIANT_RELATION_FUNCTION_TEMPLATE(__OP, __NAME) \
  template<typename... _Types> \
    constexpr bool operator __OP(const variant<_Types...>& __lhs, \
				 const variant<_Types...>& __rhs) \
    { \
      return __lhs._M_##__NAME(__rhs, std::index_sequence_for<_Types...>{}); \
    } \
\
  constexpr bool operator __OP(monostate, monostate) noexcept \
  { return 0 __OP 0; }

  _VARIANT_RELATION_FUNCTION_TEMPLATE(<, less)
  _VARIANT_RELATION_FUNCTION_TEMPLATE(<=, less_equal)
  _VARIANT_RELATION_FUNCTION_TEMPLATE(==, equal)
  _VARIANT_RELATION_FUNCTION_TEMPLATE(!=, not_equal)
  _VARIANT_RELATION_FUNCTION_TEMPLATE(>=, greater_equal)
  _VARIANT_RELATION_FUNCTION_TEMPLATE(>, greater)

#undef _VARIANT_RELATION_FUNCTION_TEMPLATE

  template<typename _Visitor, typename... _Variants>
    constexpr decltype(auto) visit(_Visitor&&, _Variants&&...);

  template<typename... _Types>
    inline enable_if_t<(is_move_constructible_v<_Types> && ...)
			&& (is_swappable_v<_Types> && ...)>
    swap(variant<_Types...>& __lhs, variant<_Types...>& __rhs)
    noexcept(noexcept(__lhs.swap(__rhs)))
    { __lhs.swap(__rhs); }

  template<typename... _Types>
    enable_if_t<!((is_move_constructible_v<_Types> && ...)
		   && (is_swappable_v<_Types> && ...))>
    swap(variant<_Types...>&, variant<_Types...>&) = delete;

  class bad_variant_access : public exception
  {
  public:
    bad_variant_access() noexcept : _M_reason("Unknown reason") { }
    const char* what() const noexcept override
    { return _M_reason; }

  private:
    bad_variant_access(const char* __reason) : _M_reason(__reason) { }

    const char* _M_reason;

    friend void __throw_bad_variant_access(const char* __what);
  };

  inline void
  __throw_bad_variant_access(const char* __what)
  { _GLIBCXX_THROW_OR_ABORT(bad_variant_access(__what)); }

  template<typename... _Types>
    class variant
    : private __detail::__variant::_Variant_base<_Types...>,
      private _Enable_default_constructor<
	__detail::__variant::_Traits<_Types...>::_S_default_ctor,
	  variant<_Types...>>,
      private _Enable_copy_move<
	__detail::__variant::_Traits<_Types...>::_S_copy_ctor,
	__detail::__variant::_Traits<_Types...>::_S_copy_assign,
	__detail::__variant::_Traits<_Types...>::_S_move_ctor,
	__detail::__variant::_Traits<_Types...>::_S_move_assign,
	variant<_Types...>>
    {
    private:
      static_assert(sizeof...(_Types) > 0,
		    "variant must have at least one alternative");
      static_assert(!(std::is_reference_v<_Types> || ...),
		    "variant must have no reference alternative");
      static_assert(!(std::is_void_v<_Types> || ...),
		    "variant must have no void alternative");

      using _Base = __detail::__variant::_Variant_base<_Types...>;
      using _Default_ctor_enabler =
	_Enable_default_constructor<
	  __detail::__variant::_Traits<_Types...>::_S_default_ctor,
	    variant<_Types...>>;

      template<typename _Tp>
	static constexpr bool
	__exactly_once = __detail::__variant::__exactly_once<_Tp, _Types...>;

      template<typename _Tp>
	static constexpr size_t __accepted_index =
	  __detail::__variant::__accepted_index<_Tp&&, variant>::value;

      template<size_t _Np, bool = _Np < sizeof...(_Types)>
	struct __to_type_impl;

      template<size_t _Np>
	struct __to_type_impl<_Np, true>
	{ using type = variant_alternative_t<_Np, variant>; };

      template<size_t _Np>
	using __to_type = typename __to_type_impl<_Np>::type;

      template<typename _Tp>
	using __accepted_type = __to_type<__accepted_index<_Tp>>;

      template<typename _Tp>
	static constexpr size_t __index_of =
	  __detail::__variant::__index_of_v<_Tp, _Types...>;

      using _Traits = __detail::__variant::_Traits<_Types...>;

    public:
      variant() = default;
      variant(const variant& __rhs) = default;
      variant(variant&&) = default;
      variant& operator=(const variant&) = default;
      variant& operator=(variant&&) = default;
      ~variant() = default;

      template<typename _Tp,
	       typename = enable_if_t<!is_same_v<decay_t<_Tp>, variant>>,
	       typename = enable_if_t<(sizeof...(_Types)>0)>,
	       typename = enable_if_t<__exactly_once<__accepted_type<_Tp&&>>
			  && is_constructible_v<__accepted_type<_Tp&&>, _Tp&&>>>
	constexpr
	variant(_Tp&& __t)
	noexcept(is_nothrow_constructible_v<__accepted_type<_Tp&&>, _Tp&&>)
	: variant(in_place_index<__accepted_index<_Tp&&>>,
		  std::forward<_Tp>(__t))
	{ __glibcxx_assert(holds_alternative<__accepted_type<_Tp&&>>(*this)); }

      template<typename _Tp, typename... _Args,
	       typename = enable_if_t<__exactly_once<_Tp>
			  && is_constructible_v<_Tp, _Args&&...>>>
	constexpr explicit
	variant(in_place_type_t<_Tp>, _Args&&... __args)
	: variant(in_place_index<__index_of<_Tp>>,
		  std::forward<_Args>(__args)...)
	{ __glibcxx_assert(holds_alternative<_Tp>(*this)); }

      template<typename _Tp, typename _Up, typename... _Args,
	       typename = enable_if_t<__exactly_once<_Tp>
			  && is_constructible_v<
			    _Tp, initializer_list<_Up>&, _Args&&...>>>
	constexpr explicit
	variant(in_place_type_t<_Tp>, initializer_list<_Up> __il,
		_Args&&... __args)
	: variant(in_place_index<__index_of<_Tp>>, __il,
		  std::forward<_Args>(__args)...)
	{ __glibcxx_assert(holds_alternative<_Tp>(*this)); }

      template<size_t _Np, typename... _Args,
	       typename = enable_if_t<
		 is_constructible_v<__to_type<_Np>, _Args&&...>>>
	constexpr explicit
	variant(in_place_index_t<_Np>, _Args&&... __args)
	: _Base(in_place_index<_Np>, std::forward<_Args>(__args)...),
	_Default_ctor_enabler(_Enable_default_constructor_tag{})
	{ __glibcxx_assert(index() == _Np); }

      template<size_t _Np, typename _Up, typename... _Args,
	       typename = enable_if_t<is_constructible_v<__to_type<_Np>,
				      initializer_list<_Up>&, _Args&&...>>>
	constexpr explicit
	variant(in_place_index_t<_Np>, initializer_list<_Up> __il,
		_Args&&... __args)
	: _Base(in_place_index<_Np>, __il, std::forward<_Args>(__args)...),
	_Default_ctor_enabler(_Enable_default_constructor_tag{})
	{ __glibcxx_assert(index() == _Np); }

      template<typename _Tp>
	enable_if_t<__exactly_once<__accepted_type<_Tp&&>>
		    && is_constructible_v<__accepted_type<_Tp&&>, _Tp&&>
		    && is_assignable_v<__accepted_type<_Tp&&>&, _Tp&&>
		    && !is_same_v<decay_t<_Tp>, variant>, variant&>
	operator=(_Tp&& __rhs)
	noexcept(is_nothrow_assignable_v<__accepted_type<_Tp&&>&, _Tp&&>
		 && is_nothrow_constructible_v<__accepted_type<_Tp&&>, _Tp&&>)
	{
	  constexpr auto __index = __accepted_index<_Tp&&>;
	  if (index() == __index)
	    std::get<__index>(*this) = std::forward<_Tp>(__rhs);
	  else
	    this->emplace<__index>(std::forward<_Tp>(__rhs));
	  __glibcxx_assert(holds_alternative<__accepted_type<_Tp&&>>(*this));
	  return *this;
	}

      template<typename _Tp, typename... _Args>
	enable_if_t<is_constructible_v<_Tp, _Args...> && __exactly_once<_Tp>,
		    _Tp&>
	emplace(_Args&&... __args)
	{
	  auto& ret =
	    this->emplace<__index_of<_Tp>>(std::forward<_Args>(__args)...);
	  __glibcxx_assert(holds_alternative<_Tp>(*this));
	  return ret;
	}

      template<typename _Tp, typename _Up, typename... _Args>
	enable_if_t<is_constructible_v<_Tp, initializer_list<_Up>&, _Args...>
		    && __exactly_once<_Tp>,
		    _Tp&>
	emplace(initializer_list<_Up> __il, _Args&&... __args)
	{
	  auto& ret =
	    this->emplace<__index_of<_Tp>>(__il,
					   std::forward<_Args>(__args)...);
	  __glibcxx_assert(holds_alternative<_Tp>(*this));
	  return ret;
	}

      template<size_t _Np, typename... _Args>
	enable_if_t<is_constructible_v<variant_alternative_t<_Np, variant>,
				       _Args...>,
		    variant_alternative_t<_Np, variant>&>
	emplace(_Args&&... __args)
	{
	  static_assert(_Np < sizeof...(_Types),
			"The index should be in [0, number of alternatives)");
	  this->~variant();
	  __try
	    {
	      ::new (this) variant(in_place_index<_Np>,
				   std::forward<_Args>(__args)...);
	    }
	  __catch (...)
	    {
	      this->_M_index = variant_npos;
	      __throw_exception_again;
	    }
	  __glibcxx_assert(index() == _Np);
	  return std::get<_Np>(*this);
	}

      template<size_t _Np, typename _Up, typename... _Args>
	enable_if_t<is_constructible_v<variant_alternative_t<_Np, variant>,
				       initializer_list<_Up>&, _Args...>,
		    variant_alternative_t<_Np, variant>&>
	emplace(initializer_list<_Up> __il, _Args&&... __args)
	{
	  static_assert(_Np < sizeof...(_Types),
			"The index should be in [0, number of alternatives)");
	  this->~variant();
	  __try
	    {
	      ::new (this) variant(in_place_index<_Np>, __il,
				   std::forward<_Args>(__args)...);
	    }
	  __catch (...)
	    {
	      this->_M_index = variant_npos;
	      __throw_exception_again;
	    }
	  __glibcxx_assert(index() == _Np);
	  return std::get<_Np>(*this);
	}

      constexpr bool valueless_by_exception() const noexcept
      { return !this->_M_valid(); }

      constexpr size_t index() const noexcept
      {
	if (this->_M_index ==
	    typename _Base::__index_type(variant_npos))
	  return variant_npos;
	return this->_M_index;
      }

      void
      swap(variant& __rhs)
      noexcept((__is_nothrow_swappable<_Types>::value && ...)
	       && is_nothrow_move_constructible_v<variant>)
      {
	if (this->index() == __rhs.index())
	  {
	    if (this->_M_valid())
	      {
		static constexpr void (*_S_vtable[])(void*, void*) =
		  { &__detail::__variant::__erased_swap<_Types&, _Types&>... };
		_S_vtable[__rhs._M_index](this->_M_storage(),
					  __rhs._M_storage());
	      }
	  }
	else if (!this->_M_valid())
	  {
	    this->_M_destructive_move(std::move(__rhs));
	    __rhs._M_reset();
	  }
	else if (!__rhs._M_valid())
	  {
	    __rhs._M_destructive_move(std::move(*this));
	    this->_M_reset();
	  }
	else
	  {
	    auto __tmp = std::move(__rhs);
	    __rhs._M_destructive_move(std::move(*this));
	    this->_M_destructive_move(std::move(__tmp));
	  }
      }

    private:
#define _VARIANT_RELATION_FUNCTION_TEMPLATE(__OP, __NAME) \
      template<size_t... __indices> \
	static constexpr bool \
	(*_S_erased_##__NAME[])(const variant&, const variant&) = \
	  { &__detail::__variant::__erased_##__NAME< \
		const variant&, __indices>... }; \
      template<size_t... __indices> \
	constexpr inline bool \
	_M_##__NAME(const variant& __rhs, \
		    std::index_sequence<__indices...>) const \
	{ \
	  auto __lhs_index = this->index(); \
	  auto __rhs_index = __rhs.index(); \
	  if (__lhs_index != __rhs_index || valueless_by_exception()) \
	    /* Modulo addition. */ \
	    return __lhs_index + 1 __OP __rhs_index + 1; \
	  return _S_erased_##__NAME<__indices...>[__lhs_index](*this, __rhs); \
	}

      _VARIANT_RELATION_FUNCTION_TEMPLATE(<, less)
      _VARIANT_RELATION_FUNCTION_TEMPLATE(<=, less_equal)
      _VARIANT_RELATION_FUNCTION_TEMPLATE(==, equal)
      _VARIANT_RELATION_FUNCTION_TEMPLATE(!=, not_equal)
      _VARIANT_RELATION_FUNCTION_TEMPLATE(>=, greater_equal)
      _VARIANT_RELATION_FUNCTION_TEMPLATE(>, greater)

#undef _VARIANT_RELATION_FUNCTION_TEMPLATE

#ifdef __clang__
    public:
      using _Base::_M_u; // See https://bugs.llvm.org/show_bug.cgi?id=31852
    private:
#endif

      template<size_t _Np, typename _Vp>
	friend constexpr decltype(auto) __detail::__variant::__get(_Vp&& __v);

      template<typename _Vp>
	friend void* __detail::__variant::__get_storage(_Vp&& __v);

#define _VARIANT_RELATION_FUNCTION_TEMPLATE(__OP) \
      template<typename... _Tp> \
	friend constexpr bool \
	operator __OP(const variant<_Tp...>& __lhs, \
		      const variant<_Tp...>& __rhs);

      _VARIANT_RELATION_FUNCTION_TEMPLATE(<)
      _VARIANT_RELATION_FUNCTION_TEMPLATE(<=)
      _VARIANT_RELATION_FUNCTION_TEMPLATE(==)
      _VARIANT_RELATION_FUNCTION_TEMPLATE(!=)
      _VARIANT_RELATION_FUNCTION_TEMPLATE(>=)
      _VARIANT_RELATION_FUNCTION_TEMPLATE(>)

#undef _VARIANT_RELATION_FUNCTION_TEMPLATE
    };

  template<size_t _Np, typename... _Types>
    constexpr variant_alternative_t<_Np, variant<_Types...>>&
    get(variant<_Types...>& __v)
    {
      static_assert(_Np < sizeof...(_Types),
		    "The index should be in [0, number of alternatives)");
      if (__v.index() != _Np)
	__throw_bad_variant_access("Unexpected index");
      return __detail::__variant::__get<_Np>(__v);
    }

  template<size_t _Np, typename... _Types>
    constexpr variant_alternative_t<_Np, variant<_Types...>>&&
    get(variant<_Types...>&& __v)
    {
      static_assert(_Np < sizeof...(_Types),
		    "The index should be in [0, number of alternatives)");
      if (__v.index() != _Np)
	__throw_bad_variant_access("Unexpected index");
      return __detail::__variant::__get<_Np>(std::move(__v));
    }

  template<size_t _Np, typename... _Types>
    constexpr const variant_alternative_t<_Np, variant<_Types...>>&
    get(const variant<_Types...>& __v)
    {
      static_assert(_Np < sizeof...(_Types),
		    "The index should be in [0, number of alternatives)");
      if (__v.index() != _Np)
	__throw_bad_variant_access("Unexpected index");
      return __detail::__variant::__get<_Np>(__v);
    }

  template<size_t _Np, typename... _Types>
    constexpr const variant_alternative_t<_Np, variant<_Types...>>&&
    get(const variant<_Types...>&& __v)
    {
      static_assert(_Np < sizeof...(_Types),
		    "The index should be in [0, number of alternatives)");
      if (__v.index() != _Np)
	__throw_bad_variant_access("Unexpected index");
      return __detail::__variant::__get<_Np>(std::move(__v));
    }

  template<typename _Visitor, typename... _Variants>
    constexpr decltype(auto)
    visit(_Visitor&& __visitor, _Variants&&... __variants)
    {
      if ((__variants.valueless_by_exception() || ...))
	__throw_bad_variant_access("Unexpected index");

      using _Result_type =
	decltype(std::forward<_Visitor>(__visitor)(
	    get<0>(std::forward<_Variants>(__variants))...));

      constexpr auto& __vtable = __detail::__variant::__gen_vtable<
	_Result_type, _Visitor&&, _Variants&&...>::_S_vtable;

      auto __func_ptr = __vtable._M_access(__variants.index()...);
      return (*__func_ptr)(std::forward<_Visitor>(__visitor),
			   std::forward<_Variants>(__variants)...);
    }

  template<bool, typename... _Types>
    struct __variant_hash_call_base_impl
    {
      size_t
      operator()(const variant<_Types...>& __t) const
      noexcept((is_nothrow_invocable_v<hash<decay_t<_Types>>, _Types> && ...))
      {
	if (!__t.valueless_by_exception())
	  {
	    namespace __edv = __detail::__variant;
	    static constexpr size_t (*_S_vtable[])(void*) =
	      { &__edv::__erased_hash<const _Types&>... };
	    return hash<size_t>{}(__t.index())
	      + _S_vtable[__t.index()](__edv::__get_storage(__t));
	  }
	return hash<size_t>{}(__t.index());
      }
    };

  template<typename... _Types>
    struct __variant_hash_call_base_impl<false, _Types...> {};

  template<typename... _Types>
    using __variant_hash_call_base =
    __variant_hash_call_base_impl<(__poison_hash<remove_const_t<_Types>>::
				   __enable_hash_call &&...), _Types...>;

  template<typename... _Types>
    struct hash<variant<_Types...>>
    : private __detail::__variant::_Variant_hash_base<
	variant<_Types...>, std::index_sequence_for<_Types...>>,
      public __variant_hash_call_base<_Types...>
    {
      using result_type [[__deprecated__]] = size_t;
      using argument_type [[__deprecated__]] = variant<_Types...>;
    };

  template<>
    struct hash<monostate>
    {
      using result_type [[__deprecated__]] = size_t;
      using argument_type [[__deprecated__]] = monostate;

      size_t
      operator()(const monostate& __t) const noexcept
      {
	constexpr size_t __magic_monostate_hash = -7777;
	return __magic_monostate_hash;
      }
    };

  template<typename... _Types>
    struct __is_fast_hash<hash<variant<_Types...>>>
    : bool_constant<(__is_fast_hash<_Types>::value && ...)>
    { };

_GLIBCXX_END_NAMESPACE_VERSION
} // namespace std

#endif // C++17

#endif // _GLIBCXX_VARIANT

