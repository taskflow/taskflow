#pragma once

#include "../taskflow.hpp"
#include <atomic>

#if __cplusplus >= TF_CPP20
#include <condition_variable>
#include <mutex>
#endif

namespace tf
{
// Lazy initializer that can be used with taskflow without deadlocking
template <class T>
class Lazy
{
	/// The internal implementation that all instances share
	struct LazyImpl
	{
		/// The cached result
		// we cannot actually use std::variant here, since we cannot construct a function return
		// value in-place that way
		alignas(std::max(alignof(T), alignof(std::exception_ptr))) unsigned char m_data[std::max(
			sizeof(T), sizeof(std::exception_ptr))];
		enum class DataType : unsigned char
		{
			NONE = 0,
			CALCULATING = 1,
			VALUE = 2,
			EXCEPTION = 3
		};
		/// Thread safety handling
		std::atomic<DataType> m_dataType;
		/// The function used to generate the result
		const std::function<T()> m_fn;
		tf::Executor& m_ex;

#if __cplusplus < TF_CPP20
		std::condition_variable m_cv;
		std::mutex m_mtx;
#endif

		LazyImpl(std::function<T()> f, tf::Executor& ex) : m_dataType(DataType::NONE), m_fn(f), m_ex(ex) {}
		~LazyImpl()
		{
			const DataType t = m_dataType.load(std::memory_order_acquire);
			if (t == DataType::VALUE)
				std::launder(reinterpret_cast<T*>(m_data))->~T();
			else if (t == DataType::EXCEPTION)
				std::launder(reinterpret_cast<std::exception_ptr*>(m_data))->~exception_ptr();
		}

		T* get()
		{
			DataType t = m_dataType.load(std::memory_order_acquire);
			if (t <= DataType::CALCULATING)
			{
				// Nothing is yet calculated, need to calculate now
				const int wid = m_ex.this_worker_id();
				if (t == DataType::NONE
					&& m_dataType.compare_exchange_strong(
						t,
						DataType::CALCULATING,
						std::memory_order_release,
						std::memory_order_acquire))
				{
					const auto calculateResult = [&]()
					{
						DataType res;
						try
						{
							new (m_data) auto(m_fn());
							res = DataType::VALUE;
						}
						catch (...)
						{
							new (m_data) auto(std::current_exception());
							res = DataType::EXCEPTION;
						}
#if __cplusplus >= TF_CPP20
						m_dataType.store(res, std::memory_order_release);
						m_dataType.notify_all();
#else
						std::unique_lock l(m_mtx);
						m_dataType.store(res, std::memory_order_release);
						l.unlock();
						m_cv.notify_all();
#endif
					};

					// Our thread is the first to arrive here, it should calculate result now
					if (wid >= 0)
					{
						// If our current thread is from tf executor - we need to force
						// threads to stop stealing older tasks to prevent deadlocks
						m_ex.isolate(
							std::make_shared<tf::TaskArena>(m_ex), calculateResult);
					}
					else
					{
						// If our thread is not visible in executor pool - run function as simple function
						calculateResult();
					}
				}
				else
				{
					if (wid >= 0)
					{
						// Other thread can help with task-stealing to the calculating thread,
						// until result is available
						m_ex.corun_until(
							[&]() {
								return m_dataType.load(std::memory_order_relaxed)
								> DataType::CALCULATING;
							});
					}
					else
					{
						// Other thread is not from taskflow, so they can only wait until result is
						// available
#if __cplusplus >= TF_CPP20
						m_dataType.wait(DataType::CALCULATING);
#else
						{
							std::unique_lock lk(m_mtx);
							m_cv.wait(
								lk,
								[&] {
									return m_dataType.load(std::memory_order_relaxed)
										!= DataType::CALCULATING;
								});
						}
#endif
					}
				}

				t = m_dataType.load(std::memory_order_acquire);
			}

			// At this point we hold either an exception or a value
			if (t == DataType::EXCEPTION)
				std::rethrow_exception(
					*std::launder(reinterpret_cast<std::exception_ptr*>(m_data)));
			return std::launder(reinterpret_cast<T*>(m_data));
		}
	};

public:
	/// Pass a nullary (factory) function to be evaluated later.
	template <typename Function>
	Lazy(Function f, tf::Executor& ex): m_impl(std::make_shared<LazyImpl>(f, ex))
	{
		/// Returning a raw pointer here is bad behaviour
		/// as it is not clear at all who owns the value
		/// and who is responsible for deleting it.
		static_assert(!std::is_pointer_v<T>, "Factory function should not return a raw pointer");
	}

	T const& operator*() const { return *m_impl->get(); }
	T const* operator->() const { return m_impl->get(); }

	/// Returns true if the result has been calculated
	operator bool() const { return m_impl->m_data.has_value(); }

private:
	std::shared_ptr<LazyImpl> m_impl;
};
}