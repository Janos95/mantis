#pragma once

#include <cstddef>
#include <vector>

#ifdef USE_TBB

#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <tbb/enumerable_thread_specific.h>

#endif

// The meshing code can make use of parallelization. We provide two backends, a serial one and a parallel one
// using tbb. You can also provide your own implementation by providing the following three concepts:
//  1. A parallel_for function
//  2. A class called ThreadSpecific which provides thread local storage
//  3. A TaskGroup class which can be used to task based parallelism.
// You can check out the serial implementations below for an example of how to implement these concepts.

// Serial backend
#ifdef USE_SERIAL
template<class F>
void parallel_for(size_t start, size_t end, F func) {
    for (int i = start; i < end; ++i) {
        func(i);
    }
}

template<class T>
struct ThreadSpecific {
    ThreadSpecific() : m_data(1) {}

    auto begin() {
        return m_data.begin();
    }

    auto begin() const {
        return m_data.begin();
    }

    auto end() {
        return m_data.end();
    }

    auto end() const {
        return m_data.end();
    }

    auto &local() {
        return m_data[0];
    }

    std::vector<T> m_data;
};

struct TaskGroup {
    TaskGroup() = default;
    template<class F>
    void run(const F &func) {
        func();
    }
    // noop, since we immediately execute the tasks
    void wait() {}
};
#endif

#ifdef USE_TBB

template<class F>
void parallel_for(size_t start, size_t end, F func) {
    tbb::parallel_for(start, end, func);
}

template<class T>
using ThreadSpecific = tbb::enumerable_thread_specific<T>;

using TaskGroup = tbb::task_group;

#endif
