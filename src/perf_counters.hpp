#ifndef PERF_COUNTERS_H
#define PERF_COUNTERS_H

#include <windows.h>
#include <pdh.h>
#include <cstring>
#include <string>
#include <vector>

struct PerfCounters {
    PDH_HQUERY query;
    std::vector<PDH_HCOUNTER> counters;
    std::vector<std::wstring> names;

    PerfCounters() : query(nullptr) {}

    void init() {
        PdhOpenQuery(nullptr, 0, &query);
    }

    void add_counter(const char* name, const char* path) {
        PDH_HCOUNTER counter;
        PdhAddCounterA(query, path, 0, &counter);
        counters.push_back(counter);
        names.push_back(std::wstring());
    }

    void start() {
        PdhCollectQueryData(query);
    }

    void stop() {
        PdhCollectQueryData(query);
    }

    double get_value(int idx) {
        PDH_FMT_COUNTERVALUE value;
        PdhGetFormattedCounterValue(counters[idx], PDH_FMT_DOUBLE, nullptr, &value);
        return value.doubleValue;
    }

    void cleanup() {
        for (auto c : counters) PdhRemoveCounter(c);
        if (query) PdhCloseQuery(query);
    }
};

#endif