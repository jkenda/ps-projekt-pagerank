#include "Timer.hpp"

Timer::Timer(float &time_dst)
    : _time_dst(time_dst)
{
    _start_time = std::chrono::high_resolution_clock::now();
}

void Timer::stop()
{
    auto end_time = std::chrono::high_resolution_clock::now();
    auto start_us = std::chrono::time_point_cast<std::chrono::microseconds>(_start_time).time_since_epoch().count();
    auto end_us   = std::chrono::time_point_cast<std::chrono::microseconds>(end_time).time_since_epoch().count();
    _time_dst = static_cast<float>(end_us - start_us) / 1'000'000;
}

Timer::~Timer()
{
    stop();
}
