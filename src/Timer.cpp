#include "Timer.hpp"
#include <fstream>
#include <iostream>

Timer::Timer(const char *name)
    : _name(name)
{
    //std::cout << name << std::endl;
    _start_time = std::chrono::high_resolution_clock::now();
}

void Timer::stop()
{
    auto end_time = std::chrono::high_resolution_clock::now();
    auto start_us = std::chrono::time_point_cast<std::chrono::microseconds>(_start_time).time_since_epoch().count();
    auto end_us   = std::chrono::time_point_cast<std::chrono::microseconds>(end_time).time_since_epoch().count();
    std::cout << _name << (static_cast<float>(end_us - start_us) / 1000) << " millis\n";
}

Timer::~Timer()
{
    stop();
}
