#pragma once
#include <chrono>
#include <string>

class Timer
{
    private:
    float &_time_dst;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

    public:
    Timer(float &time_dst);
    ~Timer();
    void stop();
};

#define TIMER(name) Timer timer##__LINE__(name);
#define AUTO_TIMER  Timer timer##__LINE__(__FUNCTION__);
