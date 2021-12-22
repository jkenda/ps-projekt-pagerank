#pragma once
#include <chrono>
#include <string>

class Timer
{
    private:
    const char *_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

    public:
    Timer(const char *name);
    ~Timer();
    void stop();
};

#define RESET_TIMER Timer::reset();
#define TIMER(name) Timer timer##__LINE__(name);
#define AUTO_TIMER  Timer timer##__LINE__(__FUNCTION__);
