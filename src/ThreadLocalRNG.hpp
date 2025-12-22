#ifndef THREADLOCALRNG_H
#define THREADLOCALRNG_H
#include <cstdint>
#include <random>

// Random number genrator
class ThreadLocalRNG
{
private:
std::mt19937 engine;
public:
    ThreadLocalRNG(){
        std::random_device rd;
        engine.seed(rd());
    }

    // [min,max] inclusive 
    size_t rand_int(int min,int max){
        std::uniform_int_distribution<size_t> dist(min,max); 
        return dist(engine);
    }

    // [0.0,1.0)
    float rand(){
        std::uniform_real_distribution<float> dist(0.0f,1.0f);
        return dist(engine);
    }

    uint32_t get(){
        return engine();
    }
};

inline thread_local ThreadLocalRNG RNG; // inline thread_local class object, Rng.get() -> random number
#endif