
#ifndef A_HPP
#define A_HPP

#include "a.hpp"
#include "b.hpp"
#include "c.hpp"

#include <iostream>




void speakA(int number)
{
    std::cout<< "For a.cpp the number is " << number << std::endl;
    std::cout<< "I'm calling speakB in a.cpp:" <<std::endl;
    speakB(2);
    std::cout<< "I'm calling speakC in a.cpp:" <<std::endl;
    speakC(3);
}

#endif