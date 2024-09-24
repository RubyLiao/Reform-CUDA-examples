
#ifndef B_HPP
#define B_HPP

#include "b.hpp"
#include "c.hpp"


#include <iostream>




void speakB(int number)
{
    std::cout<< "For b.cpp the number is " << number << std::endl;
    std::cout<< "I'm calling speakC in b.cpp:" << std::endl;
    speakC(3);
}

#endif