
#include "b.hpp"
#include "c.hpp"



using namespace std;


void speakB(int number)
{
    cout<< "For b.cpp the number is " << number << endl;
    cout<< "I'm calling speakC in b.cpp:" << endl;
    speakC(3);
}