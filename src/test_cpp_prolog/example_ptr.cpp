#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>

#include "utils.h"

using namespace std;

int main() {
    // init a string
    string str = "Hello World!";
    cout << "Orgn str:\t" << str << endl;
    cout << "Orgn addr:\t" << &str << endl;
    
    // use stringstream to do "sprintf"
    stringstream ss;
    ss << &str;
    string addr;
    ss >> addr;

    cout << "Str addr:\t" << addr << endl;
    // clear stream and its buff
    ss.clear();
    ss.str("");
    
    // reload memory address from a string through stream
    memAdd uAddr;
    
    ss << addr;
    ss >> uAddr;
    cout << "Rld addr:\t" << uAddr << endl;
    
    // use this address to reload the original data
    string* p = (string*) uAddr;
    cout << "Rld str:\t" << *p << endl;

    return 0;
}
