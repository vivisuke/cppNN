#include <iostream>
#include "cppNN.h"

using namespace std;

int main()
{
	Network net(2);		//	2入力ネットワーク
	auto fc1 = new FullyConnected(2);
	net.add(fc1);

    std::cout << endl << "OK" << endl;
}
