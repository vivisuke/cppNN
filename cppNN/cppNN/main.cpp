#include <iostream>
#include "cppNN.h"

using namespace std;

int main()
{
	Network net(2);		//	2入力ネットワーク
	auto fc1 = new AffineMap(2);
	net.add(fc1);
	fc1->print();

    std::cout << endl << "OK" << endl;
}
