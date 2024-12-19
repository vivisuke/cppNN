#include <iostream>
#include "cppNN.h"

using namespace std;

int main()
{
	Network net(2);		//	2入力ネットワーク
	auto fc1 = new AffineMap(2);		//	総結合層
	auto af1 = new AFtanh(2);			//	活性化関数：tanh()
	net.add(fc1);
	net.add(af1);
	fc1->print();

    std::cout << endl << "OK" << endl;
}
