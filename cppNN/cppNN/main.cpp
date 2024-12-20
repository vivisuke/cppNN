#include <iostream>
#include "cppNN.h"
using namespace std;
int main()
{
	Network net(2);		//	2入力ネットワーク
	auto fc1 = new AffineMap(1);		//	総結合層
	auto af1 = new AFtanh();			//	活性化関数：tanh()
	net.add(fc1);
	net.add(af1);
	//fc1->print();
	//af1->print();
	//net.print();
	vector<float> idata({1, 1});
	//fc1->forward(idata);
	net.forward(idata);
	net.print();

    std::cout << endl << "OK" << endl;
}
