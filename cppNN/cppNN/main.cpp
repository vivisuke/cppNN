#include <iostream>
#include "cppNN.h"
using namespace std;
int main()
{
	if( false ) {
		Network net(2);		//	2入力ネットワーク
		auto fc1 = new AffineMap(1);		//	総結合層
		auto af1 = new AFtanh();			//	活性化関数：tanh()
		net.add(fc1);
		//net.add(af1);
		//fc1->print();
		//af1->print();
		//net.print();
		//vector<float> idata({1, 1});
		//fc1->forward(idata);
		//net.forward(idata);
		//net.print();
	}
	//
	if( true ) {
		//	2入力、１出力ネットワーク、1 の数を数えて出力
		Network net(2);		//	2入力ネットワーク
		net.add(new AffineMap(1));			//	総結合層
		net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
		const vector<vector<float>>& teacher_data = {{2}, {1}, {1}, {0}};
		net.train(train_data, teacher_data, 10);
		net.print();
	}

    std::cout << endl << "OK" << endl;
}
