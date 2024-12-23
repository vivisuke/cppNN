﻿#include <iostream>
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
	if( false ) {
		//	2入力、１出力ネットワーク、1 の数を数えて出力
		Network net(2);		//	2入力ネットワーク
		net.add(new AffineMap(1));			//	総結合層
		net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
		const vector<vector<float>>& teacher_data = {{2}, {1}, {1}, {0}};
		net.train(train_data, teacher_data, 10);
		net.print();
	}
	if( true ) {
		//	2入力、１出力ネットワーク、y = x1 & x2
		Network net(2);		//	2入力ネットワーク
		AffineMap *af1;
		net.add(af1 = new AffineMap(1));			//	総結合層
		const vector<vector<float>>& wt1 = {{1.00f, 0.724f, 0.690f}};
		af1->set_weight(wt1);
		net.add(new AFtanh());				//	活性化関数：tanh() 
		net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
		const vector<vector<float>>& teacher_data = {{1}, {-1}, {-1}, {-1}};
		net.train(train_data, teacher_data, 1);
		net.print();
	}
	if( false ) {
		//	2入力、１出力ネットワーク、y = x1 | x2
		Network net(2);		//	2入力ネットワーク
		net.add(new AffineMap(1));			//	総結合層
		net.add(new AFtanh());				//	活性化関数：tanh() 
		net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
		const vector<vector<float>>& teacher_data = {{1}, {1}, {1}, {-1}};
		net.train(train_data, teacher_data, 10);
		net.print();
	}
	if( false ) {
		//	2入力、１出力単層ネットワーク、y = x1 ^ x2
		Network net(2);		//	2入力ネットワーク
		net.add(new AffineMap(1));			//	総結合層
		net.add(new AFtanh());				//	活性化関数：tanh() 
		net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
		const vector<vector<float>>& teacher_data = {{-1}, {1}, {1}, {-1}};
		net.train(train_data, teacher_data, 10);
		net.print();
	}
	if( false ) {
		//	2入力、１出力２層ネットワーク、y = x1 ^ x2
		Network net(2);		//	2入力ネットワーク
		net.add(new AffineMap(2));			//	総結合層
		net.add(new AFtanh());				//	活性化関数：tanh() 
		net.add(new AffineMap(1));			//	総結合層
		net.add(new AFtanh());				//	活性化関数：tanh() 
		//net.print();
		vector<float> idata({1, 1});
		net.forward(idata);
		net.print();
	}
	if( false ) {
		//	2入力、１出力２層ネットワーク、y = x1 ^ x2
		Network net(2);		//	2入力ネットワーク
		AffineMap *af1, *af2;
		net.add(af1 = new AffineMap(2));			//	総結合層
		net.add(new AFtanh());				//	活性化関数：tanh() 
		net.add(af2 = new AffineMap(1));			//	総結合層
		net.add(new AFtanh());				//	活性化関数：tanh() 
		const vector<vector<float>>& wt1 = {{0.80f, -0.43f, 0.90f}, {-0.46f, 0.41f, 0.91f}};
		af1->set_weight(wt1);
		const vector<vector<float>>& wt2 = {{0.68f, -0.68f, 0.74f}};
		af2->set_weight(wt2);
		net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
		const vector<vector<float>>& teacher_data = {{-1}, {1}, {1}, {-1}};
		net.train(train_data, teacher_data, 1);
		//net.print();
	}

    std::cout << endl << "OK" << endl;
}
