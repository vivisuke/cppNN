#include <iostream>
#include <random>
#include <assert.h>
#include "cppNN.h"

using namespace std;

random_device seed_gen2;
mt19937 mt2(seed_gen2());
std::uniform_real_distribution<float> distf(-1.0f, 1.0f);

int main()
{
	if( false ) {	//	順伝播
		//	2入力、2出力ネットワーク、y1 = x1, y2 = x2（恒等変換）
		Network net(2);		//	2入力ネットワーク
		AffineMap *af1;
		net.add(af1 = new AffineMap(2));			//	総結合層
		const vector<vector<float>>& wt0 = {{0.0f, 1.00f, 0.0f}, {0.0f, 0.0f, 1.00f}, };
		af1->set_weight(wt0);		//	重み指定
		vector<float> data1({+0.3f, -0.4f});
		for(int i = 0; i != 10; ++i) {
			data1[0] = distf(mt2);
			data1[1] = distf(mt2);
			net.forward(data1);
			net.print();
			assert( net.get_outputs() == data1 );
		}
	}
	if( false ) {	//	順伝播
		//	2入力、2出力ネットワーク、y1 = x1, y2 = x2（恒等変換）
		Network net(2);		//	2入力ネットワーク
		AffineMap *af1;
		net.add(af1 = new AffineMap(2));			//	総結合層
		const vector<vector<float>>& wt0 = {{0.0f, 1.00f, 0.0f}, {0.0f, 0.0f, 1.00f}, };
		//const vector<vector<float>>& wt1 = {{1.00f, 0.724f, 0.690f}};
		af1->set_weight(wt0);		//	重み指定
		//net.print();
		vector<float> data1({+1, +1});
		net.forward(data1);
		net.print();
		vector<float> data2({+1, -1});
		net.forward(data2);
		net.print();
		vector<float> data3({-1, -1});
		net.forward(data3);
		net.print();
		vector<float> data4({-1, +1});
		net.forward(data4);
		net.print();
	}
	if( false ) {	//	平均自乗誤差計算
		//	2入力、2出力ネットワーク、y1 = x1, y2 = x2（恒等変換）
		Network net(2);		//	2入力ネットワーク
		AffineMap *af1;
		net.add(af1 = new AffineMap(2));			//	総結合層
		const vector<vector<float>>& wt0 = {{0.0f, 1.00f, 0.0f}, {0.0f, 0.0f, 1.00f}, };
		//const vector<vector<float>>& wt1 = {{1.00f, 0.724f, 0.690f}};
		af1->set_weight(wt0);		//	重み指定
		//net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}, };
		const vector<vector<float>>& teacher_data = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}, };
		net.forward_loss(train_data, teacher_data);
		const vector<vector<float>>& wt2 = {{0.5f, 1.00f, 0.0f}, {0.0f, 0.0f, 1.00f}, };
		af1->set_weight(wt2);		//	重み指定
		net.forward_loss(train_data, teacher_data);
	}
	if( false ) {	//	ΔL/ΔW による勾配計算
		//	2入力、2出力ネットワーク、y1 = x1, y2 = x2（恒等変換）
		Network net(2);		//	2入力ネットワーク
		AffineMap *af1;
		net.add(af1 = new AffineMap(2));			//	総結合層
		const vector<vector<float>>& wt0 = {{0.0f, 1.00f, 0.0f}, {0.0f, 0.0f, 1.00f}, };
		//const vector<vector<float>>& wt1 = {{1.00f, 0.724f, 0.690f}};
		af1->set_weight(wt0);		//	重み指定
		//net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}, };
		const vector<vector<float>>& teacher_data = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}, };
		net.forward_diff(train_data, teacher_data);
	}
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
		//net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, };
		const vector<vector<float>>& teacher_data = {{2}, {1}, {1}, {0}, };
		net.train(train_data, teacher_data, 10);	//	エポック数
		net.print();
	}
	if( true ) {
		//	2入力、2出力ネットワーク、y1 = x1, y2 = x2（恒等変換）
		Network net(2);		//	2入力ネットワーク
		AffineMap *af1;
		net.add(af1 = new AffineMap(2));			//	総結合層
		const vector<vector<float>>& wt0 = {{1.0f, 1.00f, 0.0f}, {0.0f, 0.0f, 1.00f}, };
		//const vector<vector<float>>& wt1 = {{1.00f, 0.724f, 0.690f}};
		af1->set_weight(wt0);		//	重み指定
		//net.print();
		const vector<float>& train_data = {1, 1};
		const vector<float>& teacher_data = {1, 1};
		//const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}, };
		//const vector<vector<float>>& teacher_data = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}, };
		for(int i = 0; i != 5; ++i) {
			net.forward_backward(train_data, teacher_data);
			net.update_weights(0.1f);
			net.print();
		}
	}
	if( false ) {
		//	2入力、2出力ネットワーク、x1 のみ2倍
		Network net(2);		//	2入力ネットワーク
		net.add(new AffineMap(2));			//	総結合層
		//net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}, };
		const vector<vector<float>>& teacher_data = {{2, 1}, {2, -1}, {-2, -1}, {-2, 1}, };
		net.train(train_data, teacher_data, 20);	//	エポック数
		net.print();
	}
	if( false ) {
		//	2入力、2出力ネットワーク、時計方向に90度回転
		Network net(2);		//	2入力ネットワーク
		net.add(new AffineMap(2));			//	総結合層
		//net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}, };
		const vector<vector<float>>& teacher_data = {{1, -1}, {-1, -1}, {-1, 1}, {1, 1}, };
		net.train(train_data, teacher_data, 20);	//	エポック数
		net.print();
	}
	if( false ) {
		//	2入力、１出力ネットワーク、y = x1 & x2
		Network net(2);		//	2入力ネットワーク
		AffineMap *af1;
		net.add(af1 = new AffineMap(1));			//	総結合層
		//const vector<vector<float>>& wt0 = {{0.5f, 1.00f, 0.0f}};
		//const vector<vector<float>>& wt1 = {{1.00f, 0.724f, 0.690f}};
		//af1->set_weight(wt0);
		net.add(new AFtanh());				//	活性化関数：tanh() 
		//net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
		const vector<vector<float>>& teacher_data = {{1}, {-1}, {-1}, {-1}};
		//net.forward_backward(train_data, teacher_data);
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
		net.train(train_data, teacher_data, 20);
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
		//const vector<vector<float>>& wt1 = {{0.80f, -0.43f, 0.90f}, {-0.46f, 0.41f, 0.91f}};
		//af1->set_weight(wt1);
		//const vector<vector<float>>& wt2 = {{0.68f, -0.68f, 0.74f}};
		//af2->set_weight(wt2);
		net.print();
		const vector<vector<float>>& train_data = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
		const vector<vector<float>>& teacher_data = {{-1}, {1}, {1}, {-1}};
		net.train(train_data, teacher_data, 10);
		//net.print();
	}

    std::cout << endl << "OK" << endl;
}
