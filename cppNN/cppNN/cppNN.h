#pragma once

#include <vector>
#include <memory>

class Network {		//	シーケンシャルネットワーククラス
public:
	Network(int nInput);		//	入力次元
private:
	int		m_nInput;
	std::vector<float>	m_input;						//	入力値
	std::vector<std::auto_ptr<class Layer>>	m_vlayers;		//	各レイヤーへのオートポインタ配列
};

class Layer {		//	シーケンシャルネットワーク各レイヤー基底クラス
public:
	Layer() {}
	virtual ~Layer() = 0;
};

