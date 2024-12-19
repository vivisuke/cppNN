#pragma once

#include <vector>
#include <memory>

class Layer;

//	シーケンシャルネットワーククラス
class Network {
public:
	Network(int nInput);		//	入力次元
public:
	Network& add(Layer*);
private:
	int		m_nInput;
	std::vector<float>	m_input;						//	入力値
	std::vector<std::auto_ptr<Layer>>	m_layers;		//	各レイヤーへのオートポインタ配列
};

//	シーケンシャルネットワーク各レイヤー基底クラス
class Layer {
public:
	Layer(int nInput = -1, int nOutput = -1)
		: m_nInput(nInput), m_nOutput(nOutput)
	{}
	virtual ~Layer() {};
public:
	void	set_nInput(int nInput) { m_nInput = nInput; }
	int		get_nOutput() const { return m_nOutput; }
protected:
	int		m_nInput;
	int		m_nOutput;
};

//	総結合層
class FullyConnected : public Layer {
public:
	FullyConnected(int nOutput);
	~FullyConnected();
public:
	void	set_nInput(int nInput);
private:

};

