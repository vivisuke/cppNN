#pragma once

#include <vector>
#include <memory>

class Layer;

typedef unsigned char uchar;

enum {
	LT_FULLY_CNCT = 1,
	LT_TANH,
	LT_SIGMOID,
	LT_SOFTMAX,
};

//	シーケンシャルネットワーククラス
class Network {
public:
	Network(int nInput);		//	入力次元
public:
	void	print() const;
	Network& add(Layer*);
private:
	int		m_nInput;
	std::vector<float>	m_input;						//	入力値
	std::vector<std::auto_ptr<Layer>>	m_layers;		//	各レイヤーへのオートポインタ配列
};

//	シーケンシャルネットワーク各レイヤー基底クラス
class Layer {
public:
	Layer(uchar type, int nInput = 0, int nOutput = 0)
		: m_type(type), m_nInput(nInput), m_nOutput(nOutput)
	{}
	virtual ~Layer() {};
public:
	virtual void	print() const {}
	virtual void	set_nInput(int nInput) { m_nInput = nInput; }
	int		get_nOutput() const { return m_nOutput; }
protected:
	uchar	m_type;
	int		m_nInput;
	int		m_nOutput;
};

//	総結合層
class AffineMap : public Layer {
public:
	AffineMap(int nOutput);
	~AffineMap();
public:
	void	print() const;
	void	set_nInput(int nInput);
private:
	std::vector<std::vector<float>>		m_weights;
};
//	活性化関数：tanh() 
class AFtanh : public Layer {
public:
	AFtanh();
	~AFtanh();
public:
	void	print() const;
	void	set_nInput(int nInput);
private:
};

