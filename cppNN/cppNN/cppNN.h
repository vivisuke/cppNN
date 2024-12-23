#pragma once

#include <vector>
#include <memory>

class Layer;

typedef unsigned char uchar;

enum {
	LT_AFFINE = 1,
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
	void	forward(const std::vector<float>&);
	void	backward(const std::vector<float>&);
	void	train(const std::vector<std::vector<float>>&, const std::vector<std::vector<float>>&, int);
private:
	int		m_nInput;
	std::vector<float>	m_input;						//	入力値
	std::vector<std::auto_ptr<Layer>>	m_layers;		//	各レイヤーへのオートポインタ配列
	std::vector<float>	m_grad;							//	誤差値
};

//	シーケンシャルネットワーク各レイヤー基底クラス
class Layer {
	friend class Network;
public:
	Layer(uchar type, int nInput = 0, int nOutput = 0)
		: m_type(type), m_nInput(nInput), m_nOutput(nOutput)
	{
		m_outputs.resize(nOutput);
	}
	virtual ~Layer() {};
public:
	int		get_nOutput() const { return m_nOutput; }
	virtual void	print() const {}
	virtual void	set_nInput(int nInput) {
		m_nInput = nInput;
		m_grad.resize(nInput);
	}
	virtual void	init_slw() {}
	virtual void	update(float) {}
	virtual void	forward(const std::vector<float>&) {}
	virtual void	backward(const std::vector<float>&, const std::vector<float>&) {}
protected:
	uchar	m_type;
	int		m_nInput;
	int		m_nOutput;
	std::vector<float>		m_outputs;		//	出力値
	std::vector<float>		m_grad;			//	誤差値
};

//	総結合層
class AffineMap : public Layer {
public:
	AffineMap(int nOutput);
	~AffineMap();
public:
	void	print() const;
	void	set_nInput(int nInput);
	void	set_weight(const std::vector<std::vector<float>>&);
	void	init_slw();
	void	forward(const std::vector<float>&);
	void	backward(const std::vector<float>&, const std::vector<float>&);
	void	update(float alpha);
private:
	std::vector<std::vector<float>>		m_weights;
	std::vector<std::vector<float>>		m_slw;			//	∂L/∂Wij 合計
};
//	活性化関数：tanh() 
class AFtanh : public Layer {
public:
	AFtanh();
	~AFtanh();
public:
	void	print() const;
	void	set_nInput(int nInput);
	void	forward(const std::vector<float>&);
	void	backward(const std::vector<float>&, const std::vector<float>&);
private:
};

