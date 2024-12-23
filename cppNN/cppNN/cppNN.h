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

//	�V�[�P���V�����l�b�g���[�N�N���X
class Network {
public:
	Network(int nInput);		//	���͎���
public:
	void	print() const;
	Network& add(Layer*);
	void	forward(const std::vector<float>&);
	void	backward(const std::vector<float>&);
	void	train(const std::vector<std::vector<float>>&, const std::vector<std::vector<float>>&, int);
private:
	int		m_nInput;
	std::vector<float>	m_input;						//	���͒l
	std::vector<std::auto_ptr<Layer>>	m_layers;		//	�e���C���[�ւ̃I�[�g�|�C���^�z��
	std::vector<float>	m_grad;							//	�덷�l
};

//	�V�[�P���V�����l�b�g���[�N�e���C���[���N���X
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
	std::vector<float>		m_outputs;		//	�o�͒l
	std::vector<float>		m_grad;			//	�덷�l
};

//	�������w
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
	std::vector<std::vector<float>>		m_slw;			//	��L/��Wij ���v
};
//	�������֐��Ftanh() 
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

