#pragma once

#include <vector>
#include <memory>

class Layer;

//	�V�[�P���V�����l�b�g���[�N�N���X
class Network {
public:
	Network(int nInput);		//	���͎���
public:
	Network& add(Layer*);
private:
	int		m_nInput;
	std::vector<float>	m_input;						//	���͒l
	std::vector<std::auto_ptr<Layer>>	m_layers;		//	�e���C���[�ւ̃I�[�g�|�C���^�z��
};

//	�V�[�P���V�����l�b�g���[�N�e���C���[���N���X
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

//	�������w
class FullyConnected : public Layer {
public:
	FullyConnected(int nOutput);
	~FullyConnected();
public:
	void	set_nInput(int nInput);
private:

};

