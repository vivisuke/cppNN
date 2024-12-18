#pragma once

#include <vector>
#include <memory>

class Network {		//	�V�[�P���V�����l�b�g���[�N�N���X
public:
	Network(int nInput);		//	���͎���
private:
	int		m_nInput;
	std::vector<float>	m_input;						//	���͒l
	std::vector<std::auto_ptr<class Layer>>	m_vlayers;		//	�e���C���[�ւ̃I�[�g�|�C���^�z��
};

class Layer {		//	�V�[�P���V�����l�b�g���[�N�e���C���[���N���X
public:
	Layer() {}
	virtual ~Layer() = 0;
};

