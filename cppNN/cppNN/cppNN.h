#pragma once

#include <vector>


class Network {
public:
	Network(int nInput);		//	入力次元
private:
	int		m_nInput;
	std::vector<float>	m_vinput;		//	入力値
};
