#pragma once

#include <vector>


class Network {
public:
	Network(int nInput);		//	���͎���
private:
	int		m_nInput;
	std::vector<float>	m_vinput;		//	���͒l
};
