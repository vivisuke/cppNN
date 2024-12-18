
#include "cppNN.h"

Network::Network(int nInput)
	: m_nInput(nInput)
{
	m_vinput.resize(nInput+1);		//	+1 for バイアス
	m_vinput[0] = 1.0;
}
