
#include "cppNN.h"

Network::Network(int nInput)
	: m_nInput(nInput)
{
	m_input.resize(nInput+1);		//	+1 for �o�C�A�X
	m_input[0] = 1.0;
}
