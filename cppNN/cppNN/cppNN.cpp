
#include "cppNN.h"

using namespace std;

//----------------------------------------------------------------------
Network::Network(int nInput)
	: m_nInput(nInput)
{
	m_input.resize(nInput+1);		//	+1 for バイアス
	m_input[0] = 1.0;
}
Network& Network::add(Layer*ptr) {
	if( m_layers.empty() )
		ptr->set_nInput(m_nInput);
	else
		ptr->set_nInput(m_layers.back()->get_nOutput());
	m_layers.push_back(auto_ptr<Layer>(ptr));
	return *this;
}
//----------------------------------------------------------------------
FullyConnected::FullyConnected(int nOutput)
	: Layer(-1, nOutput)
{
}
FullyConnected::~FullyConnected() {
}

void FullyConnected::set_nInput(int nInput) {
	m_nInput = nInput;
}
