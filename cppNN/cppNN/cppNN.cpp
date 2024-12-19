
#include <random>
#include "cppNN.h"

using namespace std;

std::random_device seed_gen;
std::mt19937 mt(seed_gen());

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
AffineMap::AffineMap(int nOutput)
	: Layer(LT_FULLY_CNCT, -1, nOutput)
{
}
AffineMap::~AffineMap() {
}

void AffineMap::set_nInput(int nInput) {
	m_nInput = nInput;
	m_weights.clear();
	m_weights.resize(m_nOutput);
	// 平均0.0f、標準偏差 1/sqrt(nInput) 正規分布
	std::normal_distribution<float> dist(0.0f, (float)(1/sqrt((double)m_nInput)));
	for(int o = 0; o != m_nOutput; ++o) {
		m_weights[o].resize(m_nInput +1);		//	+1 for バイアス
		for(int i = 0; i <= m_nInput; ++i) {
			m_weights[o][i] = dist(mt);			//	Xavier初期化
		}
	}
}
