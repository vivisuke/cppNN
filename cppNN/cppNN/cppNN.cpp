
#include <iostream>
#include <random>
#include <math.h>
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
void Network::print() const {
	cout << " nInput = " << m_nInput << endl;
	for(int i = 0; i != m_layers.size(); ++i)
		m_layers[i]->print();
}
Network& Network::add(Layer*ptr) {
	if( m_layers.empty() )
		ptr->set_nInput(m_nInput);
	else
		ptr->set_nInput(m_layers.back()->get_nOutput());
	m_layers.push_back(auto_ptr<Layer>(ptr));
	return *this;
}
//	順伝播、inputs にバイアス分は含まない
void Network::forward(const std::vector<float>& inputs) {
	const std::vector<float>* idata = &inputs;
	for(int i = 0; i != m_layers.size(); ++i) {
		m_layers[i]->forward(*idata);
		idata = &m_layers[i]->m_outputs;
	}
}
//----------------------------------------------------------------------
AffineMap::AffineMap(int nOutput)
	: Layer(LT_FULLY_CNCT, 0, nOutput)
{
}
AffineMap::~AffineMap() {
}
void AffineMap::print() const {
	cout << "AffineMap:" << endl;
	if( m_nInput <= 0 || m_nOutput <= 0 ) return;
	for(int o = 0; o != m_nOutput; ++o) {
		cout << " " << (o+1) << ": ";
		for(int i = 0; i <= m_nInput; ++i) {
			cout << m_weights[o][i] << " ";
		}
		cout << endl;
	}
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
//	順伝播、inputs にバイアス分は含まない
void AffineMap::forward(const std::vector<float>& inputs) {
	for(int o = 0; o != m_nOutput; ++o) {
		float sum = m_weights[o][0];			//	バイアス分
		for(int i = 0; i != m_nInput; ++i) {
			sum += inputs[i] * m_weights[o][i+1];
		}
		m_outputs[o] = sum;
	}
}
//----------------------------------------------------------------------
AFtanh::AFtanh()
	: Layer(LT_TANH, 0, 0)
{
}
AFtanh::~AFtanh() {
}
void AFtanh::print() const {
	cout << "AF tanh():";
	cout << " nInput = " << m_nInput << ", nOutput = " << m_nOutput << endl;
}
void AFtanh::set_nInput(int nInput) {
	m_nInput = m_nOutput = nInput;
	m_outputs.resize(m_nOutput);
}
void AFtanh::forward(const std::vector<float>& inputs) {
	for(int o = 0; o != m_nOutput; ++o) {
		m_outputs[o] = (float)tanh((double)inputs[o]);
	}
}

