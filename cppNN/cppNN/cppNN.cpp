
#include <iostream>
#include <random>
#include <math.h>
#include "cppNN.h"

using namespace std;

random_device seed_gen;
mt19937 mt(seed_gen());

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
	m_grad.resize(ptr->m_nOutput);
	return *this;
}
//	順伝播、inputs にバイアス分は含まない
void Network::forward(const vector<float>& inputs) {
	const vector<float>* idata = &inputs;
	for(int i = 0; i != m_layers.size(); ++i) {
		m_layers[i]->forward(*idata);
		idata = &m_layers[i]->m_outputs;
	}
}
void Network::backward(const vector<float>& inputs) {
	const vector<float>* idata;
	const vector<float>* grad = &m_grad;
	for(int i = (int)m_layers.size(); --i >= 0; ) {
		if( i == 0 ) idata = &inputs;
		else idata = &m_layers[i-1]->m_outputs;
		m_layers[i]->backward(*idata, *grad);
		grad = &m_layers[i]->m_grad;
	}
}
void Network::train(const vector<vector<float>>& train_data, const vector<vector<float>>& teachr_data, int epoch) {
	const int nOutput = m_layers.back()->m_nOutput;
	m_grad.resize(nOutput);
	for(int epc = 0; epc != epoch; ++epc) {
		float loss = 0.0f;
		for(int i = 0; i != train_data.size(); ++i) {
			forward(train_data[i]);
			for(int o = 0; o != nOutput; ++o) {
				m_grad[o] = teachr_data[i][o] - m_layers.back()->m_outputs[o];
				loss += m_grad[o] * m_grad[o] / 2;
			}
			backward(train_data[i]);
		}
		cout << "loss = " << loss << endl;
		for(int i = 0; i != m_layers.size(); ++i) {
			if( m_layers[i]->m_type == LT_AFFINE )
				m_layers[i]->update(0.1f);
		}
	}
}
//----------------------------------------------------------------------
AffineMap::AffineMap(int nOutput)
	: Layer(LT_AFFINE, 0, nOutput)
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
	cout << " outputs[]: ";
	for(int o = 0; o != m_outputs.size(); ++o)
		cout << m_outputs[o] << ", ";
	cout << endl;
	cout << " grad[]: ";
	for(int o = 0; o != m_grad.size(); ++o)
		cout << m_grad[o] << ", ";
	cout << endl;
}
void AffineMap::set_nInput(int nInput) {
	m_nInput = nInput;
	m_grad.resize(nInput);		//	+1 for バイアス項
	m_weights.clear();
	m_weights.resize(m_nOutput);
	m_slw.resize(m_nOutput);
	// 平均0.0f、標準偏差 1/sqrt(nInput) 正規分布
	normal_distribution<float> dist(0.0f, (float)(1/sqrt((double)m_nInput)));
	for(int o = 0; o != m_nOutput; ++o) {
		m_weights[o].resize(m_nInput +1);		//	+1 for バイアス
		m_slw[o].resize(m_nInput +1);		//	+1 for バイアス
		for(int i = 0; i < m_nInput + 1; ++i) {
			m_weights[o][i] = dist(mt);			//	Xavier初期化
			m_slw[o][i] = 0.0f;
		}
	}
}
void AffineMap::init_slw() {
	for(int o = 0; o != m_nOutput; ++o) {
		for(int i = 0; i <= m_nInput + 1; ++i) {
			m_weights[o][i] = 0.0f;
		}
	}
}
//	順伝播、inputs にバイアス分は含まない
//			out[o] = ∑ inputs[i]*weights[o][i] + weights[o][0]
void AffineMap::forward(const vector<float>& inputs) {
	for(int o = 0; o != m_nOutput; ++o) {
		float sum = m_weights[o][0];			//	バイアス分
		for(int i = 0; i != m_nInput; ++i) {
			sum += inputs[i] * m_weights[o][i+1];
		}
		m_outputs[o] = sum;
	}
}
//	逆伝播
//			∂L/∂Wi = grad[i] * ∂y/∂Wi = grad[i] * inputs[i]
void AffineMap::backward(const vector<float>& inputs, const vector<float>& grad) {
	for (int o = 0; o != m_nOutput; ++o) {
		m_slw[o][0] += grad[o];
	}
	for(int i = 0; i != m_nInput; ++i) {
		m_grad[i] = 0.0f;
		for(int o = 0; o != m_nOutput; ++o) {
			auto g = inputs[i]*grad[o];
			m_slw[o][i+1] += g;
			m_grad[i] = g;
		}
	}
}
void AffineMap::update(float alpha) {
	for(int o = 0; o != m_nOutput; ++o) {
		for(int i = 0; i != m_nInput + 1; ++i) {
			m_weights[o][i] -= alpha * m_slw[0][i];
			m_slw[0][i] = 0.0f;
		}
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
	cout << " outputs[]:" << endl << " ";
	for(int o = 0; o != m_outputs.size(); ++o)
		cout << m_outputs[o] << ", ";
	cout << endl;
}
void AFtanh::set_nInput(int nInput) {
	m_nInput = m_nOutput = nInput;
	m_grad.resize(nInput);
	m_outputs.resize(m_nOutput);
}
void AFtanh::forward(const vector<float>& inputs) {
	for(int o = 0; o != m_nOutput; ++o) {
		m_outputs[o] = (float)tanh((double)inputs[o]);
	}
}
void AFtanh::backward(const vector<float>& inputs, const vector<float>& grad) {
	for(int i = 0; i != m_nOutput; ++i) {
		m_grad[i] = (1.0f - m_outputs[i]*m_outputs[i]) * grad[i];
	}
}

