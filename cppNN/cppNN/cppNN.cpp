
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
const vector<float>& Network::get_outputs() const {
	if( m_layers.empty() ) return m_input;
	else return m_layers.back()->m_outputs;
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
void Network::update_weights(float alpha) {
	for(int i = 0; i != m_layers.size(); ++i) {
		m_layers[i]->update_weights(alpha);
	}
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
	//const vector<float>* idata;
	const vector<float>* grad = &m_grad;
	for(int i = (int)m_layers.size(); --i >= 0; ) {
		//if( i == 0 ) idata = &inputs;
		//else idata = &m_layers[i-1]->m_outputs;
		m_layers[i]->backward(*grad);
		grad = &m_layers[i]->m_grad;
	}
}
float Network::forward_loss(const vector<vector<float>>& train_data, const vector<vector<float>>& teachr_data) {
	const int nOutput = m_layers.back()->m_nOutput;
	m_grad.resize(nOutput);
	fill(m_grad.begin(), m_grad.end(), 0.0f);
	float loss = 0.0f;
	for(int i = 0; i != train_data.size(); ++i) {
		forward(train_data[i]);
		//m_layers.back()->print();
		for(int o = 0; o != nOutput; ++o) {
			auto d = m_layers.back()->m_outputs[o] - teachr_data[i][o];
			m_grad[o] += d;
			loss += d * d / 2;
			cout << "loss = " << (d * d / 2) << endl;
		}
		backward(train_data[i]);
	}
	loss /= train_data.size() * nOutput;
	cout << "ave loss = " << loss << endl;
	return loss;
}
void Network::forward_diff(const vector<vector<float>>& train_data, const vector<vector<float>>& teachr_data) {
	const float DELTA = 0.01f;
	auto ptr = (AffineMap*)(m_layers[0].get());
	auto loss0 = forward_loss(train_data, teachr_data);
	ptr->m_bias[0] += DELTA;
	auto loss1 = forward_loss(train_data, teachr_data);
	ptr->m_bias[0] -= DELTA;
	cout << "∂L/∂b = " << (loss1 - loss0) / DELTA << endl;
}
void Network::forward_backward(const vector<float>& train_data, const vector<float>& teachr_data) {
	for(int i = 0; i != m_layers.size(); ++i)
		m_layers[i]->init_dweight();
	const int nOutput = m_layers.back()->m_nOutput;
	m_grad.resize(nOutput);
	float loss = 0.0f;
	forward(train_data);
	for(int o = 0; o != nOutput; ++o) {
		auto d = m_layers.back()->m_outputs[o] - teachr_data[o];
		m_grad[o] = d;
		loss += d * d / 2;
	}
	backward(m_grad);
}
void Network::forward_backward_batch(const vector<vector<float>>& train_data, const vector<vector<float>>& teachr_data) {
	const int nOutput = m_layers.back()->m_nOutput;
	m_grad.resize(nOutput);
	fill(m_grad.begin(), m_grad.end(), 0.0f);
	float loss = 0.0f;
	for(int i = 0; i != train_data.size(); ++i) {
		forward(train_data[i]);
		//m_layers.back()->print();
		for(int o = 0; o != nOutput; ++o) {
			auto d = m_layers.back()->m_outputs[o] - teachr_data[i][o];
			m_grad[o] += d;
			loss += d * d / 2;
		}
		backward(train_data[i]);
	}
	loss /= train_data.size() * nOutput;
	cout << "loss = " << loss << endl;
	cout << " grad[]: ";
	for(int o = 0; o != m_grad.size(); ++o)
		cout << m_grad[o] << ", ";
	cout << endl;
}
void Network::train(const vector<vector<float>>& train_data, const vector<vector<float>>& teachr_data, int epoch) {
	const int nOutput = m_layers.back()->m_nOutput;
	m_grad.resize(nOutput);
	for(int epc = 0; epc != epoch; ++epc) {
		fill(m_grad.begin(), m_grad.end(), 0.0f);
		float loss = 0.0f;
		for(int i = 0; i != train_data.size(); ++i) {
			forward(train_data[i]);
			//m_layers.back()->print();
			for(int o = 0; o != nOutput; ++o) {
				auto d = m_layers.back()->m_outputs[o] - teachr_data[i][o];
				m_grad[o] += d;
				loss += d * d / 2;
			}
			backward(train_data[i]);
		}
		loss /= train_data.size() * nOutput;
		cout << "loss = " << loss << endl;
		for(int i = 0; i != m_layers.size(); ++i) {
			//if( m_layers[i]->m_type == LT_AFFINE )
				m_layers[i]->update_weights(0.1f);
		}
	}
}
//----------------------------------------------------------------------
AffineMap::AffineMap(int nOutput)
	: Layer(LT_AFFINE, 0, nOutput)
{
	m_bias.resize(nOutput);
}
AffineMap::~AffineMap() {
}
void AffineMap::print() const {
	cout << "AffineMap:" << endl;
	if( m_nInput <= 0 || m_nOutput <= 0 ) return;
	for(int o = 0; o != m_nOutput; ++o) {
		cout << " " << (o+1) << ": ";
		cout << m_bias[o] << " ";
		for(int i = 0; i != m_nInput; ++i) {
			cout << m_weights[o][i] << " ";
		}
		cout << endl;
	}
	cout << " outputs[]: ";
	for(int o = 0; o != m_outputs.size(); ++o)
		cout << m_outputs[o] << ", ";
	cout << endl;
	cout << " sum of ∂L/∂Wij: " << endl;
	for(int o = 0; o != m_nOutput; ++o) {
		cout << " " << (o+1) << ": ";
		cout << m_dbias[o] << " ";
		for(int i = 0; i != m_nInput; ++i) {
			cout << m_dweights[o][i] << " ";
		}
		cout << endl;
	}
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
	
	m_dbias.resize(m_nOutput);
	m_dweights.resize(m_nOutput);
	// 平均0.0f、標準偏差 1/sqrt(nInput) 正規分布
	normal_distribution<float> dist(0.0f, (float)(1/sqrt((double)m_nInput)));
	for(int o = 0; o != m_nOutput; ++o) {
		m_weights[o].resize(m_nInput);
		m_dweights[o].resize(m_nInput);
		m_bias[o] = dist(mt);			//	Xavier初期化
		for(int i = 0; i < m_nInput; ++i) {
			m_weights[o][i] = dist(mt);			//	Xavier初期化
			m_dweights[o][i] = 0.0f;
		}
	}
}
void AffineMap::init_dweight() {
	for(int o = 0; o != m_nOutput; ++o) {
		m_dbias[o] = 0.0f;
		for(int i = 0; i != m_nInput; ++i) {
			m_dweights[o][i] = 0.0f;
		}
	}
}
void AffineMap::set_weight(const std::vector<std::vector<float>>& w) {		//	w はバイアスを含む
	for(int o = 0; o != m_nOutput; ++o) {
		m_bias[o] = w[o][0];
		for(int i = 0; i != m_nInput; ++i) {
			m_weights[o][i] = w[o][i+1];
		}
	}
}
//	順伝播、inputs にバイアス分は含まない
//			out[o] = ∑ inputs[i]*weights[o][i] + weights[o][0]
void AffineMap::forward(const vector<float>& inputs) {
	last_inputs = &inputs[0];
	for(int o = 0; o != m_nOutput; ++o) {
		float sum = m_bias[o];			//	バイアス
		for(int i = 0; i != m_nInput; ++i) {
			sum += inputs[i] * m_weights[o][i];
		}
		m_outputs[o] = sum;
	}
}
//	逆伝播
//			∂L/∂Wi = grad[i] * ∂y/∂Wi = grad[i] * inputs[i]
void AffineMap::backward(const vector<float>& grad) {
	for (int o = 0; o != m_nOutput; ++o) {
		m_dbias[o] += grad[o];
	}
	for(int i = 0; i != m_nInput; ++i) {
		m_grad[i] = 0.0f;
		for(int o = 0; o != m_nOutput; ++o) {
			m_dweights[o][i] += last_inputs[i] * grad[o];
			m_grad[i] += m_weights[o][i] * grad[o];
		}
	}
}
void AffineMap::update_weights(float alpha) {
	for(int o = 0; o != m_nOutput; ++o) {
		m_bias[o] -= alpha * m_dbias[o];
		for(int i = 0; i != m_nInput; ++i) {
			m_weights[o][i] -= alpha * m_dweights[o][i];
			//m_dweights[0][i] = 0.0f;
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
	cout << " outputs[]: ";
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
void AFtanh::backward(const vector<float>& grad) {
	for(int i = 0; i != m_nInput; ++i) {
		m_grad[i] = (1.0f - m_outputs[i]*m_outputs[i]) * grad[i];
	}
}

