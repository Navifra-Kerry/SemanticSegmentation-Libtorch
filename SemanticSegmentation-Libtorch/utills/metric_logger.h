#pragma once
#include <deque>
#include <string>
#include <map>
#include <torch/torch.h>


class ConfusionMatrix
{
public:

	ConfusionMatrix(int64_t cls, torch::Device deviece);
	void update(torch::Tensor a, torch::Tensor b);
	void reset();

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>  compute() const;

	int64_t _num_classes;
	torch::Tensor _mat;
	friend std::ostream& operator << (std::ostream& os, const ConfusionMatrix& confusion);
};

class SmoothedValue 
{

public:
	void update(float value);
	float median() const;
	float avg() const;
	float global_avg() const;

private:
	std::deque<float> deque;
	float total = 0;
	int count = 0;
};

class MetricLogger {

public:
	MetricLogger(std::string delimiter = "\t");
	void update(std::map<std::string, torch::Tensor> losses);
	void update(std::map<std::string, float> losses);
	SmoothedValue operator[](std::string attr);
	std::string to_string();
	std::string delimiter_;

private:
	std::map<std::string, SmoothedValue> meters;


	friend std::ostream& operator << (std::ostream& os, const MetricLogger& bml);
};
