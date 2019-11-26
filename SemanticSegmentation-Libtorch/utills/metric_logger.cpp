#include "metric_logger.h"

using namespace std;

ConfusionMatrix::ConfusionMatrix(int64_t cls, torch::Device deviece)
	: _num_classes(cls) 
{
	_mat = torch::zeros({ _num_classes,_num_classes },c10::TensorOptions().dtype(torch::kFloat64).device(deviece));
}

void ConfusionMatrix::update(torch::Tensor a, torch::Tensor b)
{
	torch::NoGradGuard;

	auto inds = _num_classes * a + b;

	_mat += torch::bincount(inds, {}, std::pow(_num_classes, 2)).reshape({ _num_classes, _num_classes });
}

void ConfusionMatrix::reset()
{
	_mat.zero_();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ConfusionMatrix::compute() const
{
	auto acc_global =  torch::diag(_mat).sum() / _mat.sum();
	auto acc = torch::diag(_mat).sum() / _mat.sum(1);
	auto iou = torch::diag(_mat) / (_mat.sum(1) + _mat.sum(0) - torch::diag(_mat));

	return { acc_global ,acc ,iou};
}

std::ostream& operator << (std::ostream& os, const ConfusionMatrix& confusion)
{
	torch::Tensor acc_global, acc, iou;

	std::tie(acc_global, acc, iou) = confusion.compute();

	os <<"test " <<"global correct: " <<acc_global.item<float>() * 100 << "  mean IoU: " << iou.mean().item<float>();

	os << "\n";

	return os;
}

void SmoothedValue::update(float value) 
{
	if (deque.size() < 21)
		deque.push_back(value);
	else
		deque.pop_front();
	count += 1;
	total += value;
}

float SmoothedValue::median() const 
{
	auto d = torch::zeros({ static_cast<int64_t>(deque.size()) });
	for (int i = 0; i < deque.size(); ++i)
		d[i] = deque[i];
	return d.median().item<float>();
}

float SmoothedValue::avg() const 
{
	auto d = torch::zeros({ static_cast<int64_t>(deque.size()) });
	for (int i = 0; i < deque.size(); ++i)
		d[i] = deque[i];
	return d.mean().item<float>();
}

float SmoothedValue::global_avg() const 
{
	return total / count;
}

MetricLogger::MetricLogger(std::string delimiter) : delimiter_(delimiter) {}

void MetricLogger::update(std::map<std::string, torch::Tensor> losses) 
{
	for (auto i = losses.begin(); i != losses.end(); ++i)
		meters[i->first].update(i->second.item<float>());
}

void MetricLogger::update(std::map<std::string, float> losses)
{
	for (auto i = losses.begin(); i != losses.end(); ++i)
		meters[i->first].update(i->second);
}

SmoothedValue MetricLogger::operator[](std::string attr) 
{
	return meters[attr];
}

std::ostream& operator << (std::ostream& os, const MetricLogger& bml) 
{
	for (auto i = bml.meters.begin(); i != bml.meters.end(); ++i) {
		os << i->first << ": " << i->second.median() << " (" << i->second.global_avg() << ")" << bml.delimiter_;
	}
	os << "\n";
	return os;
}
