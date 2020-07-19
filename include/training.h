#pragma once
#include "DataSet/COCODataSet.h"
#include "models/segmentation/SegmentationModel.h"
#include "utills/metric_logger.h"
#include "loss.h"
#include <chrono>
#include <iostream>

using namespace std;


template<typename Net, typename DataLoader>
void train_one_epoch(Net model, torch::Device device,
	DataLoader& data_loader,
	torch::optim::Optimizer& optimizer, int64_t max_iter)
{
	model->train();
	auto meters = MetricLogger(" ");

	std::map<std::string, torch::Tensor> loss_map;
	std::chrono::duration<double> data_time, batch_time;
	int iteration = 0;
	float eta_seconds;
	std::string eta_string;
	int days, hours, minutes;
	auto end = chrono::system_clock::now();
	data_time = chrono::system_clock::now() - end;

	for (const auto& batch : *data_loader)
	{
		auto data = batch.data;
		auto targets = batch.target;

		data = data.to(torch::kF32);
		targets = targets.to(torch::kLong);

		data = data.to(device);
		targets = targets.to(device);

		optimizer.zero_grad();

		auto output = model->forward(data);

		auto loss = criterion(output, targets);

		loss_map["loss"] = loss;

		meters.update(loss_map);

		loss.backward();
		optimizer.step();

		batch_time = std::chrono::system_clock::now() - end;
		end = std::chrono::system_clock::now();
		meters.update(std::map<string, float>{ {"time", static_cast<float>(batch_time.count())}, { "data", static_cast<float>(data_time.count()) }});
		eta_seconds = meters["time"].global_avg() * (max_iter - iteration);
		days = eta_seconds / 60 / 60 / 24;
		hours = eta_seconds / 60 / 60 - days * 24;
		minutes = eta_seconds / 60 - hours * 60 - days * 24 * 60;
		eta_string = to_string(days) + " day " + to_string(hours) + " h " + to_string(minutes) + " m";
		if (iteration % 20 == 0 || iteration == max_iter) {
			std::cout << "eta: " << eta_string << " "<<"iter: " << iteration << meters;
		}
	}
}

template<typename Net, typename DataLoader>
void evaluate(Net model, torch::Device device,
	DataLoader& data_loader,
	int64_t num_classes)
{
	model->eval();

	torch::NoGradGuard nograd;

	ConfusionMatrix confmat(num_classes, device);

	for (const auto& batch : *data_loader)
	{
		auto data = batch.data;
		auto targets = batch.target;

		data = data.to(torch::kF32);
		targets = targets.to(torch::kLong);

		data = data.to(device);
		targets = targets.to(device);

		auto output = model->forward(data);

		auto out = output["out"];

		confmat.update(targets.flatten(), out.argmax(1).flatten());
	}

	std::cout << confmat;
}