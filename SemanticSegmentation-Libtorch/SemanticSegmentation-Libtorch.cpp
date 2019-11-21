#include <iostream>
#include "DataSet/COCODataSet.h"
#include "models/segmentation/SegmentationModel.h"
#include "utills/metric_logger.h"
#include <chrono>

torch::DeviceType device_type;
const int64_t kTrainBatchSize = 2;
using namespace std;

void genarateColormap(std::vector<cv::Scalar>& map, int64_t numclass)
{

	for (int64_t index = 0; index < numclass; index++)
	{
		auto r = (double)(rand() % 255);
		auto g = (double)(rand() % 255);
		auto b = (double)(rand() % 255);

		map.push_back(cv::Scalar(r, g, b));
	}
}

torch::Tensor criterion(
	std::unordered_map<std::string, torch::Tensor> inputs, torch::Tensor target)
{
	std::map<std::string, torch::Tensor> losses;
	
	for (auto loss : inputs)
	{
		losses[loss.first] = torch::nll_loss2d(torch::log_softmax(loss.second,1), target, {}, 1, 255);
	}

	if (losses.size() == 1)
	{
		return losses["out"];
	}

	return losses["out"] + 0.5 * losses["aux"];
}

int max_iter = 30;

int main()
{
	auto meters = MetricLogger(" ");
	auto start_training_time = chrono::system_clock::now();

	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Training on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "Training on CPU." << std::endl;
		device_type = torch::kCPU;
	}

	torch::Device device(device_type);

	SegmentationModel segnet;
	segnet->deeplabv3_resnet101(false,3);

	//torch::load(segnet, "deeplabv3_resnet101.pt");

	segnet->train();
	segnet->to(device);
	segnet->aux_ = true;

	//for (auto param : segnet->named_parameters())
	//{
	//	std::cout << param.key() << std::endl;
	//}

	auto train_dataset = COCODataSet("annotations/instances_train2017.json", "D:/GIT/pytorch-cpp/COCOImage/train2017", true, {0,17,18})
		.map(torch::data::transforms::Stack<>());
	const size_t train_dataset_size = train_dataset.size().value();
	
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset),
		torch::data::DataLoaderOptions().batch_size(kTrainBatchSize).workers(0));

	std::cout << train_dataset_size << std::endl;

	std::vector<torch::Tensor> trainable_params;

	auto params = segnet->classifier_->named_parameters(true /*recurse*/);
	for (auto& param : params)
	{
		auto layer_name = param.key();

		if (param.value().requires_grad())
		{
			trainable_params.push_back(param.value());
		}
	}

	params = segnet->aux_classifier_->named_parameters(true /*recurse*/);
	for (auto& param : params)
	{
		auto layer_name = param.key();

		if (param.value().requires_grad())
		{
			trainable_params.push_back(param.value());
		}
	}
	
	torch::optim::Adam optimizer(trainable_params, torch::optim::AdamOptions(1e-3 /*learning rate*/));

	std::map<std::string, torch::Tensor> loss_map;
	std::chrono::duration<double> data_time, batch_time;
	int iteration = 0;
	float eta_seconds;
	std::string eta_string;
	int days, hours, minutes;
	auto end = chrono::system_clock::now();

	int checkpoint_period = 2500;

	for (int i = 0; i < max_iter; i++)
	{
		data_time = chrono::system_clock::now() - end;
		iteration += 1;

		for (const auto& batch : *train_loader)
		{
			auto data = batch.data;
			auto targets = batch.target;

			data = data.to(torch::kF32);
			targets = targets.to(torch::kLong);

			data = data.to(device);
			targets = targets.to(device);

			optimizer.zero_grad();

			auto output = segnet->forward(data);

			auto loss = criterion(output, targets);

			loss_map["loss"] = loss;

			meters.update(loss_map);

			loss.backward();
			optimizer.step();
	
		}

		batch_time = std::chrono::system_clock::now() - end;
		end = std::chrono::system_clock::now();
		meters.update(std::map<string, float>{ {"time", static_cast<float>(batch_time.count())}, { "data", static_cast<float>(data_time.count()) }});
		eta_seconds = meters["time"].global_avg() * (max_iter - iteration);
		days = eta_seconds / 60 / 60 / 24;
		hours = eta_seconds / 60 / 60 - days * 24;
		minutes = eta_seconds / 60 - hours * 60 - days * 24 * 60;
		eta_string = to_string(days) + " day " + to_string(hours) + " h " + to_string(minutes) + " m";
		if (iteration % 20 == 0 || iteration == max_iter) {
			std::cout << "eta: " << eta_string << meters.delimiter_ << "iter: " << iteration << meters.delimiter_ << meters << meters.delimiter_ << meters.delimiter_;
		}

		if (iteration % checkpoint_period == 0)
			torch::save(segnet, "model_" + to_string(iteration) + ".pt");
		if (iteration == max_iter)
			torch::save(segnet, "model_final.pth");
	}

	chrono::duration<double> total_training_time = chrono::system_clock::now() - start_training_time;
	days = total_training_time.count() / 60 / 60 / 24;
	hours = total_training_time.count() / 60 / 60 - days * 24;
	minutes = total_training_time.count() / 60 - hours * 60 - days * 24 * 60;
	std::cout << "Total training time: " << to_string(days) + " day " + to_string(hours) + " h " + to_string(minutes) + " m" << " ( " << total_training_time.count() / max_iter << "s / it)";
}
