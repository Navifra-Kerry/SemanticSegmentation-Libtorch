#include <iostream>
#include "DataSet/COCODataSet.h"
#include "models/segmentation/SegmentationModel.h"
#include "utills/metric_logger.h"
#include <chrono>

torch::DeviceType device_type;
const int64_t kTrainBatchSize = 8;
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

void training()
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
	segnet->deeplabv3_resnet101(false, 3);

	segnet->train();
	segnet->to(device);
	segnet->aux_ = true;

	auto train_dataset = COCODataSet("annotations/instances_train2017.json", "D:/GIT/pytorch-cpp/COCOImage/train2017", true, { 0,17,18 })
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

	//::optim::Adam optimizer(trainable_params, torch::optim::AdamOptions(1e-3 /*learning rate*/));
	torch::optim::SGD optimizer(trainable_params, torch::optim::SGDOptions(0.01 /*learning rate*/).momentum(0.9).weight_decay(1e-4));

	std::map<std::string, torch::Tensor> loss_map;
	std::chrono::duration<double> data_time, batch_time;
	int iteration = 0;
	float eta_seconds;
	std::string eta_string;
	int days, hours, minutes;
	auto end = chrono::system_clock::now();

	int checkpoint_period = 500;

	for (int i = 0; i < max_iter; i++)
	{
		data_time = chrono::system_clock::now() - end;

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

		}

		torch::save(segnet, "model_" + to_string(i) + ".pt");
	}


	torch::save(segnet, "model_final.pt");

	chrono::duration<double> total_training_time = chrono::system_clock::now() - start_training_time;
	days = total_training_time.count() / 60 / 60 / 24;
	hours = total_training_time.count() / 60 / 60 - days * 24;
	minutes = total_training_time.count() / 60 - hours * 60 - days * 24 * 60;
	std::cout << "Total training time: " << to_string(days) + " day " + to_string(hours) + " h " + to_string(minutes) + " m" << " ( " << total_training_time.count() / max_iter << "s / it)";
}

void inference()
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
	segnet->deeplabv3_resnet101(false, 3);

	segnet->train();
	segnet->to(device);
	segnet->aux_ = true;

	std::vector<cv::Scalar> colomap;
	genarateColormap(colomap, 3);

	cv::Mat image = cv::imread("000000093673.jpg", cv::IMREAD_COLOR);
	cv::resize(image, image, cv::Size(693, 520));

	cv::Mat rgb[3];
	cv::split(image, rgb);
	// Concatenate channels
	cv::Mat rgbConcat;
	cv::vconcat(rgb[0], rgb[1], rgbConcat);
	cv::vconcat(rgbConcat, rgb[2], rgbConcat);

	// Convert Mat image to tensor C x H x W
	at::Tensor tensor_image = torch::from_blob(rgbConcat.data, { image.channels(), image.rows, image.cols }, at::kByte);

	// Normalize tensor values from [0, 255] to [0, 1]
	tensor_image = tensor_image.toType(at::kFloat);
	tensor_image = tensor_image.div_(255);
	auto normalizeChannels = torch::data::transforms::Normalize<>({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 });
	tensor_image = normalizeChannels(tensor_image);
	tensor_image = torch::stack(tensor_image);
	tensor_image = tensor_image.to(device);

	torch::NoGradGuard no_grad;

	clock_t s = (int)std::clock();
	auto out = segnet->forward(tensor_image);
	clock_t e = (int)std::clock();

	std::printf("process time %0.3f s \n", (float)(e - s) / CLOCKS_PER_SEC);

	auto pred = out["out"].argmax(1);
	auto pred1 = out["aux"].argmax(1);
	auto data = pred[0];

	////average the channels of the activations	
	cv::Mat bin_mask = cv::Mat::eye(data.size(0), data.size(1), CV_8UC1);
	data = data.clamp(0, 255).to(torch::kU8);
	data = data.to(torch::kCPU);
	std::memcpy(bin_mask.data, data.data_ptr(), sizeof(torch::kU8) * data.numel());

	cv::Mat mask_ch[3];
	mask_ch[2] = bin_mask;
	mask_ch[0] = cv::Mat::zeros(bin_mask.size(), CV_8UC1);
	mask_ch[1] = cv::Mat::zeros(bin_mask.size(), CV_8UC1);
	cv::Mat mask;
	cv::merge(mask_ch, 3, mask);

	mask.convertTo(mask, CV_8UC3);
	mask = mask;

	uchar* data_ptr = (uchar*)mask.data;

	for (int y = 0; y < mask.rows; y++)
	{
		for (int x = 0; x < mask.cols; x++)
		{
			if (data_ptr[y * mask.cols * 3 + x * 3 + 2] == 0)
			{
				continue;
			}
			else
			{
				int64_t cls = data_ptr[y * mask.cols * 3 + x * 3 + 2];
				data_ptr[y * mask.cols * 3 + x * 3] = colomap[cls][2];
				data_ptr[y * mask.cols * 3 + x * 3 + 1] = colomap[cls][1];
				data_ptr[y * mask.cols * 3 + x * 3 + 2] = colomap[cls][0];
			}
		}
	}

	cv::addWeighted(image, 1, mask, 0.7, 0, image);
	cv::imshow("Image", image);
	cv::waitKey(0);
}

int main()
{
	training();
	//inference();
}
