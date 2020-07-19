#include <iostream>
#include "training.h"
#include <filesystem>

torch::DeviceType device_type;
const int64_t kTrainBatchSize = 2;
const int64_t class_num = 3; //0 is background;
const int max_iter = 15;

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
void training(std::string data_dir)
{
try
{
	auto start_training_time = chrono::system_clock::now();


	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Training on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "Training on CPU." << std::endl;
		device_type = torch::kCPU;
	}

	if (std::filesystem::exists("resnet101_Python.pt") != true)
	{
		std::cout << "resnet101_Python.pt file could not be found. First run Convert.py and copy it to the run folder." << std::endl;

		return;
	}

	torch::Device device(device_type);

	SegmentationModel segnet;
	segnet->deeplabv3_resnet101(false, class_num);
	segnet->to(device);


	auto val_dataset = COCODataSet(data_dir + "annotations/instances_val2017.json", data_dir + "val2017", true, { 0,17,18 })
		.map(torch::data::transforms::Stack<>());
	const size_t va_dataset_size = val_dataset.size().value();
	
	auto val_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(val_dataset),
		torch::data::DataLoaderOptions().batch_size(kTrainBatchSize).workers(8));


	auto train_dataset = COCODataSet(data_dir + "annotations/instances_train2017.json", data_dir + "train2017", true, { 0,17,18 })
		.map(torch::data::transforms::Stack<>());
	const size_t train_dataset_size = train_dataset.size().value();
	
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset),
		torch::data::DataLoaderOptions().batch_size(kTrainBatchSize).workers(0));



	std::vector<torch::Tensor> trainable_params;
	
	
	auto params = segnet->_classifier->named_parameters(true /*recurse*/);
	for (auto& param : params)
	{
		auto layer_name = param.key();
	
		if (param.value().requires_grad())
		{
			trainable_params.push_back(param.value());
		}
	}
	
	params = segnet->_backbone->named_parameters(true /*recurse*/);
	for (auto& param : params)
	{
		if (param.value().requires_grad())
		{
			trainable_params.push_back(param.value());
		}
	}

	torch::optim::SGD optimizer(trainable_params, torch::optim::SGDOptions(0.01 /*learning rate*/).momentum(0.9).weight_decay(1e-4));

	for (int i = 0; i < max_iter; i++)
	{
		train_one_epoch(segnet, device, train_loader, optimizer, max_iter);
		evaluate(segnet, device, val_loader, class_num);
		torch::save(segnet, "model_" + to_string(i) + ".pt");
	}

	torch::save(segnet, "model_final.pt");
}
catch (std::exception ex)
{
	std::cout << ex.what();
}
	
}

void inference(std::string data_dir, std::string modelpath)
{
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

	torch::load(segnet, modelpath);

	segnet->train();
	segnet->to(device);

	std::vector<cv::Scalar> colomap;
	genarateColormap(colomap, 3);

	cv::Mat image = cv::imread(data_dir, cv::IMREAD_COLOR);
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

const cv::String keys =
"{@train		 |		| train or inference}"
"{@data_dir      |<none>| path to coco dataset root folder  example input D:\\COCOImage\\}"
"{@mode_path     |<none>| Only Need inference}"
"{help h usage ? |      | print this message   }";


int main(int argc, char* argv[])
{
	try
	{
		cv::CommandLineParser parser(argc, argv, keys);
		parser.about("Libtorch Example");

		if (parser.has("help") || argc == 1) {
			parser.printMessage();
			return 0;
		}

		std::string train = parser.get<cv::String>(0);
		std::string data_dir = parser.get<cv::String>(1);	
		std::string modelPath = parser.get<cv::String>(2);

		if (train == "train")
		{
			training(data_dir);
		}
		else if(train == "inference")
		{
			inference(data_dir, modelPath);
		}
		else
		{
			std::cout << "Not Found Run Mode Input Mode is " + train;
		}
	}
	catch (std::exception ex)
	{
		std::cout << ex.what();
	}
}
