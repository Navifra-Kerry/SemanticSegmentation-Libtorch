#include <iostream>
#include "DataSet/COCODataSet.h"
#include "models/segmentation/SegmentationModel.h"

torch::DeviceType device_type;
const int64_t kTrainBatchSize = 8;

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

int main()
{
	torch::manual_seed(1);

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

	segnet->eval();
	segnet->to(device);

	for (auto param : segnet->named_parameters())
	{
		std::cout << param.key() << std::endl;
	}

#ifdef NET_TEST
	for (auto param : segnet->named_parameters())
	{
		std::cout << param.key() << std::endl;
	}

	{
		cv::Mat image = cv::imread("D:/GIT/pytorch-cpp/COCOImage/val2017/000000001000.jpg", cv::IMREAD_COLOR);
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
		tensor_image = tensor_image.to(device_type);

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

		std::vector<cv::Scalar> colomap;
		genarateColormap(colomap, 21);

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

		cv::addWeighted(image, 1, mask, 0.8, 0, image);
		cv::imshow("Image", image);	
		cv::waitKey(0);
	}
#endif

	auto train_dataset = COCODataSet("annotations/instances_val2017.json", "D:/GIT/pytorch-cpp/COCOImage/val2017", true, {0,17,18})
		.map(torch::data::transforms::Stack<>());
	const size_t train_dataset_size = train_dataset.size().value();
	
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset),
		torch::data::DataLoaderOptions().batch_size(kTrainBatchSize).workers(0));

	std::cout << train_dataset_size << std::endl;
	
	for (const auto& batch : *train_loader)
	{
		auto data = batch.data;

		data = data.to(torch::kF32);
		data = data.to(device_type);

		auto output = segnet->forward(data);

	}

}
