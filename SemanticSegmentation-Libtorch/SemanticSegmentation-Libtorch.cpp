#include <iostream>
#include "DataSet/COCODataSet.h"

torch::DeviceType device_type;
const int64_t kTrainBatchSize = 24;

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

	auto train_dataset = COCODataSet("annotations/instances_train2017.json", "D:/GIT/pytorch-cpp/COCOImage/train2017", false)
		.map(torch::data::transforms::Stack<>());
	const size_t train_dataset_size = train_dataset.size().value();

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset),
		torch::data::DataLoaderOptions().batch_size(kTrainBatchSize).workers(0));

	for (const auto& batch : *train_loader)
	{
		int a = 0;
	}

}
