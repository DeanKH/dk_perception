#include "dk_perception/dnn/deep_lsd.hpp"

#include <numeric>
#include <thread>

namespace dklib::dnn {
DeepLsdAttractionField::DeepLsdAttractionField(const std::filesystem::path& model_path,
                                               InferenceDevice inference_device, InputSize input_size)
    : inference_device_(inference_device), input_size_(input_size) {
  env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DeepLSD");
  session_options_ = Ort::SessionOptions();
  session_options_.SetIntraOpNumThreads(std::thread::hardware_concurrency());
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  switch (inference_device_) {
    case InferenceDevice::kCPU:
      break;
    case InferenceDevice::kCUDA: {
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = 0;  // Use the first CUDA device
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchDefault;
      cuda_options.gpu_mem_limit = 0;  // Use all available GPU memory
      cuda_options.arena_extend_strategy = 1;
      cuda_options.do_copy_in_default_stream = 1;
      cuda_options.has_user_compute_stream = 0;
      cuda_options.default_memory_arena_cfg = nullptr;
      session_options_.AppendExecutionProvider_CUDA(cuda_options);
      session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    } break;
    default:
      throw std::runtime_error("Unsupported inference device");
  }

  Ort::AllocatorWithDefaultOptions allocator;
  session_ = std::make_unique<Ort::Session>(env_, model_path.string().c_str(), session_options_);

  configureInOutNodes();
}

void DeepLsdAttractionField::configureInOutNodes() {
  if (!session_) {
    throw std::runtime_error("Session is not initialized.");
  }

  auto num_input_nodes = session_->GetInputCount();
  input_node_names_.reserve(num_input_nodes);
  for (size_t i = 0; i < num_input_nodes; ++i) {
    Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
    input_node_names_.push_back(input_name.release());
    input_node_shapes_.push_back(session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  auto num_output_nodes = session_->GetOutputCount();
  output_node_names_.reserve(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; ++i) {
    Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
    output_node_names_.push_back(output_name.release());
    output_node_shapes_.push_back(session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
}

DeepLsdAttractionField::Result DeepLsdAttractionField::inference(const cv::Mat& input) {
  cv::Mat normalized = preprocessImage(input);
  auto output_tensor = inferenceImpl(normalized);
  return postProcessImage(output_tensor);
}

std::vector<Ort::Value> DeepLsdAttractionField::inferenceImpl(const cv::Mat& preprocessed_image) {
  // input_node_shapes_[0] = {1, 3, 720, 1280} の形にpreprocessed_imageを変換
  assert(preprocessed_image.size() == cv::Size(1280, 720));
  assert(preprocessed_image.depth() == CV_32F);
  assert(preprocessed_image.channels() == 3);

  // Calculate input tensor size
  const size_t input_tensor_size =
      std::accumulate(input_node_shapes_[0].begin(), input_node_shapes_[0].end(), 1, std::multiplies<size_t>());
  std::cout << "Input tensor size: " << input_tensor_size << std::endl;

  // Create a vector to hold input tensor data
  std::vector<float> input_tensor_values(input_tensor_size);

  // ONNX Runtime expects data in NCHW format (batch, channel, height, width)
  // but OpenCV stores images in HWC format
  // We need to transpose from HWC to NCHW

  const int height = preprocessed_image.rows;          // 720
  const int width = preprocessed_image.cols;           // 1280
  const int channels = preprocessed_image.channels();  // 3

  // For each pixel and channel, copy the data to the correct position in the tensor
  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        // In NCHW format: (0 * C * H * W) + (c * H * W) + (h * W) + w
        // Where 0 is the batch index (single image)
        const size_t tensor_idx = c * height * width + h * width + w;

        // Get value from OpenCV Mat (HWC format)
        input_tensor_values[tensor_idx] = preprocessed_image.at<cv::Vec3f>(h, w)[c];
      }
    }
  }

  // Memory info for the input tensor
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create input tensor
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size,
                                                            input_node_shapes_[0].data(), input_node_shapes_[0].size());

  // Run inference
  std::vector<Ort::Value> output_tensors;
  try {
    output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, 1,
                                   output_node_names_.data(), output_node_names_.size());
  } catch (const Ort::Exception& e) {
    throw std::runtime_error("ONNX Runtime inference failed: " + std::string(e.what()));
  }

  return output_tensors;
}

DeepLsdAttractionField::Result DeepLsdAttractionField::postProcessImage(const std::vector<Ort::Value>& tensor) {
  // df_norm = tensor[0]
  // df = tensor[1]
  // line_level = tensor[2]

  Result result;
  if (tensor.size() != 3) {
    throw std::runtime_error("Expected 3 output tensors, got " + std::to_string(tensor.size()));
    // return result;  // Return empty result if tensor size is not as expected
  }

  const std::vector<int64_t> df_norm_shape = tensor[0].GetTensorTypeAndShapeInfo().GetShape();
  // print shape
  std::cout << "df_norm shape: ";
  for (const auto& dim : df_norm_shape) {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  const float* const df_norm_data = tensor[0].GetTensorData<float>();
  result.df_norm = cv::Mat(df_norm_shape[1], df_norm_shape[2], CV_32F, (void*)df_norm_data);

  const std::vector<int64_t> df_shape = tensor[1].GetTensorTypeAndShapeInfo().GetShape();
  const float* const df_data = tensor[1].GetTensorData<float>();
  result.df = cv::Mat(df_shape[1], df_shape[2], CV_32F, (void*)df_data);

  const std::vector<int64_t> line_level_shape = tensor[2].GetTensorTypeAndShapeInfo().GetShape();
  const float* const line_level_data = tensor[2].GetTensorData<float>();
  result.line_level = cv::Mat(line_level_shape[1], line_level_shape[2], CV_32F, (void*)line_level_data);

  return result;
}

}  // namespace dklib::dnn