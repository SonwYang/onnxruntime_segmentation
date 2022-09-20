#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>
#include <tensorrt_provider_options.h>

using namespace std;

typedef struct
{
    float r;
    int dw;
    int dh;
    int new_unpad_w;
    int new_unpad_h;
    bool flag;
} YOLOPScaleParams;


void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs, int target_height, int target_width, YOLOPScaleParams &scale_params)
{
    if (mat.empty()) return;
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 144));

    // scale ratio (new / old) new_shape(h, w)
    float w_r = (float) target_width / (float) img_width;
    float h_r = (float) target_height / (float) img_height;
    float r = std::min(w_r, h_r);

    // compute padding
    int new_unpad_w = static_cast<int>((float) img_width * r);
    int new_unpad_h = static_cast<int>((float) img_height * r);
    int pad_w = target_width - new_unpad_w;
    int pad_h = target_height - new_unpad_h;

    int dw = pad_w / 2;
    int dh = pad_h / 2;

    // resize with unscaling
    cv::Mat new_unpad_mat = mat.clone();
    cv::resize(new_unpad_mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // record scale params
    scale_params.r = r;
    scale_params.dw = dw;
    scale_params.dh = dh;
    scale_params.new_unpad_w = new_unpad_w;
    scale_params.new_unpad_h = new_unpad_h;
    scale_params.flag = true;
}


void normalize_inplace(cv::Mat &mat_inplace, const float mean[3], const float scale[3])
{
    if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
    for(unsigned int i = 0; i < mat_inplace.rows; ++i)
    {
        cv::Vec3f *p = mat_inplace.ptr<cv::Vec3f>(i);
        for(unsigned int j = 0; j < mat_inplace.cols; ++j)
        {
            p[j][0] = (p[j][0] - mean[0]) * scale[0];
            p[j][1] = (p[j][1] - mean[1]) * scale[1];
            p[j][2] = (p[j][2] - mean[2]) * scale[2];
        }
    }
}

Ort::Value create_tensor(const cv::Mat &mat,
                        const std::vector<int64_t> &tensor_dims,
                        const Ort::MemoryInfo &memory_info_handler,
                        std::vector<float> &tensor_value_handler,
                        unsigned int data_format)
{
    const unsigned int rows = mat.rows;
    const unsigned int cols = mat.cols;
    const unsigned int channels = mat.channels();

    cv::Mat mat_ref;
    if(mat.type() != CV_32FC(channels)) mat.convertTo(mat_ref, CV_32FC(channels));
    else mat_ref = mat;

    if (tensor_dims.size() != 4) throw std::runtime_error("dims mismatch. ");
    if (tensor_dims.at(0) != 1) throw std::runtime_error("batch != 1");


    // turn CHW mat into tensor
    if (data_format == 0)
    {
        const unsigned int target_channel = tensor_dims.at(1);
        const unsigned int target_height = tensor_dims.at(2);
        const unsigned int target_width = tensor_dims.at(3);
        const unsigned int target_tensor_size = target_channel * target_height * target_width;
        if (target_channel != channels) throw std::runtime_error("channels mismatch.");

        tensor_value_handler.resize(target_tensor_size);

        std::vector<cv::Mat> mat_channels;
        cv::split(mat_ref, mat_channels);

        for (unsigned int i = 0; i < channels; ++i)
            std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
                        mat_channels.at(i).data, target_height * target_width * sizeof(float));

        return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
                                            target_tensor_size, tensor_dims.data(),
                                            tensor_dims.size());
    }

    // turn HWC mat into tensor
    const unsigned int target_height = tensor_dims.at(1);
    const unsigned int target_width = tensor_dims.at(2);
    const unsigned int target_channel = tensor_dims.at(3);
    const unsigned int target_tensor_size = target_channel * target_height * target_width;
    if (target_channel != channels) throw std::runtime_error("channel mismatch!!!");
    tensor_value_handler.resize(target_tensor_size);

    cv::Mat resize_mat_ref;
    if (target_height != rows || target_width != cols)
        cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
    else resize_mat_ref = mat_ref;

    std::memcpy(tensor_value_handler.data(), resize_mat_ref.data, target_tensor_size * sizeof(float));

    return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
                                            target_tensor_size, tensor_dims.data(),
                                            tensor_dims.size());
}


Ort::Value transform_tensor(const cv::Mat &mat, 
                    const unsigned int img_height, 
                    const unsigned int img_width)
{
    cv::Mat canvas;
    
    // normalize params
    const float mean_vals[3] = {0.485f, 0.456f, 0.406f};
    const float scale_vals[3] = {1.f / 0.229f, 1.f / 0.224f, 1.f / 0.225f};

    // init dynamic input dims
    std::vector<std::vector<int64_t>> dynamic_input_node_dims; // >=1 inputs.
    unsigned int dynamic_input_height = img_height; // init only, will change according to input mat.
    unsigned int dynamic_input_width = img_width; // init only, will change according to input mat.

    dynamic_input_node_dims.push_back({1, 3, dynamic_input_height, dynamic_input_width});
    dynamic_input_node_dims.at(0).at(2) = dynamic_input_height;
    dynamic_input_node_dims.at(0).at(3) = dynamic_input_width;
    unsigned int dynamic_input_tensor_size = 1 * 3 * dynamic_input_height * dynamic_input_width; // init only, will change according to input mat.
    
    std::vector<float> dynamic_input_values_handler;
    dynamic_input_values_handler.resize(dynamic_input_tensor_size);
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeDefault);

    cv::cvtColor(mat, canvas, cv::COLOR_BGR2RGB);
    canvas.convertTo(canvas, CV_32FC3, 1.f / 255.f, 0.f);

    normalize_inplace(canvas, mean_vals, scale_vals);

    return create_tensor(canvas, dynamic_input_node_dims.at(0), 
                            memory_info_handler, dynamic_input_values_handler, 0);
}


int main(int, char**) 
{
    // init
    Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions ortAlloc;

    constexpr int64_t numChannles = 3;
    constexpr int64_t width = 768;
    constexpr int64_t height = 768;
    constexpr int64_t numClasses = 2;
    constexpr int64_t numInputElements = numChannles * height * width;
    constexpr int64_t numOutputElements = numClasses * height * width;

    const string imgFile = "/home/yp/self-driving/sense/YOLOP/inference/grapery/IMG_20220823_141235.jpg";
    auto modelPath = "/home/yp/self-driving/sense/YOLOP/weights/yolop-768-768_seg.onnx";
    const bool isGPU = true;
    const bool isTRT = true;
    
    sessionOptions = Ort::SessionOptions();

    // cuda setting
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;
    cudaOption.device_id = 0;

     // tensorrt
    OrtTensorRTProviderOptionsV2 trtOptions{};
    // trtOptions.device_id = 0;
    trtOptions.has_user_compute_stream = 0;
    trtOptions.trt_max_partition_iterations = 1000;
    trtOptions.trt_min_subgraph_size = 1;
    trtOptions.trt_max_workspace_size = 1 << 30;
    trtOptions.trt_fp16_enable = false;
    trtOptions.trt_int8_enable = false;
    trtOptions.trt_int8_calibration_table_name = "";
    trtOptions.trt_int8_use_native_calibration_table = false;
    trtOptions.trt_dla_core = 0;
    trtOptions.trt_dla_enable = false;
    trtOptions.trt_dump_subgraphs = false;
    trtOptions.trt_engine_cache_enable = true;
    trtOptions.trt_engine_cache_path = "/home/yp/onnx/onnxruntime-segmentation/cache";
    trtOptions.trt_engine_decryption_lib_path = "";
    trtOptions.trt_force_sequential_engine_build = false;

    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else if (isTRT)
    {
        std::cout << "Inference device: Trt" << std::endl;
        sessionOptions.AppendExecutionProvider_TensorRT_V2(trtOptions);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }


    // // create session
    printf("Creating session.");
    Ort::Session session(env, modelPath, sessionOptions);

    // // Get the number of input
    // Ort::AllocatorWithDefaultOptions ortAlloc;

    // size_t num_input_nodes = session.GetInputCount();
    // std::vector<const char*> input_node_names(num_input_nodes);
    // std::vector<int64_t> input_node_dims;

    // std::cout << "Number of inputs: " << num_input_nodes << std::endl;

    // // define shape
    // const array<int64_t, 4> inputShape = {1, numChannles, height, width};
    // const array<int64_t, 4> outputShape = {1, numClasses, height, width};

    // // define array
    // array<float, numInputElements> input;
    // array<float, numOutputElements> results;

    // // load image and turn mat into Tensor
    cv::Mat image = cv::imread(imgFile);
    cv::Mat resize_mat;
    YOLOPScaleParams scale_params;

    resize_unscale(image, resize_mat, height, width, scale_params);

    float r = scale_params.r;
    int dw = scale_params.dw;
    int dh = scale_params.dh;
    int new_unpad_w = scale_params.new_unpad_w;
    int new_unpad_h = scale_params.new_unpad_h;

    transform_tensor(image, height, width);

    Ort::Value input_tensor = transform_tensor(resize_mat, height, width);

    // define names
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;

    inputNames.push_back(session.GetInputName(0, ortAlloc));
    outputNames.push_back(session.GetOutputName(0, ortAlloc));

    std::cout << "Input name: " << inputNames[0] << std::endl;
    std::cout << "output name: " << outputNames[0] << std::endl;

   
    auto start = std::chrono::system_clock::now();
    
    // get inference out pixel to pixel
    auto outputTensor =  session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor, 1, outputNames.data(), 1);

    // post process
    float* prob = outputTensor.front().GetTensorMutableData<float>();
    
    cv::Mat outImg(new_unpad_h, new_unpad_w, CV_8UC1);
    
    for (int row = dh; row < dh + new_unpad_h; ++row)
    {
        uchar *uc_pixel = outImg.data + (row - dh) * outImg.step;
        for (int col = dw; col < dw + new_unpad_w; ++col)
        {
            if (prob[row * width + col] > prob[width * height + row * width + col])
            {
                uc_pixel[col - dw] = 0;
            }
            else
            {
                uc_pixel[col - dw] = 255;
            }
        }
    }

    
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms +post process" << std::endl;
    cv::imwrite("../result.png", outImg);
 
    return 0;
}


// cd .. && rm -r build && mkdir build && cd build && cmake .. && make && ./seg