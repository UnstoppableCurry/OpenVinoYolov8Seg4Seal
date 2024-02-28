#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>


#include "common.h"
#define NOMINMAX
#include <windows.h>
#include <chrono>
#include <deque>
#include <iostream>
#include <vector>
#include <array>
#include <codecvt>
#include <locale>
#include <filesystem> 
#include <stdio.h> // 或者 
#include <sstream>
#include <string>

using namespace cv;
// stuff we know about the network and the input/output blobs
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int _segWidth = 640;
static const int _segHeight = 640;
static const int _segChannels = 4;
static const int CLASSES = 13;
static const int Num_box = 8400;
static const int OUTPUT_SIZE = Num_box * (CLASSES + 4 + _segChannels);//output0
static const int OUTPUT_SIZE1 = _segChannels * _segWidth * _segHeight;//output1


static const float CONF_THRESHOLD = 0.1;
static const float NMS_THRESHOLD = 0.5;
static const float MASK_THRESHOLD = 0.9;
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output0";//detect
const char* OUTPUT_BLOB_NAME1 = "output1";//mask

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}


struct OutputSeg {
	int id;             //结果类别id
	float confidence;   //结果置信度
	cv::Rect box;       //矩形框
	cv::Mat boxMask;       //矩形框内mask，节省内存空间和加快速度
};

/*void DrawPred(Mat& img, std::vector<OutputSeg> result) {
	//生成随机颜色
	std::vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < CLASSES; i++) {
		int b = 0;
		int g = 0;
		int r = 0;
		color.push_back(Scalar(b, g, r));
	}
	Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 4, 8);

		mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
		char label[100];
		printf(label, "%d   :  %.2f", result[i].id, result[i].confidence);

		//std::string label = std::to_string(result[i].id) + ":" + std::to_string(result[i].confidence);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}

	addWeighted(img, 0.5, mask, 0.8, 1, img); //将mask加在原图上面
}
*/
void DrawPred(Mat& img, std::vector<OutputSeg>  result) {
	// 生成随机颜色
	std::vector<Scalar> colors;
	srand(static_cast<unsigned>(time(0)));
	for (int i = 0; i < CLASSES; ++i) {
		int b = 0;
		int g = 0;
		int r = 0;
		colors.push_back(Scalar(b, g, r));
	}

	Mat mask = img.clone();
	for (size_t i = 0; i < result.size(); ++i) {
		int left = result[i].box.x;
		int top = result[i].box.y;

		// 绘制矩形
		rectangle(img, result[i].box, colors[result[i].id], 1, 1);
		mask(result[i].box).setTo(colors[result[i].id], result[i].boxMask);


		// 创建标签文本
		char label[100];
		// 正确使用 sprintf 来格式化字符串
		sprintf_s(label, "%d: %.1f", result[i].id, result[i].confidence);

		// 计算文本大小
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		// 调整文本绘制的顶部位置，确保文本不会超出图像边界
		top = max(top, labelSize.height);
		// 将文本绘制在检测框的左上角，稍微调整位置以避免覆盖框边界
		putText(img, label, Point(left, top - baseLine - 1), FONT_HERSHEY_SIMPLEX, 0.5, colors[result[i].id], 2);

	}

	// 将mask加在原图上面
	addWeighted(img, 0.5, mask, 0.5, 1, img);
}
// 快速计算指数函数
inline float fast_exp(float x)
{
	union {
		uint32_t i;
		float f;
	} v{};
	v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
	return v.f;
}

// Sigmoid 函数的实现
inline float sigmoid(float x)
{
	return 1.0f / (1.0f + fast_exp(-x));
}

// 用于替换现有的 Sigmoid 计算
cv::Mat fast_sigmoid(const cv::Mat& src)
{
	// 确保源矩阵是单通道且类型为 CV_32F
	CV_Assert(src.channels() == 1 && src.type() == CV_32F);

	// 创建一个与源矩阵大小和类型相同的目标矩阵
	cv::Mat dest = cv::Mat::zeros(src.size(), CV_32F);

	// 遍历源矩阵中的所有元素
	for (int y = 0; y < src.rows; ++y)
	{
		for (int x = 0; x < src.cols; ++x)
		{
			// 对每个元素应用 sigmoid 函数
			dest.at<float>(y, x) = sigmoid(src.at<float>(y, x));
		}
	}

	return dest;  // 返回计算结果
}

ov::CompiledModel initYoloModel(std::string model_path) {
	// Step 1. Initialize OpenVINO Runtime core
	ov::Core core;
	// Step 2. Read a model    
	std::shared_ptr<ov::Model> model = core.read_model(model_path);


	// Step 4. Inizialize Preprocessing for the model
	ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
	// Specify input image format
	ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
	// Specify preprocess pipeline to input image without resizing
	ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255., 255., 255. });
	//  Specify model's input layout
	ppp.input().model().set_layout("NCHW");
	// Specify output results format
	//ppp.output().tensor().set_element_type(ov::element::f32);
	// Embed above steps in the graph
	model = ppp.build();
	ov::CompiledModel compiled_model = core.compile_model(model, "AUTO");
	// 获取输入信息
	// limiting the available parallel slack for the 'throughput' hint via the ov::hint::num_requests
	// so that certain parameters (like selected batch size) are automatically accommodated accordingly 
	//ov::CompiledModel compiled_model = core.compile_model(model, "AUTO", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), ov::hint::num_requests(2));
	return compiled_model;
}

static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h, std::vector<int>& padsize) {
	int w, h, x, y;
	float r_w = input_w / (img.cols * 1.0);
	float r_h = input_h / (img.rows * 1.0);
	if (r_h > r_w) {//¿í´óÓÚ¸ß
		w = input_w;
		h = r_w * img.rows;
		x = 0;
		y = (input_h - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = input_h;
		x = (input_w - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
	padsize.push_back(h);
	padsize.push_back(w);
	padsize.push_back(y);
	padsize.push_back(x);// int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
	//cv::imwrite("back.jpg", out);

	return out;
}
std::vector<OutputSeg> postSegProcessModelAOutput(float* prob, float* prob1, float ratio_w, float ratio_h, int padw, int padh, cv::Mat img) {
	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	std::vector<cv::Mat> picked_proposals;  //后续计算mask

	// 处理box
	int net_length = CLASSES + 4 + _segChannels;
	cv::Mat out1 = cv::Mat(net_length, Num_box, CV_32F, prob);

	for (int i = 0; i < Num_box; i++) {
		//输出是1*net_length*Num_box;所以每个box的属性是每隔Num_box取一个值，共net_length个值
		cv::Mat scores = out1(Rect(i, 4, 1, CLASSES)).clone();
		Point classIdPoint;
		double max_class_socre;
		minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
		max_class_socre = (float)max_class_socre;
		if (max_class_socre >= CONF_THRESHOLD) {
			cv::Mat temp_proto = out1(Rect(i, 4 + CLASSES, 1, _segChannels)).clone();
			picked_proposals.push_back(temp_proto.t());
			float x = (out1.at<float>(0, i) - padw) * ratio_w;  //cx
			float y = (out1.at<float>(1, i) - padh) * ratio_h;  //cy
			float w = out1.at<float>(2, i) * ratio_w;  //w
			float h = out1.at<float>(3, i) * ratio_h;  //h
			int left = MAX((x - 0.5 * w), 0);
			int top = MAX((y - 0.5 * h), 0);
			int width = (int)w;
			int height = (int)h;
			if (width <= 0 || height <= 0) { continue; }
			classIds.push_back(classIdPoint.y);
			confidences.push_back(max_class_socre);
			boxes.push_back(Rect(left, top, width, height));
		}

	}

	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_result);

	std::vector<cv::Mat> temp_mask_proposals;
	std::vector<OutputSeg> output;
	Rect holeImgRect(0, 0, img.cols, img.rows);
	//std::cout << "处理NMS结果。" << std::endl;
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx] & holeImgRect;
		output.push_back(result);
		temp_mask_proposals.push_back(picked_proposals[idx]);
	}

	//std::cout << "开始处理mask。" << std::endl;
	// 处理mask
	Mat maskProposals;
	for (int i = 0; i < temp_mask_proposals.size(); ++i) {
		temp_mask_proposals[i].convertTo(temp_mask_proposals[i], CV_32F);
		maskProposals.push_back(temp_mask_proposals[i]);
	}
	Mat protos = Mat(_segChannels, _segWidth * _segHeight, CV_32F, prob1);
	if (maskProposals.empty()) {
		printf("图像为空");
		return output;
	}
	if (maskProposals.type() != protos.type()) {
		std::cout << "类型不一致" << std::endl;
		std::cout << "maskProposals 类型: " << type2str(maskProposals.type()) << std::endl;
		std::cout << "protos 类型: " << type2str(protos.type()) << std::endl;
		maskProposals.convertTo(maskProposals, CV_32F); // 将maskProposals转换为CV_32F即32位浮点型
		//protos.convertTo(protos, CV_32F); // 将maskProposals转换为CV_32F即32位浮点型
	}

	Mat matmulRes = (maskProposals * protos).t();
	Mat masks = matmulRes.reshape(output.size(), { _segWidth, _segHeight });
	std::vector<Mat> maskChannels;
	cv::split(masks, maskChannels);
	Rect roi(int((float)padw / INPUT_W * _segWidth),
		int((float)padh / INPUT_H * _segHeight),
		_segWidth - int((float)padw / INPUT_W * _segWidth * 2),
		_segHeight - int((float)padh / INPUT_H * _segHeight * 2));
	//std::cout << "处理mask的每一行。" << std::endl;
	for (int i = 0; i < output.size(); ++i) {
		//std::cout << "开始处理第 " << i << " 个输出..." << std::endl;
		Mat dest, mask;
		//  std::cout << "计算sigmoid..." << std::endl;
		cv::exp(-maskChannels[i], dest);//sigmoid
		dest = 1.0 / (1.0 + dest);
		dest = dest(roi);
		resize(dest, mask, cv::Size(img.cols, img.rows), INTER_NEAREST);
		Rect temp_rect = output[i].box;
		mask = mask(temp_rect) > MASK_THRESHOLD;
		output[i].boxMask = mask;
	}
	return output;
}
std::vector<OutputSeg> inferYolo(cv::Mat img, ov::CompiledModel compiled_model, int img_len) {


	auto start0 = std::chrono::system_clock::now();


	int img_width = img.cols;
	int img_height = img.rows;
	static float data[3 * INPUT_H * INPUT_W];
	Mat pr_img0, pr_img;
	std::vector<int> padsize;
	pr_img = preprocess_img(img, INPUT_H, INPUT_W, padsize);       // Resize
	int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
	float ratio_h = (float)img.rows / newh;
	float ratio_w = (float)img.cols / neww;
	float* input_data = (float*)pr_img.data;
	ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
	ov::InferRequest infer_request = compiled_model.create_infer_request();
	infer_request.set_input_tensor(input_tensor);
	infer_request.infer();
	const ov::Tensor& output_tensor = infer_request.get_output_tensor(0);
	const ov::Tensor& output_tensor2 = infer_request.get_output_tensor(1);
	ov::Shape output_shape = output_tensor.get_shape();
	ov::Shape output_shape2 = output_tensor2.get_shape();
	float* prob = output_tensor.data<float>();
	float* prob1 = output_tensor2.data<float>();

	auto start = std::chrono::system_clock::now();
	std::cout << "推理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(start - start0).count() << "ms" << std::endl;

	std::vector<OutputSeg> output = postSegProcessModelAOutput(prob, prob1, ratio_w, ratio_h, padw, padh, img);

	auto end = std::chrono::system_clock::now();
	std::cout << "后处理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


	return output;
}

int main() {
	cv::VideoCapture cap(1); // 0 表示默认摄像头
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 3264);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1832);
	cv::Mat frame;
	cv::Mat rgb = cv::imread("G:\\ncnn-v8\\3.jpg");
	std::string model_path = "G:\\ncnn-v8\\v8\\v8seg.xml";
	std::string model_path1 = "G:\\ncnn-v8\\v8_seg_run39_openvino.onnx";
	ov::CompiledModel yolo_model = initYoloModel(model_path);
	while (true) {
		cap >> frame;
		// 计算裁剪区域的起始点
		int startX = (720 - 300) / 2;
		int startY = (1280 - 300) / 2;

		// 检查是否可以从图像中心裁剪出300x300的区域
		if (startX < 0 || startY < 0) {
			std::cerr << "Error: 图像尺寸小于300x300，无法裁剪" << std::endl;
			return -1;
		}

		// 检查是否可以从图像中心裁剪出300x300的区域
		if (startX < 0 || startY < 0) {
			std::cerr << "Error: 图像尺寸小于300x300，无法裁剪" << std::endl;
			return -1;
		}

		// 设置裁剪区域
		Rect roi(startX, startY, 600, 600);

		// 裁剪图像
		Mat croppedImg = frame(roi);
		if (croppedImg.empty()) {
			break;
		}
		cv::imshow("input", croppedImg);

		cv::cvtColor(croppedImg, croppedImg, cv::COLOR_BGR2GRAY);
		cv::cvtColor(croppedImg, croppedImg, cv::COLOR_GRAY2BGR);

		std::vector<OutputSeg>  output = inferYolo(croppedImg, yolo_model, INPUT_H);
		DrawPred(croppedImg, output);
		cv::imshow("result", croppedImg);
		cv::waitKey(1);
	}

}