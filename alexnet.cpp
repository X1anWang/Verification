#include <torch/library.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <string>
#include <io.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <ctime>

namespace F = torch::nn::functional;

// AlexNet

struct alexnet : public torch::nn::Module
{
    torch::nn::Conv2d C1;
    torch::nn::Conv2d C3;
    torch::nn::Conv2d C6;
    torch::nn::Conv2d C8;
    torch::nn::Conv2d C10;
    torch::nn::Linear FC1;
    torch::nn::Linear FC2;
    torch::nn::Linear FC3;

    alexnet():
        C1(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 11).padding(2))),
        C3(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).padding(2))),
        C6(torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1))),
        C8(torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1))),
        C10(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1))),
        FC1(torch::nn::Linear(9216, 4096)),
        FC2(torch::nn::Linear(4096, 4096)),
        FC3(torch::nn::Linear(4096, 1000))
        {
            register_module("C1", C1);
            register_module("C3", C3);
            register_module("C6", C6);
            register_module("C8", C8);
            register_module("C10", C10);
            register_module("FC1", FC1);
            register_module("FC2", FC2);
            register_module("FC3", FC3);
        }
    
    torch::Tensor forward(torch::Tensor input)
    {
        auto x = F::max_pool2d(F::relu(C1(input)), F::MaxPool2dFuncOptions(3));
        x = F::max_pool2d(F::relu(C3(x)), F::MaxPool2dFuncOptions(3));
        x = F::max_pool2d(F::relu(C10(F::relu(C8(F::relu(C6(x)))))), F::MaxPool2dFuncOptions(3));
        x = x.view({ -1, num_flat_features(x) });
        x = F::dropout(x, F::DropoutFuncOptions().p(0.5));
        x = F::dropout(F::relu(FC1(x), F::DropoutFuncOptions().p(0.5));
        x = FC3(F::relu(FC2(x)));

        return x;
    }

    long num_flat_features(torch::Tensor x)
    {
        // To except the batch dimension:
        // auto size = x.size()[1:]
        // For AlexNet:
        auto size = x.sizes();
        auto num_features = 1;
        for (auto s : size)
        {
            num_features *= s;
        }
        return num_features;
    }
}

int main()
{
    std::cout << "AlexNet - CPU version" << std::endl;

    alexnet model;

    time_t start = time(0);

    for (int i = 0; i < 50; i++)
    {
        auto input = torch::ones({1, 3, 224, 224});
        torch::Tensor output = model.forward(input);
    }

    time_t stop = time(0);
    double duration;
    duration = difftime(stop, start);

    std::cout << "Completed." << std::endl;

    return 0;
}