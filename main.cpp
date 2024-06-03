#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <matplot/matplot.h>
#include<pybind11/numpy.h>
#include<algorithm>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

int add(int i, int j) 
{
    return i + j;
}

void sine_wave(double frequency, int num_points) 
{
    const double amplitude = 1.0; 
    std::vector<double> result(num_points);
    for (int i = 0; i < num_points; ++i) 
    {
        result[i] = amplitude * sin(2 * M_PI * frequency * i / num_points);
    }
    matplot::plot(result);
    matplot::show();
}

void cosine_wave(double frequency, int num_points)
{
    const double amplitude = 1.0; 
    std::vector<double> result(num_points);
    for (int i = 0; i < num_points; ++i) 
    {
        result[i] = amplitude * cos(2 * M_PI * frequency * i / num_points);
    }
    matplot::plot(result);
    matplot::show();
}

void square_wave(double frequency, int num_points)
{
    const double amplitude = 1.0; 
    std::vector<double> result(num_points);
    for (int i = 0; i < num_points; ++i) 
    {
        result[i] = amplitude * (sin(2 * M_PI * frequency * i / num_points) >= 0 ? 1 : -1);
    }
    matplot::plot(result);
    matplot::show();
}

 

void sawtooth_wave(double frequency, int num_points) 
{
    std::vector<double> wave(num_points);

    for (int i = 0; i < num_points; ++i) 
    {
        double t = i / static_cast<double>(num_points - 1); 
        wave[i] = 2.0 * (frequency * t - std::floor(frequency * t + 0.5));
    }

    
    double max_value = *std::max_element(wave.begin(), wave.end());
    double min_value = *std::min_element(wave.begin(), wave.end());
    double range = std::max(std::abs(max_value), std::abs(min_value));
    for (int i = 0; i < num_points; ++i) {
        wave[i] /= range;
    }


    matplot::plot(wave);

    matplot::show();

    
}

py::array_t<double> order_filter_vector(py::array_t<double> input, int size, int rank)
{
    py::buffer_info buf = input.request();
    if (buf.ndim != 1) 
    {
        throw std::runtime_error("Input should be a 1-D numpy array");
    }

    double* ptr = static_cast<double*>(buf.ptr);
    std::ptrdiff_t length = buf.shape[0];

    std::vector<double> result(length);
    std::vector<double> window(size);

    for (std::ptrdiff_t i = 0; i < length; ++i) 
    {
        std::ptrdiff_t half_size = size / 2;
        for (std::ptrdiff_t j = 0; j < size; ++j) 
        {
            std::ptrdiff_t index = i + j - half_size;
            if (index < 0 || index >= length) 
            {
                window[j] = 0;
            }
            else {
                window[j] = ptr[index];
            }
        }
        std::sort(window.begin(), window.end());
        result[i] = window[rank];
    }

    py::array_t<double> result_array(result.size());
    std::memcpy(result_array.mutable_data(), result.data(), result.size() * sizeof(double));
    return result_array;
}



std::vector<int16_t> read_wav(const std::string& filename)
{
    std::vector<int16_t> audio_data;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        throw std::runtime_error("couldnt open file");
    }

    
    file.seekg(44);

    while (file) {
        int16_t sample;
        file.read(reinterpret_cast<char*>(&sample), sizeof(sample));
        if (file) {
            audio_data.push_back(sample);
        }
    }

    return audio_data;
}

void write_wav(const std::vector<int16_t>& audio_data, const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);

    if (!file) 
    {
        throw std::runtime_error("couldnt open file");
    }

    
    file.write("RIFF", 4);
    int32_t file_size_minus_8 = 36 + audio_data.size() * sizeof(int16_t);
    file.write(reinterpret_cast<char*>(&file_size_minus_8), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    int32_t subchunk1_size = 16;
    file.write(reinterpret_cast<char*>(&subchunk1_size), 4);
    int16_t audio_format = 1;
    file.write(reinterpret_cast<char*>(&audio_format), 2);
    int16_t num_channels = 1;
    file.write(reinterpret_cast<char*>(&num_channels), 2);
    int32_t sample_rate = 44100;
    file.write(reinterpret_cast<char*>(&sample_rate), 4);
    int32_t byte_rate = sample_rate * num_channels * sizeof(int16_t);
    file.write(reinterpret_cast<char*>(&byte_rate), 4);
    int16_t block_align = num_channels * sizeof(int16_t);
    file.write(reinterpret_cast<char*>(&block_align), 2);
    int16_t bits_per_sample = 16;
    file.write(reinterpret_cast<char*>(&bits_per_sample), 2);
    file.write("data", 4);
    int32_t subchunk2_size = audio_data.size() * sizeof(int16_t);
    file.write(reinterpret_cast<char*>(&subchunk2_size), 4);

    for (int16_t sample : audio_data) 
    {
        file.write(reinterpret_cast<char*>(&sample), sizeof(sample));
    }
}

std::vector<int16_t> apply_gaussian_filter(const std::vector<int16_t>& input, double sigma) 
{
    int kernel_radius = std::ceil(3 * sigma);
    int kernel_size = 2 * kernel_radius + 1;
    std::vector<double> kernel(kernel_size);

    double sum = 0.0;
    for (int i = 0; i < kernel_size; ++i) 
    {
        int x = i - kernel_radius;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma)) / (std::sqrt(2 * M_PI) * sigma);
        sum += kernel[i];
    }

    for (double& value : kernel) 
    {
        value /= sum;
    }

    std::vector<int16_t> result(input.size(), 0);

    for (std::size_t i = 0; i < input.size(); ++i) 
    {
        double filtered_value = 0.0;
        for (int j = -kernel_radius; j <= kernel_radius; ++j) 
        {
            if (i + j >= 0 && i + j < input.size()) {
                filtered_value += input[i + j] * kernel[j + kernel_radius];
            }
        }
        result[i] = static_cast<int16_t>(std::round(filtered_value));
    }

    return result;
}
std::vector<std::vector<double>> create_gaussian_kernel(int kernel_size, double sigma)
{
    

    if (kernel_size % 2 == 0)
    {
        kernel_size += 1;
    }

    std::vector<std::vector<double>> kernel(kernel_size, std::vector<double>(kernel_size));
    double sum = 0.0;
    int half_size = kernel_size / 2;
    double sigma_squared = sigma * sigma;

    for (int i = -half_size; i <= half_size; ++i) 
    {
        for (int j = -half_size; j <= half_size; ++j) 
        {
            kernel[i + half_size][j + half_size] = std::exp(-(i * i + j * j) / (2 * sigma_squared));
            sum += kernel[i + half_size][j + half_size];
        }
    }

    for (int i = 0; i < kernel_size; ++i) 
    {
        for (int j = 0; j < kernel_size; ++j) 
        {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}


std::vector<std::vector<double>> apply_convolution(const std::vector<std::vector<double>>& channel, const std::vector<std::vector<double>>& kernel)
{
    int rows = channel.size();
    int cols = channel[0].size();
    int kernel_size = kernel.size();
    int half_size = kernel_size / 2;
    std::vector<std::vector<double>> filtered_channel(rows, std::vector<double>(cols, 0.0));

    for (int i = half_size; i < rows - half_size; ++i)
    {
        for (int j = half_size; j < cols - half_size; ++j) 
        {
            double sum = 0.0;
            for (int k = -half_size; k <= half_size; ++k) 
            {
                for (int l = -half_size; l <= half_size; ++l) 
                {
                    sum += channel[i + k][j + l] * kernel[k + half_size][l + half_size];
                }
            }
            filtered_channel[i][j] = sum;
        }
    }
    return filtered_channel;
}

py::array_t<uint8_t> apply_gaussian_filter_to_image(py::array_t<uint8_t> input_image, double sigma, const std::vector<std::vector<double>>& kernel)
{
    py::buffer_info buf = input_image.request();
    if (buf.ndim != 3 || buf.shape[2] != 3) 
    {
        throw std::runtime_error("error");
    }

    int rows = buf.shape[0];
    int cols = buf.shape[1];
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

    std::vector<std::vector<double>> r_channel(rows, std::vector<double>(cols));
    std::vector<std::vector<double>> g_channel(rows, std::vector<double>(cols));
    std::vector<std::vector<double>> b_channel(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            r_channel[i][j] = ptr[i * cols * 3 + j * 3];
            g_channel[i][j] = ptr[i * cols * 3 + j * 3 + 1];
            b_channel[i][j] = ptr[i * cols * 3 + j * 3 + 2];
        }
    }

    auto r_filtered = apply_convolution(r_channel, kernel);
    auto g_filtered = apply_convolution(g_channel, kernel);
    auto b_filtered = apply_convolution(b_channel, kernel);

    py::array_t<uint8_t> output_image({ rows, cols, 3 });
    uint8_t* out_ptr = static_cast<uint8_t*>(output_image.request().ptr);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out_ptr[i * cols * 3 + j * 3] = static_cast<uint8_t>(std::clamp(r_filtered[i][j], 0.0, 255.0));
            out_ptr[i * cols * 3 + j * 3 + 1] = static_cast<uint8_t>(std::clamp(g_filtered[i][j], 0.0, 255.0));
            out_ptr[i * cols * 3 + j * 3 + 2] = static_cast<uint8_t>(std::clamp(b_filtered[i][j], 0.0, 255.0));
        }
    }

    return output_image;
}


PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
           sine_wave
           cosine_wave
           square_wave
           sawtooth_wave
           order_filter_vector
           read_wav
           write_wav
           apply_gaussian_filter
           create_gaussian_kernel
           apply_convolution
           apply_gaussian_filter_to_image

    )pbdoc";


    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("sine_wave", &sine_wave, R"pbdoc(
        Generate and plot a sine wave

        Parameters
        ----------
        frequency : double
            Frequency of the sine wave
        num_points : int
            Number of points in the wave
    )pbdoc");

    m.def("cosine_wave", &cosine_wave, R"pbdoc(
        Generate and plot a cosine wave

        Parameters
        ----------
        frequency : double
            Frequency of the cosine wave
        num_points : int
            Number of points in the wave
    )pbdoc");

    m.def("square_wave", &square_wave, R"pbdoc(
        Generate and plot a square wave

        Parameters
        ----------
        frequency : double
            Frequency of the square wave
        num_points : int
            Number of points in the wave
    )pbdoc");

    m.def("sawtooth_wave", &sawtooth_wave, R"pbdoc(
        Generate and plot a sawtooth wave

        Parameters
        ----------
        frequency : double
            Frequency of the sawtooth wave
        num_points : int
            Number of points in the wave
    )pbdoc");

    m.def("order_filter_vector", &order_filter_vector, R"pbdoc(
        Apply an order filter to a vector

        Parameters
        ----------
        input : numpy array
            Input vector to filter
        size : int
            Size of the filter window
        rank : int
            Rank of the value to select from the sorted window

        Returns
        -------
        numpy array
            Filtered vector
    )pbdoc");


    m.def("add", [](int i, int j) { return i + j; }, R"pbdoc(Add two numbers)pbdoc");
    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(Subtract two numbers)pbdoc");
    m.def("read_wav", &read_wav, R"pbdoc(Read a WAV file and return the audio data as a list)pbdoc");
    m.def("write_wav", &write_wav, R"pbdoc(Write audio data to a WAV file)pbdoc");
    m.def("apply_gaussian_filter", &apply_gaussian_filter, R"pbdoc(Apply a Gaussian filter to the audio data)pbdoc");

    m.def("apply_convolution", &apply_convolution, R"pbdoc(
        Apply convolution to input data with a given kernel

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data as a NumPy array
        kernel : numpy.ndarray
            Convolution kernel as a NumPy array
        Returns
        -------
        numpy.ndarray
            Convolved data as a NumPy array
    )pbdoc");

    m.def("apply_gaussian_filter_to_image", &apply_gaussian_filter_to_image, R"pbdoc(
        Apply Gaussian filter to image
    Parameters
    ----------
    input_image : numpy.ndarray
        Input image as a NumPy array
    sigma : float
        Standard deviation for Gaussian filter
    
   
        Returns
        -------
        numpy.ndarray
            Filtered image as a NumPy array
    )pbdoc");

    m.def("create_gaussian_kernel", &create_gaussian_kernel, R"pbdoc(
        Create a Gaussian kernel

        Parameters
        ----------
        kernel_size : int
            Size of the kernel (should be an odd number)
        sigma : float
            Standard deviation for Gaussian filter
        Returns
        -------
        numpy.ndarray
            Gaussian kernel as a NumPy array
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}