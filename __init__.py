from __future__ import annotations
from ._core import __doc__, __version__, add, subtract, sine_wave, cosine_wave, square_wave, sawtooth_wave, read_wav, write_wav, apply_gaussian_filter, create_gaussian_kernel, apply_convolution, apply_gaussian_filter_to_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def process_and_plot_audio_with_gaussian_filter(input_filename, output_filename, sigma):
    
    audio_data = read_wav(input_filename)
   
    filtered_audio_data = apply_gaussian_filter(audio_data, sigma)
  
    write_wav(filtered_audio_data, output_filename)
    

    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(audio_data[:500])  
    plt.title('Fragment pliku audio przed filtracja')
    plt.xlabel('Numer probki')
    plt.ylabel('Amplituda (probki)')
    
    plt.subplot(2, 1, 2)
    plt.plot(filtered_audio_data[:500])  
    plt.title('Fragment pliku audio po przetworzeniu filtrem Gaussa')
    plt.xlabel('Numer probki')
    plt.ylabel('Amplituda (probki)')
    
    plt.tight_layout()
    plt.show()


def process_and_plot_image_with_gaussian_filter(input_filename, output_filename, sigma):
        input_image = np.array(Image.open(input_filename))
        kernel_size = max(3, min(15, int(round(sigma * 5))))  
        kernel = create_gaussian_kernel(kernel_size, sigma)
        print("Kernel size:", kernel_size)
        filtered_image = apply_gaussian_filter_to_image(input_image, sigma, kernel)  
        print("3 okkk")
        Image.fromarray(filtered_image.astype(np.uint8)).save(output_filename)
        print(" 4 ok");
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(input_image)
        plt.title("Oryginalny obraz")

        plt.subplot(1, 2, 2)
        plt.imshow(filtered_image)
        plt.title("Obraz po przetworzeniu filtrem Gaussa")

        plt.tight_layout()
        plt.show()
 




def main_menu():
    while True:
        print("1. Generate and plot sine wave")
        print("2. Generate and plot cosine wave")
        print("3. Generate and plot square wave")
        print("4. Generate and plot sawtooth wave")
        print("5. Process and plot audio with Gaussian filter")
        print("6. Process and plot image with Gaussian filter")
        print("q. Quit")
        choice = input(" (1-6, or q ): ")
        if choice == '1':
            frequency = float(input("Enter the frequency of the sine wave: "))
            num_points = int(input("Enter the number of points in the wave: "))
            sine_wave(frequency,num_points)
        elif choice == '2':
            frequency = float(input("Enter the frequency of the cosine wave: "))
            num_points = int(input("Enter the number of points in the wave: "))
            cosine_wave(frequency,num_points)
        elif choice == '3':
            frequency = float(input("Enter the frequency of the square wave: "))
            num_points = int(input("Enter the number of points in the wave: "))
            square_wave(frequency, num_points)
        elif choice == '4':
            frequency = float(input("Enter the frequency of the sawtooth wave: "))
            num_points = int(input("Enter the number of points in the wave: "))
            sawtooth_wave(frequency,num_points)
        elif choice == '5':
            input_filename = input("Enter the input audio filename: ")
            output_filename = input("Enter the output audio filename: ")
            sigma = float(input("Enter the value of sigma for the Gaussian filter: "))
            process_and_plot_audio_with_gaussian_filter(input_filename, output_filename, sigma)
        elif choice == '6':
            input_filename = input("Enter the input image filename: ")
            output_filename = input("Enter the output image filename: ")
            sigma = float(input("Enter the value of sigma for the Gaussian filter: "))
            process_and_plot_image_with_gaussian_filter(input_filename, output_filename, sigma)
        elif choice.lower() == 'q':
            print("Exit.")
            break
        else:
            print(" 1-6, or q.")

__all__ = ["__doc__", "__version__", "add", "subtract", "sine_wave", "cosine_wave", "square_wave", "sawtooth_wave", "read_wav", "write_wav", "apply_gaussian_filter", "create_gaussian_kernel", "apply_convolution", "apply_gaussian_filter_to_image", "main_menu"]

if __name__ == "__main__":
    main_menu()