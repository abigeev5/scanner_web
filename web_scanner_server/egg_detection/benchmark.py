from inference import inference
import time
import os


base_image_folder = "Test_images/"
test_folder = "Test"


def create_folder_for_n(n):
	os.remove(test_folder)
	images = [i for i in os.listdir()]
	for i in range(n):
		os.copy(images[i % len(images)], f"{test_folder}/{i}.bmp")


def main():
	table_size = 121
	print('-' * table_size)
    print(f"|{'Samples':^29}|{n1:^29.5g}|{n2:^29.5g}|{n3:^29.5g}|")
    print('-' * table_size)
    print(f"|{'':^29}|{'CPU avg':^10}|{'GPU avg':^18}|{'CPU med':^10}|{'GPU med':^18}|{'CPU worst':^10}|{'GPU worst':^18}|")
    print('-' * table_size)
	for num_images in [1, 10, 50, 100]:
		start_time = time.time()
        create_folder_for_n(num_images)
		inference(test_folder, device="cpu:0")
		cpu_time = time.time() - start_time

        start_time = time.time()
        inference(test_folder, device="gpu:0")
        gpu_time = time.time() - start_time

        print(f"|{num_images:^29}|{cpu_time:^10.5g}|{gpu_time:^18.5g}|{cpu_time:^10.5g}|{gpu_time:^18.5g}|{cpu_time:^10.5g}|{gpu_time:^18.5g}|")
        print('-' * table_size)
    print('-' * table_size)


if __name__ == "__main__":
	main()