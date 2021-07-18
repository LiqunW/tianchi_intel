import tensorflow as tf
import torch
import paddle


tf.test.is_gpu_available()
print(torch.cuda.is_available())