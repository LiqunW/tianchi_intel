import argparse
import pickle


def load_characters(path=r'data_process/chars.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        char_str = f.read().splitlines()
    char_str = ''.join(char_str)
    return char_str

# todo 模型训练配置文件

parser = argparse.ArgumentParser()

"""
数据处理配置，目前支持读取lmdb格式数据，支持多数据源按比例配置
"""
# todo 支持在线训练，支持调用数据生成服务

parser.add_argument('--train_data', type=str, default=r'C:/Work/1_work/tianchi_intel/dataset/huawei/lmdb',
                    help='path to training dataset')
parser.add_argument('--valid_data', type=str,default=r'C:/Work/1_work/tianchi_intel/dataset/huawei/lmdb/special',
                    help='path to validation dataset')

# 多个真实数据比例，使用单个数据集，--select_data='/', --batch_ratio='1'
parser.add_argument('--select_data', type=str, default='common-special',
                    help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
parser.add_argument('--batch_ratio', type=str, default='0.8-0.2',
                    help='assign ratio for each selected data in the batch')
parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                    help='total data usage ratio, this ratio is multiplied to total number of data.')

parser.add_argument('--use_gen_service', type=bool, default=False,
                    help='generate data by service')
parser.add_argument('--gen_url', type=str, default=u'')
parser.add_argument('--gen_data_ratio', type=str, default='0.5-0.5',
                    help='assign ratio for generate data and selected data in the batch')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int, default=200, help='input batch size')
parser.add_argument('--batch_max_length', type=int, default=50, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=400, help='the width of the input image')
parser.add_argument('--rgb', action='store_false', help='use rgb input')
parser.add_argument('--PAD', action='store_false', help='whether to keep ratio then pad for image resize')
parser.add_argument('--character', type=str,
                    # default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
                    default=load_characters(), help='character label')
# 不过滤数据
parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')


# 数据增强
aug_parser = parser.add_subparsers(help='data augmentation config')
aug_config = aug_parser.add_parser('use_data_aug', help='data augmentation config')
aug_config.add_argument('--mix_aug', action='store_false', help='use multiple augmentation methods')
aug_config.add_argument('--brightened', action='store_false', help='change image brightness')
aug_config.add_argument('--colored', action='store_false', help='change image color')
aug_config.add_argument('--contrasted', action='store_false', help='change image contrast')
aug_config.add_argument('--sharped', action='store_false', help='change image sharp')
aug_config.add_argument('--random_hsv_transform', action='store_false', help='change image color in hsv')
aug_config.add_argument('--warp_perspective_image', action='store_false', help='affine transformation')
aug_config.add_argument('--smooth', action='store_false', help='image smooth')
aug_config.add_argument('--speckle', action='store_false', help='add speckle noise')
aug_config.add_argument('--gaussian_noise', action='store_false', help='add gaussian_noise')
aug_config.add_argument('--multiple_gaussian_noise', action='store_false', help='add multiple gaussian noise')
aug_config.add_argument('--pepperand_salt_noise', action='store_false', help='add pepperand salt noise')
aug_config.add_argument('--poisson_noise', action='store_false', help='add poisson noise')
aug_config.add_argument('--lognormal_noise', action='store_false', help='add lognormal noise')
aug_config.add_argument('--gamma_noise', action='store_false', help='add gamma noise')
aug_config.add_argument('--multiple_gamma_noise', action='store_false', help='add multiple gamma noise')
aug_config.add_argument('--gaussian_blur', action='store_false', help='add gaussian blur')
aug_config.add_argument('--down_up_sample', action='store_false', help='add down up sample')

"""
模型架构配置，分为4个阶段，图像矫正(非必须) -> 特征提取 -> lstm(非必须) -> 解码(ctc|attn)
tps控制点，lstm核数量设置
"""
parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default='ResNet',
                    help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=3,
                    help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

"""
模型训练配置，训练次数，验证频率，学习率，优化器设置等
"""
parser.add_argument('--exp_name', default=r'huawei_0731_ft_ver', help='where to store logs and models')
parser.add_argument('--config', default='train_config_2.pkl', help='save train config for next train')
parser.add_argument('--manualSeed', type=int, default=1024, help='for random seed setting')
parser.add_argument('--num_iter', type=int, default=500000, help='number of iterations to train for')
parser.add_argument('--valInterval', type=int, default=5000, help='Interval between each validation')
parser.add_argument('--saved_model', default='pre_trained_models/TPS-ResNet-BiLSTM-CTC.pth',
                    help="path to model to continue training")
parser.add_argument('--FT', default=True, help='whether to do fine-tuning')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')

opt = parser.parse_args()

"""
将训练配置保存，方便下次使用
"""
with open(opt.config, 'wb') as f:
    pickle.dump(opt, f)
