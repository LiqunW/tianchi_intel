import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import string
import pickle
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F


try:
    from .utils import CTCLabelConverter, AttnLabelConverter
    from .dataset import RawDataset, AlignCollate # RawDataset_no_short, AlignCollate_no_aug, RawDataset, AlignCollate
    from .model import Model
except:
    from utils import CTCLabelConverter, AttnLabelConverter
    from dataset import RawDataset, AlignCollate  # RawDataset_no_short, AlignCollate_no_aug, RawDataset, AlignCollate
    from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)
                preds = F.softmax(preds, dim=2)
                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                pred_p, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str, preds_prob = converter.decode(preds_index, pred_p, preds_size)

            log = open(f'./Result.txt', 'w', encoding='utf-8')

            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_prob):
                confidence_score = np.mean(pred_max_prob)

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()


def load_lexion(path=r'words_dict.pkl'):
    with open(path, 'rb') as f:
        words_dict = pickle.load(f)


def rectify_res(pred_str, pred_prob, threshold=0.85):
    # TODO 过滤开头和结尾的空格
    for p_str, p_prob in zip(pred_str, pred_prob):
        # 按照空格分隔，并且找到对应的index
        if ' ' not in p_str:
            return pred_str, pred_prob
        else:
            w_list = p_str.split(' ')
            blank_idx = []
            for i in range(len(p_str)):
                if p_str[i] == ' ':
                    blank_idx.append(i)
            words_prob = []
            for i in range(len(blank_idx)):
                if i == 0:
                    words_prob.append(p_prob[:blank_idx[0]])
                else:
                    words_prob.append(p_prob[blank_idx[i-1]+1:blank_idx[i]])
                

def load_characters(path=r'data_process/chars.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        char_str = f.read().splitlines()
    char_str = ''.join(char_str)
    return char_str


def run_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default=r'demo_images',
                        help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--saved_model', help="path to saved_model to evaluation",
                        default=r'models/best_norm_ED.pth')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=50, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=400, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', default=True, help='use rgb input')
    # parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--character', type=str,
                        default=load_characters("chars.txt"), help='character label') # 载入中文字符集
    parser.add_argument('--sensitive', default=False, action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_false', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=False, default="None", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=False, default="ResNet",  help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=False,  default="None", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=False, default="CTC", help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    print(opt)

    """ vocab / character number configuration """
    # if opt.sensitive:
    #     opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)


if __name__ == '__main__':
    run_demo()
