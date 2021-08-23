import os

import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import cv2
from .utils import CTCLabelConverter, AttnLabelConverter
from .model import Model
from .dataset import RawDataset, AlignCollate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def load_characters(path=r'data_process/chars.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        char_str = f.read().splitlines()
    char_str = ''.join(char_str)
    return char_str


class Text_Recognition:
    def __init__(self, mp):
        self.workers = 0
        self.batch_size = 1
        self.batch_max_length = 50
        self.saved_model = mp
        self.imgH = 32
        self.imgW = 400
        self.rgb = 'store_true'
        self.character = load_characters("CTC/chars.txt")
        self.sensitive = False
        self.PAD = 'store_false'
        self.Transformation = 'None'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'None'
        self.Prediction = 'CTC'
        self.image_folder = "ctc_img_storage"
        self.num_fiducial = 20
        self.input_channel = 3
        self.output_channel = 512
        self.hidden_size = 256
        cudnn.benchmark = True
        cudnn.deterministic = True
        self.num_gpu = torch.cuda.device_count()
        self.load_model()

    def load_model(self):
        """ model configuration """
        if 'CTC' in self.Prediction:
            self.converter = CTCLabelConverter(self.character)
        else:
            self.converter = AttnLabelConverter(self.character)
        self.num_class = len(self.converter.character)
        if self.rgb:
            self.input_channel = 3
            self.model = Model(self)
        print('model input parameters', self.imgH, self.imgW, self.num_fiducial, self.input_channel, self.output_channel,
              self.hidden_size, self.num_class, self.batch_max_length, self.Transformation, self.FeatureExtraction,
              self.SequenceModeling, self.Prediction)
        self.model = torch.nn.DataParallel(self.model).to(device)

        # load model
        print('loading pretrained model from %s' % self.saved_model)
        self.model.load_state_dict(torch.load(self.saved_model, map_location=device), False)

    def demo(self, image):
        os.makedirs(self.image_folder, exist_ok=True)
        cv2.imwrite(os.path.join(self.image_folder, "ctc.jpg"), image)
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=self.imgH, imgW=self.imgW, keep_ratio_with_pad=self.PAD)
        demo_data = RawDataset(root=self.image_folder, opt=self)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.batch_size,
            shuffle=False,
            num_workers=int(self.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)

        # predict
        self.model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, self.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in self.Prediction:
                    preds = self.model(image, text_for_pred)
                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode(preds_index, preds_size)
                else:
                    preds = self.model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)


                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in self.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

        return pred





if __name__ == '__main__':
    ModelRecognition = Text_Recognition(r"../model/best_norm_ED.pth")
