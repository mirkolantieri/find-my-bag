import os
import sys
import time
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchextractor as tx #(https://github.com/antoinebrl/torchextractor)
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

root = os.path.dirname(os.path.realpath(__file__))
WEIGHT_DIR = f"{root}/neural_weight/"


class CNN():
    def __init__(self, labels=None, arch='MOBNET', ft=True):

        # Check gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if arch=='MOBNET':
            if ft:
                # Load trained model info
                best_model = torch.load(os.path.join(WEIGHT_DIR, arch, 'model_best.pth.tar'),
                    map_location=self.device)
                self.mapping = best_model['class_mapping']

                # Images labels
                if labels:
                    self.X = labels
                    self.targets = pd.Series(list(map(self.mapping.get, self.X['class'])))
                    self.classes = list(self.mapping.values())

                model = models.mobilenet_v3_large()
                model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features,
                    len(best_model['class_mapping'].values()))

                model.load_state_dict(best_model['state_dict'])
                self.layers_feat = ['classifier.0']

            else:
                model = models.mobilenet_v3_large(pretrained=True)
                self.layers_feat = ['classifier.0']

        elif arch=='VGG16':
            model = models.vgg16(pretrained=True)
            self.layers_feat = ['classifier.3']
        
        else:
            print('Arch not supported.')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        
        model.eval()

        # Utility wrapper for extract features from specified layer(s)
        self.model = tx.Extractor(model, self.layers_feat)

        print("Pre-trained {} neural model loaded.".format(arch))

        # Transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def _extract(self, image):
        with torch.no_grad():
            # Apply image transformation
            input = self.transform(image).float()

            # CPU or GPU calculation
            input.to(self.device)
            input = input.unsqueeze(0)

            # Get model outputs and features 
            output, features = self.model(input)

            feature_shapes = {name: f.shape for name, f in features.items()}
            #print(feature_shapes)

            feat_array = features[self.layers_feat[0]].data.cpu().numpy().flatten()
        return feat_array


    def _extractBatch(self, csv, db_folder):
        class AverageMeter(object):
            """Computes and stores the average and current value"""
            def __init__(self, name, fmt=':f'):
                self.name = name
                self.fmt = fmt
                self.reset()

            def reset(self):
                self.val = 0
                self.avg = 0
                self.sum = 0
                self.count = 0

            def update(self, val, n=1):
                self.val = val
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count

            def __str__(self):
                fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
                return fmtstr.format(**self.__dict__)

        class ProgressMeter(object):
            def __init__(self, num_batches, meters, prefix=''):
                self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
                self.meters = meters
                self.prefix = prefix

            def display(self, batch):
                entries = [self.prefix + self.batch_fmtstr.format(batch)]
                entries += [str(meter) for meter in self.meters]
                print('\t'.join(entries))

            def save(self, batch):
                entries = [self.prefix + self.batch_fmtstr.format(batch)]
                entries += [str(meter) for meter in self.meters]
                return str('\t'.join(entries)+'\n')

            def _get_batch_fmtstr(self, num_batches):
                num_digits = len(str(num_batches // 1))
                fmt = '{:' + str(num_digits) + 'd}'
                return '[' + fmt + '/' + fmt.format(num_batches) + ']'

        class CustomDataset(Dataset):
            def __init__(self, labels, root_dir, transform=None):
                self.X = labels
                self.root_dir = root_dir
                self.transform = transform
                
            def __len__(self):
                return (len(self.X))
            
            def __getitem__(self, idx):
                if torch.is_tensor(idx):
                    idx = idx.tolist()
                img_name = os.path.join(self.root_dir,
                                        self.X.iloc[idx]['name'])
                image = Image.open(img_name).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image


        data = CustomDataset(labels=csv, root_dir=os.path.realpath(db_folder), transform=self.transform)
        dataload = DataLoader(data, batch_size=64, shuffle=False, num_workers=0)

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        progress = ProgressMeter(
            len(dataload),
            [batch_time, data_time])

        feat_array = []

        with torch.no_grad():
            end = time.time()
            for i, (images) in enumerate(dataload):
                data_time.update(time.time() - end)

                input = images.to(self.device)

                # Get model outputs and features 
                _, features = self.model(input)
                feat_array.append(features[self.layers_feat[0]].data.cpu().numpy())

                batch_time.update(time.time() - end)
                end = time.time()
                progress.display(i)

        return np.vstack(feat_array)