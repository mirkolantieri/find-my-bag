# Class for feature extraction methods
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .utils import load_image, pil_to_opencv
from .Descriptors.neuralNet import CNN
from .Descriptors.traditionalFeatures import colorHSV, texture, huMoments

root = os.path.dirname(os.path.realpath(__file__))
DB_PATH = f"{root}/DB/index"


class FeatExtractor(object):
    """ Utility class for feature extraction from both neural network and handcrafted methods """
    def __init__(self, db_folder=None, csv_file=None):
        arch = 'MOBNET'
        ft = True
        self.neuralModel = CNN(arch=arch, ft=ft)
        self.savepath = os.path.join(DB_PATH, 'neural_{}{}_feats.npy'.format(arch, 'ft' if ft else ''))
        self.csv_file = csv_file
        self.db_folder = db_folder

        self.scaler = None
        self.pca = None

    
    def get_neuralFeat_batch(self):
        """ Extract CNN features in batch """
        if not os.path.isfile(self.savepath):
            print('Neural features not already computed...')
            neuralFeatures = self.neuralModel._extractBatch(csv=self.csv_file, db_folder=self.db_folder)
            np.save(file=self.savepath, arr=neuralFeatures)
        else:
            print('Loading neural feats')
            neuralFeatures = np.load(self.savepath)
        return neuralFeatures


    def get_neuralFeatures(self, image):
        """ Extract CNN features for a single image """
        features = []
        if isinstance(image, str) and os.path.isfile(image):
            # provided path of image
            img = load_image(image)
        else:
            img = image

        features = self.neuralModel._extract(img)
        return features


    def get_handFeatures(self, image, mask):
        """ Extract handcrafted features for a single image """

        features = []
        if isinstance(image, str) and os.path.isfile(image):
            # provided path of image
            img = load_image(image)
        else:
            img = image
        
        img_cv2 = pil_to_opencv(img)
        color_features = colorHSV(img_cv2, mask)
        texture_features = texture(img_cv2, mask)
        hu_features = huMoments(mask)

        feats = [color_features, texture_features, hu_features]
        features = np.hstack(feats)

        return features


    def fuse_features(self, feats, pca=True):

        # Raw feature concatenation
        combined = np.hstack(feats)
        fused = combined

        if pca:
            if fused.ndim == 1:
                # Inference
                fused = self.scaler.transform(combined.reshape(1,-1))
                fused_sel = self.pca.transform(fused.reshape(1,-1))
            else:
                # Fitting
                # Feature normalization/standardization
                self.scaler = StandardScaler().fit(combined)
                fused = self.scaler.transform(combined)
            
                # Feature selection with PCA
                self.pca = PCA(.95)
                #self.pca = PCA(n_components=512)
                self.pca.fit(fused)

                #print(self.pca.explained_variance_)
                #print(self.pca.explained_variance_ratio_)
                #print(self.pca.explained_variance_ratio_.cumsum())   

                fused_sel = self.pca.transform(fused)

            return fused_sel, self.scaler, self.pca

        return fused