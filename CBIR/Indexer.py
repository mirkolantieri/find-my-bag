import os
import sys
import pandas as pd
import numpy as np
import pickle 
from timeit import default_timer as timer
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from .utils import load_image
from .Extractor import FeatExtractor
from .Processing import u2netSegmentation
from .Processing.groupClassifier import groupClassifier

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Avoid truncated images interruptions

root = os.path.dirname(os.path.realpath(__file__))

DB_FOLDER = f"{root}/DB/images"             # DB image directory
DB_FOLDER_MASK = f"{root}/DB/images_mask"   # DB image directory mask
DB_CSV = f"{root}/DB/train.csv"             # DB image csv

DB_INDEX = f"{root}/DB/index/index.pkl"           # DB index as pickle format
DB_KNN_INDEX = f"{root}/DB/index/knn_index.pkl"
DB_SCALER = f"{root}/DB/index/scaler.pkl"
DB_SCALER_PCA = f"{root}/DB/index/pca_scaler.pkl"
DB_PCA = f"{root}/DB/index/pca.pkl"

TOPK = 5


class Index(object):
    def __init__(self, extractor=None, reset=True):

        self.db_path = DB_FOLDER
        self.db_csv_path = DB_CSV
        self.db_csv = pd.read_csv(self.db_csv_path)
        self.index_filename = DB_INDEX
        self.knn_index_filename = DB_KNN_INDEX
        self.scaler_filename = DB_SCALER
        self.pca_scaler_filename = DB_SCALER_PCA
        self.pca_filename = DB_PCA
        self.topk = TOPK

        # Istantiate U2Net for segmentation
        self.u2net = u2netSegmentation.U2NET()

        # Istantiate GroupClassifier for object presence "detection"
        self.groupNeuralModel = groupClassifier()

        # Istantiate CNN for feature extractor
        if not extractor:
            self.extractor = FeatExtractor(db_folder=self.db_path, csv_file=self.db_csv)
        else:
            self.extractor = extractor

        self.data = None
        self.knn = None
        self.scaler = None
        self.pca_scaler = None
        self.pca = None

        # Check if index already exists and load
        if os.path.isfile(DB_INDEX):
            if reset:
                print('Index already exists [resetting]...')
                #os.remove(DB_INDEX)
                if os.path.isfile(DB_KNN_INDEX):
                    os.remove(DB_KNN_INDEX)
                if os.path.isfile(DB_SCALER):
                    os.remove(DB_SCALER)
                self._genIndex()
                self._genIndexKNN()
            else:
                print('Index already exists [loading]...')
                self.data = pd.read_pickle(DB_INDEX)
                if os.path.isfile(DB_KNN_INDEX):
                    self.knn = pickle.load(open(DB_KNN_INDEX, 'rb'))
                    if os.path.isfile(DB_SCALER):
                        self.scaler = pickle.load(open(DB_SCALER, 'rb'))
                    elif os.path.isfile(DB_SCALER_PCA):
                        self.pca_scaler = pickle.load(open(DB_SCALER_PCA, 'rb'))
                        self.pca = pickle.load(open(DB_PCA, 'rb'))
                        self.extractor.scaler = self.pca_scaler
                        self.extractor.pca = self.pca
                else:
                    self._genIndexKNN()
        else:
            print('Index doesn\'t exists [creating]...')
            self._genIndex()

    
    def _genIndex(self):
        print('Generating index...')

        # Check if db folder isn't empty and print number of images
        imgs_name = os.listdir(self.db_path)
        if len(imgs_name) == 0:
            print("No images in DB.")
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        else:
            print("{} images found in DB.".format(len(imgs_name)))

        # Compute neural features in batch (speed optimization)
        start = timer()
        neuralFeatures = self.extractor.get_neuralFeat_batch()
        print('[Neural feature extraction time] {} seconds'.format(round(timer() - start,4)))

        # Compute handcrafted features for every images in loop
        start = timer()
        handFeatures = []
        for img in tqdm(self.db_csv.iterrows(), total=self.db_csv.shape[0]):
            # Image loading with resize
            image = load_image(os.path.join(DB_FOLDER, img[1]['name']))
            # Preprocessing (segmentation)
            #segmented_image, binary_mask = self.u2net.segment(image)
            # load pre-masked images
            mask_name = img[1]['name'].split('.')[0] + '_mask'
            binary_mask = np.array(load_image(os.path.join(DB_FOLDER_MASK, mask_name)).convert('L'))
            #binary_mask = np.ones_like(np.array(image)[:,:,0])
            # Feature extraction
            handFeatures.append(self.extractor.get_handFeatures(image=image, mask=binary_mask))

        print('[Handcrafted feature extraction time] {} seconds'.format(round(timer() - start,4)))

        import pdb; pdb.set_trace()
        # Index table creation
        self.data = pd.DataFrame({
                        'name' : self.db_csv['name'],
                        'class' : self.db_csv['class'],
                        'group' : self.db_csv['group'],
                        'neuralFeat' : neuralFeatures.tolist(),
                        'handFeat' : handFeatures,
                    })

        # Save index to file
        self.data.to_pickle(self.index_filename)
        
        # Feature combination
        start = timer()
        combined = self.extractor.fuse_features([np.vstack(self.data['neuralFeat']), 
                                                    np.vstack(self.data['handFeat'])], pca=False)
        print('[Combined feature time] {} seconds'.format(round(timer() - start,4)))
        self.data['combined'] = combined.tolist()

        # Save index to file
        self.data.to_pickle(self.index_filename)

        # Feature combination with PCA
        start = timer()
        combined_pca, pca_scaler = self.extractor.fuse_features([np.vstack(self.data['neuralFeat']), 
                                                                    np.vstack(self.data['handFeat'])], pca=True)
        print('[Combined feature with PCA time] {} seconds'.format(round(timer() - start,4)))
        self.data['combined_pca'] = combined_pca.tolist()

        # Save index to file
        self.data.to_pickle(self.index_filename)
    
        # Neural features with PCA
        start = timer()
        neuralPCA, pca_scaler, pca = self.extractor.fuse_features([np.vstack(self.data['neuralFeat'])], pca=True)
        print('[Neural feature with PCA time] {} seconds'.format(round(timer() - start,4)))
        self.data['neuralPCA'] = neuralPCA.tolist()

        # Save index to file
        self.data.to_pickle(self.index_filename)

        # Save standard scaler if PCA
        if pca_scaler:
            scalerPickle = open(self.pca_scaler_filename, 'w+b')
            self.scaler = pca_scaler
            pickle.dump(pca_scaler, scalerPickle)
            pcaPickle = open(self.pca_filename, 'w+b')
            self.pca = pca
            pickle.dump(pca, pcaPickle)
            print('PCA and PCA standard scaler saved.')
        


    def _genIndexKNN(self):

        if self.scaler:
            # Standard scaler
            FEATURES = self.data['neuralFeat']
            self.scaler = StandardScaler().fit(FEATURES)
            scalerPickle = open(self.scaler_filename, 'w+b')

            # Save standard scaler
            pickle.dump(self.scaler, scalerPickle)
            print('Scaler saved.')
            FEATURES_SCL = self.scaler.transform(FEATURES)

        elif self.pca_scaler:
            # PCA standard scaler
            self.extractor.scaler = self.pca_scaler
            self.extractor.pca = self.pca
            FEATURES_SCL = np.vstack(self.data['neuralPCA'])
            
        # KD-tree
        #n_neigh = len(self.data) # All
        n_neigh = self.topk
        start = timer()
        knn = NearestNeighbors(n_neighbors=n_neigh, 
                                algorithm='ball_tree', leaf_size=500, metric='euclidean').fit(FEATURES_SCL)
        print('[KNN indexing time] {} seconds'.format(round(timer() - start,4)))
        self.knn = knn

        # Save fitted KNN model with pickle
        knnPickle = open(self.knn_index_filename, 'w+b') 
        pickle.dump(knn, knnPickle)
        print('KNN index saved.')        


    def __len__(self):
        return len(self.data)