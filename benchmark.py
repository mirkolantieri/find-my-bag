## Script for CBIR performance evaluation on single query and entire query-testset
        
import os
import pandas as pd
import numpy as np
import time
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from CBIR.Processing import u2netSegmentation
from CBIR.Indexer import Index
from CBIR.Extractor import FeatExtractor
from CBIR.Matcher import image_retrieval
from CBIR.utils import load_image, show_images, show_images_linear, pil_to_opencv, AP, apk, mapk
from CBIR.Processing.ImageProcessing import ImageProcessing

#from sklearn import svm


root = os.path.dirname(os.path.realpath(__file__))
TEST_IMGS_FOLDER = f"{root}/CBIR/DB/images"
TEST_IMGS_CSV = f"{root}/CBIR/DB/test.csv"

SEG_TEST_FOLDER = f"{root}/CBIR/DB/segmentation_test"
SEG_TEST_IMGS_FOLDER = f"{root}/CBIR/DB/segmentation_test/images"
SEG_TEST_MASK_FOLDER = f"{root}/CBIR/DB/segmentation_test/mask_gt"
SEG_TEST_IMGS_CSV = f"{root}/CBIR/DB/segmentation_test/seg_test.csv"

PROC_TEST_IMGS_FOLDER = f"{root}/CBIR/DB/processing_test/images"
PROC_TEST_IMGS_CSV = f"{root}/CBIR/DB/processing_test/proc_test.csv"


### QUERY BENCHMARK ###
def query_single(index, sample, show_res=False, count=None):

    img_path = os.path.join(TEST_IMGS_FOLDER, sample['name'])
    img = load_image(img_path)

    # Processing
    processing = ImageProcessing(img=pil_to_opencv(img), debug=False)

    # Background remove
    #_, binary_mask = index.u2net.segment(img)

    quality_ok, blur, empty, noisy, contrast  = processing.check_quality()
    if quality_ok:
        enh_image, enh = processing.enhance()
        group = processing.check_object_presence()


    # Feature extraction
    query_features = index.extractor.get_neuralFeatures(image=img)
    
    #binary_mask = np.ones_like(np.array(img)[:,:,0])
    #query_features_h = index.extractor.get_handFeatures(image=img, mask=binary_mask)
    #combined = index.extractor.fuse_features([query_features, query_features_h], pca=True)

    nn_pca, _ = index.extractor.fuse_features([query_features], pca=True)

    # Scaler
    #query_features = index.scaler.transform(query_features_h.reshape(1,-1))
    
    #query_features = combined
    query_features = nn_pca

    # Retrieval
    best_idxs, dist = image_retrieval(query_features, index)

    # Show results
    # Get images with lowest distance with query 
    top_images = []
    for i in best_idxs:
        top_images.append(load_image(os.path.join(index.db_path, index.data.iloc[i]['name'])))

    if show_res:
        show_images_linear(
            img,
            sample['class'],
            sample['group'],
            top_images,
            index.data.loc[best_idxs]['class'].tolist(),
            index.data.loc[best_idxs]['group'].tolist(),
            dist,
            count)
    #import pdb; pdb.set_trace()

    return best_idxs


### PROCESSING BENCHMARK ###
def processing():
    img_dir = PROC_TEST_IMGS_FOLDER
    db_csv = pd.read_csv(PROC_TEST_IMGS_CSV)

    #index = Index(reset=False)

    #oc_svm = svm.OneClassSVM(kernel='rbf')
    #train = np.vstack(index.data['neuralPCA'])
    #train = np.vstack(index.data['handFeat'])
    #oc_svm.fit(train)

    iterationTime = []

    blurs = []
    emptys = []
    noisys = []
    quality_oks = []
    groups = []
    contrasts = []
    names = []


    end = time.time()
    for img in tqdm(db_csv.iterrows(), total=db_csv.shape[0]):
        endIteration = time.time()

        names.append(img[1]['name'])

        image = pil_to_opencv(load_image(os.path.join(img_dir, img[1]['name'])))

        group = None

        processing = ImageProcessing(img=image)
        quality_ok, blur, empty, noisy, contrast  = processing.check_quality()
        if quality_ok:
            enh_image, enh = processing.enhance()
            group = processing.check_object_presence()

        blurs.append(blur)
        emptys.append(not empty)
        noisys.append(noisy)
        quality_oks.append(quality_ok)
        groups.append(group)
        contrasts.append(contrast)

        #image = load_image(os.path.join(img_dir, img[1]['name']))
        #_, binary_mask = index.u2net.segment(image)
        #image = Image.fromarray(cv2.cvtColor(enh_image, cv2.COLOR_BGR2RGB))
        #query_features = index.extractor.get_handFeatures(image=image, mask=binary_mask)
        #query_features = index.extractor.get_neuralFeatures(image=image)
        #query_features = index.extractor.get_groupFeatures(image=image)
        #image_feat, _ = index.extractor.fuse_features([query_features], pca=True)
        #image_feat = index.scaler.transform(query_features.reshape(1,-1))
        #oc_svm_pred = oc_svm.predict(image_feat.reshape(1,-1))
        #print(oc_svm_pred)
        #image.show()
        #new_img, binary_mask = processing.remove_bg()

        iterationTime.append(time.time()-endIteration)


    print('TOTAL TIME: {}'.format(str(time.time() - end)))
    print('Iteration mean time {}'.format(str(np.array(iterationTime).mean())))

    res_csv = pd.DataFrame({'name':names, 'blur':blurs, 'noise':noisys, 'low_contrast':contrasts, 'object_presence': emptys, 'group':groups})
    import pdb; pdb.set_trace()


### SEGMENTATION BENCHMARK ###
def segmentation():
    dir = SEG_TEST_FOLDER
    img_dir = SEG_TEST_IMGS_FOLDER
    db_csv = pd.read_csv(SEG_TEST_IMGS_CSV)

    masked_dir = os.path.join(dir,'masked', 'ourSeg')
    if not os.path.exists(masked_dir):
        os.makedirs(masked_dir)
    
    mask_dir = os.path.join(dir,'mask', 'ourSeg')
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    iterationTime = []

    end = time.time()
    for img in tqdm(db_csv.iterrows(), total=db_csv.shape[0]):
        endIteration = time.time()
    
        image = pil_to_opencv(load_image(os.path.join(img_dir, img[1]['name'])))
        #image = pil_to_opencv(load_image('/CBIR/DB/images/28909.jpg')) #SKU
        #image = pil_to_opencv(load_image('/CBIR/DB/images/28906.jpg')) #real

        import pdb; pdb.set_trace()
        processing = ImageProcessing(img=image)
        #processing.check_quality()
        #enh_image = processing.enhance()
        new_img, binary_mask = processing.remove_bg()

        iterationTime.append(time.time()-endIteration)

        # Save
        name = img[1]['name'].split('.')
        new_img.save(os.path.join(masked_dir, name[0]+'_masked'), format='png')
        Image.fromarray(binary_mask).save(os.path.join(mask_dir, name[0]+'_mask'), format='png')


    print('TOTAL TIME: {}'.format(str(time.time() - end)))
    print('Iteration mean time {}'.format(str(np.array(iterationTime).mean())))


def u2net_segmentation():
    """ Test on segmentation with U2NET """

    dir = SEG_TEST_FOLDER
    #img_dir = SEG_TEST_IMGS_FOLDER
    #db_csv = pd.read_csv(SEG_TEST_IMGS_CSV)
    
    img_dir = TEST_IMGS_FOLDER
    db_csv = pd.read_csv(TEST_IMGS_CSV)

    masked_dir = os.path.join(dir,'masked_all', 'u2netp')
    if not os.path.exists(masked_dir):
        os.makedirs(masked_dir)
    
    mask_dir = os.path.join(dir,'mask_all', 'u2netp')
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    u2net = u2netSegmentation.U2NET()
    
    iterationTime = []

    end = time.time()
    for img in tqdm(db_csv.iterrows(), total=db_csv.shape[0]):

        endIteration = time.time()

        image = load_image(os.path.join(img_dir, img[1]['name']))
        new_img, binary_mask = u2net.segment(image)

        iterationTime.append(time.time()-endIteration)

        # Save
        name = img[1]['name'].split('.')
        new_img.save(os.path.join(masked_dir, name[0]+'_masked'), format='png')
        Image.fromarray(binary_mask).save(os.path.join(mask_dir, name[0]+'_mask'), format='png')


    print('TOTAL TIME: {}'.format(str(time.time() - end)))
    print('Iteration mean time {}'.format(str(np.array(iterationTime).mean())))


def show_overlap(pred,gt):
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    pred[:,:,1] = pred[:,:,0]*255
    pred = Image.fromarray(pred).convert('RGB')
    gt = Image.fromarray(gt).convert('RGB')

    blend = Image.blend(gt, pred, 0.5)
    blend.show()


def iouMeasure():
    dir = SEG_TEST_FOLDER
    db_csv = pd.read_csv(SEG_TEST_IMGS_CSV)
    pred_dir = os.path.join(SEG_TEST_FOLDER, 'mask', 'u2netp')
    gt_dir = SEG_TEST_MASK_FOLDER

    metrics = []
    for img in tqdm(db_csv.iterrows(), total=db_csv.shape[0]):
        name = img[1]['name'].split('.')[0]
        pred = np.array(load_image(os.path.join(pred_dir, name + '_mask')).convert("L"))
        gt = np.array(load_image(os.path.join(gt_dir, name + '_mask.png')).convert("L"))

        mask1_area = np.count_nonzero(np.uint8(pred>128)*255 == 255)
        mask2_area = np.count_nonzero(gt == 255)
        intersection = np.count_nonzero( np.logical_and( pred, gt) )
        iou = np.clip(intersection/(mask1_area+mask2_area-intersection),0,1)
        metrics.append(iou)
        
        #print(iou)
        #show_overlap(pred,gt)

    print('Media: {}'.format(str(np.array(metrics).mean())))
    return metrics





if __name__ == "__main__":

    #### Segmentation benchmark
    #u2net_segmentation()
    #segmentation()
    #metrics = iouMeasure()

    #### Processing benchmark
    #processing()

    #### CBIR benchmark
    ## Creazione/caricamento index
    start = timer()
    index = Index(reset=False)
    print('[Total indexing time] {} seconds'.format(round(timer() - start,4)))

    query_csv = pd.read_csv(TEST_IMGS_CSV)

    ## Search per singola immagine scelta randomicamente tra quelle di test
    samples = query_csv.sample(10, random_state=14)

    tot_relevants = []
    results_class = []
    gt_class = []

    count = 11
    for sample in samples.iterrows():
        count = count+1
        end= time.time()
        ranked_idx = query_single(index, sample[1], show_res=True, count=count)
        print(time.time()-end)
        gt_class.append(np.array([sample[1]['class']]))
        results_class.append(index.data.loc[ranked_idx]['class'].values)
        tot_relevants.append(sum(sample[1]['class']==index.data['class'].values))

    
    for i in range(0,len(results_class)):
        ap5 = apk(gt_class[i], results_class[i], tot_relevant=tot_relevants[i], k=index.topk, all=False)
        print(ap5)

    import pdb; pdb.set_trace()

    ## Tutte le query del test set
    query_time = []
    total_time = time.time()

    tot_relevants = []
    results_class = []
    gt_class = []

    tot_relevants_group = []
    results_group = []
    gt_group = []

    for test_img in tqdm(query_csv.iterrows(), total=query_csv.shape[0]):
        end= time.time()
        ranked_idx = query_single(index, test_img[1], show_res=False)
        query_time.append(time.time()-end)
        gt_class.append(np.array([test_img[1]['class']]))
        results_class.append(index.data.loc[ranked_idx]['class'].values)
        gt_group.append(np.array([test_img[1]['group']]))
        results_group.append(index.data.loc[ranked_idx]['group'].values)
        tot_relevants.append(sum(test_img[1]['class']==index.data['class'].values))
        tot_relevants_group.append(sum(test_img[1]['group']==index.data['group'].values))

    total_time = time.time()-total_time

    ## MAP@5 (KNN search algorithm evaluation)
    #map5 = mapk(gt_class, results_class, tot_relevants, k=index.topk)
    #map5_group = mapk(gt_group, results_group, tot_relevants_group, k=index.topk)

    #print('MAP@5 class: {}'.format(map5))
    #print('MAP@5 group: {}'.format(map5))
    #print(np.around(np.mean(query_time),4))
    #print(np.around(total_time,4))

    
    ## MAP@N for N=5,10,...,100 (Descriptors evaluation)
    all_map_class = []
    for k in range(5,101,5):
        all_map_class.append(mapk(gt_class, results_class, tot_relevants, k))

    tot_map_class = mapk(gt_class, results_class,tot_relevants, all=True)

    all_map_group = []
    for k in range(5,101,5):
        all_map_group.append(mapk(gt_group, results_group,tot_relevants_group, k))

    tot_map_group = mapk(gt_group, results_group, tot_relevants_group, all=True)

    eval_res = pd.DataFrame({'@k':range(5,101,5), 'map_class':all_map_class, 'map_grup':all_map_group})
    eval_res = eval_res.append({'@k':0, 'map_class':tot_map_class, 'map_grup':tot_map_group}, ignore_index=True)
    eval_res.to_csv('hand_nomask.csv')

    print(np.around(np.mean(query_time),4))
    print(np.around(total_time,4))
    import pdb; pdb.set_trace()