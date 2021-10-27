import os, sys, shutil
import numpy as np
import cv2
from PIL import Image
from Bot.Updater import Updater
from CBIR.Indexer import Index
from CBIR.Matcher import image_retrieval
from CBIR.Processing.ImageProcessing import ImageProcessing
from CBIR.utils import load_image, pil_to_opencv, opencv_to_pil


root = os.path.dirname(os.path.realpath(__file__))
print(f'Working directory: {root}')

# folder where images will be received from user.
QUERY_FOLDER = f"{root}/user_query/"

GROUPS = {
    0: 'Borsa',
    1: 'Zaino',
    2: 'Borsello o Marsupio',
    3: 'Portafoglio'}

def imageHandler(bot, message, chat_id, local_filename, debug=False):
    query_image_path = local_filename
    img = pil_to_opencv(load_image(query_image_path))

    # HANDLE RECEIVED IMAGE
    processing = ImageProcessing(bot=bot, chat_id=chat_id, img=img, 
                                    grpChecker=index.groupNeuralModel, debug=debug)
    bot.sendMessage(chat_id, "Immagine acquisita e caricata.")

    ## QUALITY CHECK
    bot.sendMessage(chat_id, "Controllo qualità...")
    quality_ok = processing.check_quality()

    if quality_ok:
        ## ENHANCEMENT
        enh_image, enh = processing.enhance()
        if enh:
            new_img = enh_image.copy()
            if debug:
                bot.sendMessage(chat_id, "L'immagine è stata migliorata...")
                bot.sendImage(chat_id, image=opencv_to_pil(new_img), caption="Immagine migliorata")
        else:
            new_img = img
        
        ## OBJECT PRESENCE CHECK
        group = processing.check_object_presence()

        if group != -1:
            bot.sendMessage(chat_id, "L'oggetto sembra essere uno(a): {}".format(GROUPS[group]))
        else:
            bot.sendMessage(chat_id, "L'oggetto non sembra essere nessuno dei prodotti disponibili nel DB.\nProseguo ugualmente con la ricerca.")		

        # ## BACKGROUND REMOVE
        # no_bg, mask = processing.remove_bg()
        # no_bg, mask = index.u2net.segment(opencv_to_pil(img))
        # if debug:
        # 	bot.sendImage(chat_id, image=no_bg, caption="Rimozione background")

        ## FEATURE EXTRACTION
        nn_features = index.extractor.get_neuralFeatures(image=opencv_to_pil(new_img))
        features, _, _ = index.extractor.fuse_features([nn_features], pca=True)

        ## MATCHING
        results, dist = image_retrieval(features, index)
        best_image = load_image(os.path.join(index.db_path, index.data.iloc[results[0]]['name']))

        ## SEND RESULTS
        bot.sendImage(chat_id, image=best_image, 
                        caption="L'immagine più simile è uno(a) {} appartenente alla classe: {}".format(
                            GROUPS[index.data.loc[results[0]]['group']],
                            str(index.data.loc[results[0]]['class'])
                        ))
        if debug:
            bot.sendMessage(chat_id, "La distanza dalla query è di: {}".format(np.around(dist[0],3)))


        top_images = []
        for i in results:
            _img = cv2.imread(os.path.join(index.db_path, index.data.iloc[i]['name']))
            height = 512
            width = img.shape[1] # keep original height
            dim = (width, height)
            
            top_images.append(cv2.resize(_img, dim))

        top_images_pil = np.hstack(top_images)
        top_images_pil = opencv_to_pil(top_images_pil)

        bot.sendImage(chat_id, image=top_images_pil,
                        caption="Le prime 5 immagini più simili trovate.")

        bot.sendMessage(chat_id, "Grazie per aver usato Find My Bag!")

def commandHandler(bot, message, chat_id, text):
    if text == '/start':
        bot.sendMessage(chat_id, f"Ciao {message['from']['first_name']}, benvenuto in Find My Bag!\nPer iniziare invia una immagine.")
    else:
        bot.sendMessage(chat_id, f"Comando non supportato.")


    
if __name__ == "__main__":
    try:
        bot_id = '1462674082:AAEMUlZr1NpZVOuao3O2QuxFP2cxPq04EUA'
        
        # ## INDEX LOADING
        index = Index(reset=False)

        # Check if folder for user query images exists, if yes, empty
        if os.path.exists(QUERY_FOLDER):
            shutil.rmtree(QUERY_FOLDER)
        os.makedirs(QUERY_FOLDER)

        # START SERVER
        updater = Updater(bot_id, download_folder=QUERY_FOLDER)
        updater.setTextHandler(commandHandler)
        updater.setPhotoHandler(imageHandler)
        updater.start()
        
    except KeyboardInterrupt:
        print('Bye')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)