# Methods for retrieval result from index

from .utils import euclidean
from timeit import default_timer as timer


def image_retrieval(query_features, index):
    """ Returns dict containing ranked index and loaded image of retrieval result and relative distances """

    #topk = len(index)
    topk = index.topk

    # Brute-force distances
    #start = timer()
    #distances = index.data[['feat']].apply(lambda x: euclidean(x.to_numpy()[0], query_features), axis=1)
    #print('[index time] {} seconds'.format(round(timer() - start,4)))
    # Get index of db image with lowest distance (top-5)
    #best_idxs = distances[:topk].index.tolist()
    #best_idxs = distances[1].flatten()

    #start = timer()
    distances = index.knn.kneighbors(query_features.reshape(1,-1), return_distance=True)
    #print('[search time] {} seconds'.format(round(timer() - start,4)))

    return distances[1].flatten(), distances[0].flatten()