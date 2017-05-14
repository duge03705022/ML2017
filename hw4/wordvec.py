# from argparse import ArgumentParser


import word2vec
import numpy as np
import nltk
import sys


# parser = ArgumentParser()
# parser.add_argument('--train', action='store_true',
#                     help='Set this flag to train word2vec model')
# parser.add_argument('--corpus-path', type=str, default='hp/all',
#                     help='Text file for training')
# parser.add_argument('--model-path', type=str, default='hp/model.bin',
#                     help='Path to save word2vec model')
# parser.add_argument('--plot-num', type=int, default=500,
#                     help='Number of words to perform dimensionality reduction')
# args = parser.parse_args()

# train - train data_path model_output
# load - model_path plot-num


if sys.argv[1]=="train":
    # DEFINE your parameters for training
    print("Phrasing starts")

    MIN_COUNT = 5
    WORDVEC_DIM = 100
    WINDOW = 5
    NEGATIVE_SAMPLES = 0
    ITERATIONS = 0
    MODEL = 1
    LEARNING_RATE = 0.025
    model_name = sys.argv[2] + ".txt"
    model_phrase = sys.argv[2] + "-phrase.txt"

    word2vec.word2phrase(model_name, model_phrase, verbose=True)
    print("===============")
    print("Training starts")
    # train model
    word2vec.word2vec(
        train=model_phrase,
        output=sys.argv[3]+".bin",
        size=WORDVEC_DIM,
        min_count=MIN_COUNT,
        window=WINDOW,
        negative=NEGATIVE_SAMPLES,
        alpha=LEARNING_RATE,
        verbose=False
        )
else:
    # load model for plotting
    model = word2vec.load("hp/"+sys.argv[1])

    vocabs = []                 
    vecs = []                   
    for vocab in model.vocab:
        vocabs.append(vocab)
        vecs.append(model[vocab])
    vecs = np.array(vecs)[:int(sys.argv[2])]
    vocabs = vocabs[:int(sys.argv[2])]

    '''
    Dimensionality Reduction
    '''
    # from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(vecs)


    '''
    Plotting
    '''
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    # filtering
    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
    
    
    plt.figure()
    texts = []
    for i, label in enumerate(vocabs):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):
            x, y = reduced[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

    plt.savefig('hp/%s_%s.png' % (sys.argv[1],sys.argv[2]), dpi=600)
    # plt.show()
