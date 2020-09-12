from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import codecs
import get_vocab_dict as vocabDic
import process_email as procEm
import random
from p6_1 import choose_params 

emails = [500, 2551, 250] 
config = [10, 10, 10]

def read_data_dir(X_train, y_train, X_val, y_val, dic, dir, num_files, porc, spam):
    set_ = set(np.arange(1, num_files + 1 , 1))
    sec = random.sample(set_, k = int(porc))
    
    for i in range(1, num_files + 1) :
        email_contents = codecs.open('{0}/{1:04d}.txt'.format(dir, i), 'r', encoding='utf-8', errors='ignore').read()
        email = procEm.email2TokenList(email_contents)
        email = set(email)
        
        res = [1 if elem in email else 0 for elem in dic.keys()]
        
        if(i in sec):
            X_val = np.vstack((X_val, res))
            y_val = np.vstack((y_val, spam))
        else:
            X_train = np.vstack((X_train, res))
            y_train = np.vstack((y_train, spam))
            
    return X_train, y_train, X_val, y_val




def process_emails(config):
    
    dic = vocabDic.getVocabDict()
    
    X_train = np.empty((0, len(dic)))
    y_train = np.empty((0, 1))
    X_val = np.empty((0, len(dic)))
    y_val = np.empty((0, 1))
    

    X_train, y_train, X_val, y_val = read_data_dir(X_train, y_train, X_val, y_val, dic, 'spam', emails[0], (config[0] * emails[0]) / 100, 1)
    
    X_train, y_train, X_val, y_val = read_data_dir(X_train, y_train, X_val, y_val, dic, 'easy_ham', emails[1], (config[1] * emails[1]) / 100, 0)
    
    X_train, y_train, X_val, y_val = read_data_dir(X_train, y_train, X_val, y_val, dic, 'hard_ham', emails[2], (config[2] * emails[2]) / 100, 0)
    
    return (X_train, y_train, X_val, y_val)

X_train, y_train, X_val, y_val = process_emails(config)

c, sigma = choose_params(X_train, y_train.ravel(), X_val, y_val.ravel())

clf = svm.SVC(kernel='rbf', C=c, gamma = 1 / (2* sigma ** 2))
clf.fit(X_train, y_train.ravel())
print(clf.score(X_val, y_val))