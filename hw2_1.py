import struct
import argparse

import numpy as np
from scipy.stats import multivariate_normal as gaussian_dist


train_image = './train-images-idx3-ubyte'
train_label = './train-labels-idx1-ubyte'
test_image = './t10k-images-idx3-ubyte'
test_label = './t10k-labels-idx1-ubyte'


class NaiveBayes(object):
    def fit(self, images, labels, mode):
        self.prior = dict()
        self.gaussian = dict()
        self.bin_dist = dict()
        classes = set(labels)

        for i in classes:
            imgs = images[labels == i]
            if mode == 1: 
                self.gaussian[i] = {
                    'mean': np.mean(imgs, axis=0),
                    'var': np.var(imgs, axis=0)
                }
            else:
                dist = np.zeros((784, 32), dtype=np.float)
                imgs = imgs // 8
                for img in imgs:
                    for idx, pixel in enumerate(img):
                        dist[idx, pixel] += 1

                self.bin_dist[i] = dist
            self.prior[i] = float(len(imgs))/len(labels)
            
        return

    def predict(self, images, mode):
        predict_result = np.zeros((len(images), 10), dtype=np.float)
        if mode == 1:
            for key, val in self.gaussian.items():
                predict_result[:, key] = gaussian_dist.logpdf(images, mean=val['mean'], cov=val['var'], allow_singular=True) + np.log10(self.prior[key])

        else:
            for x, img in enumerate(images):
                result = np.zeros((10), dtype=np.float)
                img = img // 8
                for label in range(10):
                    result[label] = np.log10(self.prior[label])
                    for i, pixel in enumerate(img):
                        bin_count = self.bin_dist[label][i, pixel]
                        if bin_count == 0:
                            minval = np.min(self.bin_dist[label][i, self.bin_dist[label][i] > 0])
                            result[label] += np.log10(float(minval / np.sum(self.bin_dist[label][i])))
                        else:
                            result[label] += np.log10(float(bin_count / np.sum(self.bin_dist[label][i])))
                    
                predict_result[x] = result

        return predict_result


def read_mnist(image_path, label_path):
    with open(label_path, 'rb') as label:
        magic, n = struct.unpack('>II', label.read(8))
        labels = np.fromfile(label, dtype=np.uint8)
    with open(image_path, 'rb') as image:
        magic, num, rows, cols = struct.unpack('>IIII', image.read(16))
        images = np.fromfile(image, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def display_prob(result, labels):
    for i, img in enumerate(result):
        mg_result = result[i] / np.sum(result[i])
        print (f'Posterior (in log scale):')
        for j, prob in enumerate(mg_result):
            print (f'{j}: {prob}')
        print (f'Prediction: {np.argmax(result[i])}, Ans: {labels[i]}\n')
    predict_result = np.argmax(result, axis=1)
    print (f'Total Accuracy: {len(predict_result[labels==predict_result])/len(labels)*100}%')
    return


def discrete_mode(train_images, train_labels, test_images, test_labels):
    nb = NaiveBayes()

    nb.fit(train_images, train_labels, 0)
    result = nb.predict(test_images, 0)
    
    display_prob(result, test_labels)

    return


def continous_mode(train_images, train_labels, test_images, test_labels):
    nb = NaiveBayes()
    nb.fit(train_images, train_labels, 1)
    result = nb.predict(test_images, 1)
    
    display_prob(result, test_labels)

    return
    

if __name__ == "__main__":
    train_images, train_labels = read_mnist(train_image, train_label)
    test_images, test_labels = read_mnist(test_image, test_label)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=int, help="Mode of bayesian classifer")
    args = parser.parse_args()

    if args.mode == 0:
        discrete_mode(train_images, train_labels, test_images, test_labels)
    else:
        continous_mode(train_images, train_labels, test_images, test_labels)

