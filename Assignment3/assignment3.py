# =============================================================================
# ID: vsr266
# Name: Virendra Singh Rajpurohit
# =============================================================================
import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k
        
    def distance(self, featureA, featureB):
        diffs = (featureA - featureB)**2
        return np.sqrt(diffs.sum())

    def train(self, X, y):
        self.X_train=X
        self.y_train=y
        None

    def getRes(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=lambda el:el[1], reverse=True)
        return sortedVotes[0][0]
    
    def getN(self, X_train, y_train, testInstance, k):
        distances = []
        for x in range(len(X_train)):
            dist = self.distance(testInstance, X_train[x])
            distances.append((X_train[x], y_train[x], dist))
        distances.sort(key=lambda el: el[2])
        neighbors = []
        for x in range(k):
            neighbors.append((distances[x][0], distances[x][1]))
        return neighbors
    
    def predict(self, X):
        predictions=[]
        for x in range(len(X)):
            neighbors = self.getN(self.X_train, self.y_train, X[x], self.k)
            result = self.getRes(neighbors)
            predictions.append(result)
        return np.ravel(predictions)
        
class ID3:
    def __init__(self, nbins, data_range):
        self.bin_size = nbins
        self.range = data_range

    def preprocess(self, data):
        norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
        categorical_data = np.floor(self.bin_size*norm_data).astype(int)
        return categorical_data

    def train(self, X, y):
         categorical_data = self.preprocess(X)
         data=[]
         labels=[]
         for i in range(len(categorical_data)):
             a=[]
             for j in categorical_data[i]:
                 a.append(j)
             a.append(y[i])
             data.append(a)
         for i in range(len(data[0])):
             labels.append(i)
             
         self.tree = self.tree(data, labels)

    def entropy(self, data):
        entries = len(data)
        labels = {}
        for feat in data:
            label = feat[-1]
            if label not in labels.keys():
                labels[label] = 0
                labels[label] += 1
        entropy = 0.0
        for key in labels:
            probability = float(labels[key])/entries
            entropy -= probability * np.log2(probability)
        return entropy
    
    def split(self, data, axis, val):
        newData = []
        for feat in data:
            if feat[axis] == val:
                reducedFeat = feat[:axis]
                reducedFeat.extend(feat[axis+1:])
                newData.append(reducedFeat)
        return newData
    
    def chooseBest(self, data):
        features = len(data[0]) - 1
        baseEntropy = self.entropy(data)
        bestInfoGain = -999.0;
        bestFeat = -1
        for i in range(features):
            featList = [ex[i] for ex in data]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:
                newData = self.split(data, i, value)
                probability = len(newData)/float(len(data))
                newEntropy += probability * self.entropy(newData)
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeat = i
        return bestFeat
    
    def majority(self,classList):
        classCount={}
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=lambda el:el[1], reverse = True )
        return sortedClassCount[0][0]
    
    def tree(self, data,labels):
        
        classList = [ex[-1] for ex in data]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(data[0]) == 1:
            return self.majority(classList)
        bestFeat = self.chooseBest(data)
        bestFeatLabel = labels[bestFeat]
        theTree = {bestFeatLabel:{}}
        del(labels[bestFeat])
        featValues = [ex[bestFeat] for ex in data]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            theTree[bestFeatLabel][value] = self.tree(self.split(data, bestFeat, value),subLabels)
        return theTree
    
    def predict(self, X):
        categorical_data = self.preprocess(X)
        query = {}
        predictions=[]

        for row in categorical_data:
            for j in range(len(row)):
                query[j] = row[j]

            predictions.append(self.pre(query, self.tree))
        return np.ravel(predictions)
    
    def pre(self, query, tree):
        predictions=[]
        for key in list(query.keys()):
            if key in list(tree.keys()):
                try :
                    result = tree[key][query[key]]
                except:
                    result = 1
                
                if isinstance(result, dict):
                    return self.pre(query, result)
                else:
                    predictions.append(result)
        return predictions

class Perceptron:
    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w
        self.b = b
    
    def train(self, X, y, steps):
        for step in range(steps):
            prev_w = self.w
            prev_b = self.b
            for i in range(len(X)):
                prediction = self.compute(X[i])
                self.w += self.lr * (y[i] - prediction) * X[i]
                self.b += self.lr * (y[i] - prediction)
            #I have used stopping condition as it was taking 4 minutes on my pc 
            #The stopping condition is logical as the weights and biases doesn't changes after certain number of iterations
            #The following two lines can be removed if avoiding time complexity 
            if (np.array_equal(self.w, prev_w) and np.array_equal(self.b,prev_b)):
                break

    def compute(self, x):
        summation = np.dot(x, self.w) + self.b
        return 1 if summation > 0 else 0
    
    def predict(self, X):
        predictions = []
        for row in X:
            prediction = self.compute(row)
            predictions.append(prediction)
        return np.ravel(predictions)
    
class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = Sigmoid()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()

    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y, steps):
        for s in range(steps):
            i = s % y.size
            if(i == 0):
                X, y = self.shuffle(X,y)
            xi = np.expand_dims(X[i], axis=0)
            yi = np.expand_dims(y[i], axis=0)
            
            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi) 

            grad = self.MSEGrad(pred, yi)
            grad = self.a2.backward(grad)
            grad = self.l2.backward(grad)
            grad = self.a1.backward(grad)
            grad = self.l1.backward(grad)

    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        pred = np.round(pred)
        return np.ravel(pred)

class FCLayer:

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w    #Each column represents all the weights going into an output node
        self.b = b

    def forward(self, input):
        a = np.dot(input, self.w) + self.b
        self.z = list(a)
        return a

    def backward(self, gradients):
        inp = self.z.pop()
        a = gradients.dot(self.w.T)
        self.w += self.lr* gradients.dot(inp)
        self.b += self.lr* gradients
        return a

class Sigmoid:

    def __init__(self):
        None

    def forward(self, input):
        a = 1/(1+np.exp(-input))
        self.y = list(a)
        return a

    def backward(self, gradients):
        o = self.y.pop()
        a = gradients * (o*(1-o))
        return a 

# If we comment out shuffle, it will always pick the same set for the neural network 
        # which produces the same accuracy everytime
        # but this is only for particular block of data 
        # and doesn't say about the accuracy of the whole dataset