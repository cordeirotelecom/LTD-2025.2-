# Machine Learning para Web: Guia Técnico Completo

## Fundamentos de Machine Learning

Machine Learning representa uma revolução paradigmática na computação, permitindo que sistemas computacionais desenvolvam capacidades de aprendizado e tomada de decisão através da análise de dados, sem necessidade de programação explícita para cada cenário específico.

### **Definição Matemática**:

Para um conjunto de dados D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}, onde x representa features e y representa labels, o objetivo do ML é encontrar uma função f: X → Y que minimize uma função de custo L(f).

**Função de Custo Genérica**:
```
L(f) = 1/n ∑ᵢ₌₁ⁿ l(f(xᵢ), yᵢ) + λR(f)
```

Onde:
- l() é a função de perda
- R(f) é o termo de regularização
- λ é o parâmetro de regularização

## Taxonomia de Algoritmos ML

### **1. Aprendizado Supervisionado**

#### **Regressão Linear**:
```javascript
class LinearRegression {
    constructor() {
        this.weights = null;
        this.bias = 0;
        this.learningRate = 0.01;
        this.iterations = 1000;
    }
    
    fit(X, y) {
        const m = X.length;
        const n = X[0].length;
        
        // Inicializar pesos
        this.weights = new Array(n).fill(0);
        this.bias = 0;
        
        // Gradient Descent
        for (let iter = 0; iter < this.iterations; iter++) {
            const predictions = this.predict(X);
            
            // Calcular gradientes
            let dw = new Array(n).fill(0);
            let db = 0;
            
            for (let i = 0; i < m; i++) {
                const error = predictions[i] - y[i];
                db += error;
                
                for (let j = 0; j < n; j++) {
                    dw[j] += error * X[i][j];
                }
            }
            
            // Atualizar parâmetros
            for (let j = 0; j < n; j++) {
                this.weights[j] -= (this.learningRate * dw[j]) / m;
            }
            this.bias -= (this.learningRate * db) / m;
        }
    }
    
    predict(X) {
        return X.map(row => {
            let prediction = this.bias;
            for (let j = 0; j < row.length; j++) {
                prediction += this.weights[j] * row[j];
            }
            return prediction;
        });
    }
    
    score(X, y) {
        const predictions = this.predict(X);
        const totalSumSquares = y.reduce((sum, val) => {
            const mean = y.reduce((a, b) => a + b) / y.length;
            return sum + Math.pow(val - mean, 2);
        }, 0);
        
        const residualSumSquares = predictions.reduce((sum, pred, i) => {
            return sum + Math.pow(y[i] - pred, 2);
        }, 0);
        
        return 1 - (residualSumSquares / totalSumSquares); // R²
    }
}

// Exemplo de uso
const regression = new LinearRegression();
const X = [[1, 2], [2, 3], [3, 4], [4, 5]];
const y = [3, 5, 7, 9];

regression.fit(X, y);
const predictions = regression.predict([[5, 6]]);
console.log('Predição:', predictions[0]); // ~11
```

#### **Árvore de Decisão**:
```javascript
class DecisionTree {
    constructor(maxDepth = 10, minSamples = 2) {
        this.maxDepth = maxDepth;
        this.minSamples = minSamples;
        this.tree = null;
    }
    
    calculateGini(y) {
        const counts = {};
        y.forEach(label => {
            counts[label] = (counts[label] || 0) + 1;
        });
        
        let gini = 1;
        const total = y.length;
        
        Object.values(counts).forEach(count => {
            const probability = count / total;
            gini -= probability * probability;
        });
        
        return gini;
    }
    
    findBestSplit(X, y) {
        let bestGini = Infinity;
        let bestFeature = null;
        let bestThreshold = null;
        
        for (let featureIndex = 0; featureIndex < X[0].length; featureIndex++) {
            const values = X.map(row => row[featureIndex]);
            const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
            
            for (let i = 0; i < uniqueValues.length - 1; i++) {
                const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
                
                const leftIndices = [];
                const rightIndices = [];
                
                X.forEach((row, index) => {
                    if (row[featureIndex] <= threshold) {
                        leftIndices.push(index);
                    } else {
                        rightIndices.push(index);
                    }
                });
                
                if (leftIndices.length === 0 || rightIndices.length === 0) continue;
                
                const leftY = leftIndices.map(i => y[i]);
                const rightY = rightIndices.map(i => y[i]);
                
                const weightedGini = 
                    (leftY.length / y.length) * this.calculateGini(leftY) +
                    (rightY.length / y.length) * this.calculateGini(rightY);
                
                if (weightedGini < bestGini) {
                    bestGini = weightedGini;
                    bestFeature = featureIndex;
                    bestThreshold = threshold;
                }
            }
        }
        
        return { feature: bestFeature, threshold: bestThreshold, gini: bestGini };
    }
    
    buildTree(X, y, depth = 0) {
        // Condições de parada
        if (depth >= this.maxDepth || 
            y.length < this.minSamples || 
            new Set(y).size === 1) {
            
            const counts = {};
            y.forEach(label => {
                counts[label] = (counts[label] || 0) + 1;
            });
            
            const prediction = Object.keys(counts).reduce((a, b) => 
                counts[a] > counts[b] ? a : b
            );
            
            return { prediction, samples: y.length };
        }
        
        const split = this.findBestSplit(X, y);
        
        if (split.feature === null) {
            const counts = {};
            y.forEach(label => {
                counts[label] = (counts[label] || 0) + 1;
            });
            
            const prediction = Object.keys(counts).reduce((a, b) => 
                counts[a] > counts[b] ? a : b
            );
            
            return { prediction, samples: y.length };
        }
        
        const leftIndices = [];
        const rightIndices = [];
        
        X.forEach((row, index) => {
            if (row[split.feature] <= split.threshold) {
                leftIndices.push(index);
            } else {
                rightIndices.push(index);
            }
        });
        
        const leftX = leftIndices.map(i => X[i]);
        const leftY = leftIndices.map(i => y[i]);
        const rightX = rightIndices.map(i => X[i]);
        const rightY = rightIndices.map(i => y[i]);
        
        return {
            feature: split.feature,
            threshold: split.threshold,
            left: this.buildTree(leftX, leftY, depth + 1),
            right: this.buildTree(rightX, rightY, depth + 1),
            samples: y.length
        };
    }
    
    fit(X, y) {
        this.tree = this.buildTree(X, y);
    }
    
    predictSingle(x, node = this.tree) {
        if (node.prediction !== undefined) {
            return node.prediction;
        }
        
        if (x[node.feature] <= node.threshold) {
            return this.predictSingle(x, node.left);
        } else {
            return this.predictSingle(x, node.right);
        }
    }
    
    predict(X) {
        return X.map(x => this.predictSingle(x));
    }
}

// Exemplo de uso
const tree = new DecisionTree();
const X_tree = [[1, 2], [2, 3], [3, 1], [4, 2], [5, 3], [6, 1]];
const y_tree = ['A', 'A', 'B', 'B', 'A', 'B'];

tree.fit(X_tree, y_tree);
const treePredictions = tree.predict([[2.5, 2.5], [5.5, 1.5]]);
console.log('Predições da árvore:', treePredictions);
```

### **2. Aprendizado Não Supervisionado**

#### **K-Means Clustering**:
```javascript
class KMeans {
    constructor(k = 3, maxIterations = 100, tolerance = 1e-4) {
        this.k = k;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.centroids = [];
        this.labels = [];
    }
    
    euclideanDistance(point1, point2) {
        return Math.sqrt(
            point1.reduce((sum, val, i) => 
                sum + Math.pow(val - point2[i], 2), 0
            )
        );
    }
    
    initializeCentroids(data) {
        const dimensions = data[0].length;
        this.centroids = [];
        
        for (let i = 0; i < this.k; i++) {
            const centroid = [];
            for (let d = 0; d < dimensions; d++) {
                const min = Math.min(...data.map(point => point[d]));
                const max = Math.max(...data.map(point => point[d]));
                centroid.push(Math.random() * (max - min) + min);
            }
            this.centroids.push(centroid);
        }
    }
    
    assignClusters(data) {
        const newLabels = data.map(point => {
            let minDistance = Infinity;
            let closestCentroid = 0;
            
            this.centroids.forEach((centroid, index) => {
                const distance = this.euclideanDistance(point, centroid);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCentroid = index;
                }
            });
            
            return closestCentroid;
        });
        
        return newLabels;
    }
    
    updateCentroids(data, labels) {
        const newCentroids = [];
        const dimensions = data[0].length;
        
        for (let k = 0; k < this.k; k++) {
            const clusterPoints = data.filter((_, i) => labels[i] === k);
            
            if (clusterPoints.length === 0) {
                newCentroids.push([...this.centroids[k]]);
                continue;
            }
            
            const newCentroid = [];
            for (let d = 0; d < dimensions; d++) {
                const mean = clusterPoints.reduce((sum, point) => 
                    sum + point[d], 0) / clusterPoints.length;
                newCentroid.push(mean);
            }
            newCentroids.push(newCentroid);
        }
        
        return newCentroids;
    }
    
    centroidsChanged(oldCentroids, newCentroids) {
        for (let i = 0; i < oldCentroids.length; i++) {
            const distance = this.euclideanDistance(oldCentroids[i], newCentroids[i]);
            if (distance > this.tolerance) {
                return true;
            }
        }
        return false;
    }
    
    fit(data) {
        this.initializeCentroids(data);
        
        for (let iteration = 0; iteration < this.maxIterations; iteration++) {
            // Atribuir pontos aos clusters
            const newLabels = this.assignClusters(data);
            
            // Atualizar centroides
            const oldCentroids = this.centroids.map(c => [...c]);
            this.centroids = this.updateCentroids(data, newLabels);
            
            // Verificar convergência
            if (!this.centroidsChanged(oldCentroids, this.centroids)) {
                console.log(`Convergiu na iteração ${iteration + 1}`);
                break;
            }
            
            this.labels = newLabels;
        }
    }
    
    predict(data) {
        return this.assignClusters(data);
    }
    
    calculateInertia(data) {
        let inertia = 0;
        data.forEach((point, i) => {
            const centroid = this.centroids[this.labels[i]];
            inertia += Math.pow(this.euclideanDistance(point, centroid), 2);
        });
        return inertia;
    }
}

// Exemplo de uso
const kmeans = new KMeans(2);
const clusterData = [
    [1, 2], [1, 4], [1, 0],
    [4, 2], [4, 4], [4, 0],
    [7, 2], [7, 4], [7, 0]
];

kmeans.fit(clusterData);
console.log('Centroides:', kmeans.centroids);
console.log('Labels:', kmeans.labels);
console.log('Inércia:', kmeans.calculateInertia(clusterData));
```

## Machine Learning com TensorFlow.js

### **Rede Neural para Classificação**:
```javascript
import * as tf from '@tensorflow/tfjs';

class NeuralNetworkClassifier {
    constructor() {
        this.model = null;
        this.isCompiled = false;
    }
    
    createModel(inputShape, numClasses, hiddenLayers = [64, 32]) {
        this.model = tf.sequential();
        
        // Camada de entrada
        this.model.add(tf.layers.dense({
            inputShape: [inputShape],
            units: hiddenLayers[0],
            activation: 'relu',
            kernelInitializer: 'glorotUniform'
        }));
        
        // Camadas ocultas
        for (let i = 1; i < hiddenLayers.length; i++) {
            this.model.add(tf.layers.dropout({ rate: 0.3 }));
            this.model.add(tf.layers.dense({
                units: hiddenLayers[i],
                activation: 'relu',
                kernelInitializer: 'glorotUniform'
            }));
        }
        
        // Camada de saída
        this.model.add(tf.layers.dropout({ rate: 0.5 }));
        this.model.add(tf.layers.dense({
            units: numClasses,
            activation: numClasses === 1 ? 'sigmoid' : 'softmax',
            kernelInitializer: 'glorotUniform'
        }));
        
        // Compilar modelo
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: numClasses === 1 ? 'binaryCrossentropy' : 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        this.isCompiled = true;
        
        console.log('Modelo criado:');
        this.model.summary();
    }
    
    async train(X, y, options = {}) {
        if (!this.isCompiled) {
            throw new Error('Modelo não foi compilado. Use createModel() primeiro.');
        }
        
        const defaultOptions = {
            epochs: 100,
            batchSize: 32,
            validationSplit: 0.2,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (epoch % 10 === 0) {
                        console.log(`Época ${epoch}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
                    }
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        
        // Converter dados para tensores
        const xs = tf.tensor2d(X);
        const ys = tf.tensor2d(y);
        
        try {
            const history = await this.model.fit(xs, ys, finalOptions);
            
            // Limpar memória
            xs.dispose();
            ys.dispose();
            
            return history;
        } catch (error) {
            xs.dispose();
            ys.dispose();
            throw error;
        }
    }
    
    predict(X) {
        if (!this.model) {
            throw new Error('Modelo não foi treinado.');
        }
        
        const xs = tf.tensor2d(X);
        const predictions = this.model.predict(xs);
        
        // Converter para array JavaScript
        const results = predictions.dataSync();
        
        // Limpar memória
        xs.dispose();
        predictions.dispose();
        
        return Array.from(results);
    }
    
    evaluate(X, y) {
        const xs = tf.tensor2d(X);
        const ys = tf.tensor2d(y);
        
        const evaluation = this.model.evaluate(xs, ys);
        
        // Extrair loss e accuracy
        const loss = evaluation[0].dataSync()[0];
        const accuracy = evaluation[1].dataSync()[0];
        
        // Limpar memória
        xs.dispose();
        ys.dispose();
        evaluation.forEach(tensor => tensor.dispose());
        
        return { loss, accuracy };
    }
    
    async saveModel(path) {
        if (!this.model) {
            throw new Error('Nenhum modelo para salvar.');
        }
        
        await this.model.save(path);
        console.log(`Modelo salvo em: ${path}`);
    }
    
    async loadModel(path) {
        this.model = await tf.loadLayersModel(path);
        this.isCompiled = true;
        console.log(`Modelo carregado de: ${path}`);
    }
    
    getModelSummary() {
        if (!this.model) {
            return 'Nenhum modelo disponível.';
        }
        
        let summary = 'Arquitetura do Modelo:\n';
        this.model.layers.forEach((layer, index) => {
            summary += `Camada ${index + 1}: ${layer.constructor.name}`;
            if (layer.units) summary += ` (${layer.units} neurônios)`;
            if (layer.activation) summary += ` - Ativação: ${layer.activation.constructor.name}`;
            summary += '\n';
        });
        
        return summary;
    }
}

// Sistema de Preprocessamento de Dados
class DataPreprocessor {
    constructor() {
        this.scalers = {};
        this.encoders = {};
    }
    
    standardScale(data, feature = null) {
        if (feature && this.scalers[feature]) {
            // Usar scaler existente
            const { mean, std } = this.scalers[feature];
            return data.map(val => (val - mean) / std);
        }
        
        // Criar novo scaler
        const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
        const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
        const std = Math.sqrt(variance);
        
        if (feature) {
            this.scalers[feature] = { mean, std };
        }
        
        return data.map(val => (val - mean) / std);
    }
    
    minMaxScale(data, feature = null) {
        if (feature && this.scalers[feature]) {
            const { min, max } = this.scalers[feature];
            return data.map(val => (val - min) / (max - min));
        }
        
        const min = Math.min(...data);
        const max = Math.max(...data);
        
        if (feature) {
            this.scalers[feature] = { min, max };
        }
        
        return data.map(val => (val - min) / (max - min));
    }
    
    oneHotEncode(data, feature = null) {
        if (feature && this.encoders[feature]) {
            const { categories } = this.encoders[feature];
            return data.map(val => {
                const encoded = new Array(categories.length).fill(0);
                const index = categories.indexOf(val);
                if (index !== -1) encoded[index] = 1;
                return encoded;
            });
        }
        
        const categories = [...new Set(data)];
        
        if (feature) {
            this.encoders[feature] = { categories };
        }
        
        return data.map(val => {
            const encoded = new Array(categories.length).fill(0);
            const index = categories.indexOf(val);
            if (index !== -1) encoded[index] = 1;
            return encoded;
        });
    }
    
    trainTestSplit(X, y, testSize = 0.2, randomState = null) {
        if (randomState !== null) {
            Math.seedrandom(randomState);
        }
        
        const indices = Array.from({ length: X.length }, (_, i) => i);
        
        // Shuffle indices
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        
        const splitIndex = Math.floor(X.length * (1 - testSize));
        const trainIndices = indices.slice(0, splitIndex);
        const testIndices = indices.slice(splitIndex);
        
        return {
            XTrain: trainIndices.map(i => X[i]),
            XTest: testIndices.map(i => X[i]),
            yTrain: trainIndices.map(i => y[i]),
            yTest: testIndices.map(i => y[i])
        };
    }
}

// Exemplo de uso completo
async function exemploClassificacao() {
    // Dados de exemplo (Iris dataset simplificado)
    const rawData = [
        [5.1, 3.5, 1.4, 0.2, 'setosa'],
        [4.9, 3.0, 1.4, 0.2, 'setosa'],
        [7.0, 3.2, 4.7, 1.4, 'versicolor'],
        [6.4, 3.2, 4.5, 1.5, 'versicolor'],
        [6.3, 3.3, 6.0, 2.5, 'virginica'],
        [5.8, 2.7, 5.1, 1.9, 'virginica']
        // ... mais dados
    ];
    
    // Preprocessamento
    const preprocessor = new DataPreprocessor();
    
    // Separar features e labels
    const X = rawData.map(row => row.slice(0, 4));
    const y = rawData.map(row => row[4]);
    
    // Normalizar features
    const XNormalized = X.map(row => [
        preprocessor.standardScale([row[0]], 'feature0')[0],
        preprocessor.standardScale([row[1]], 'feature1')[0],
        preprocessor.standardScale([row[2]], 'feature2')[0],
        preprocessor.standardScale([row[3]], 'feature3')[0]
    ]);
    
    // Codificar labels
    const yEncoded = preprocessor.oneHotEncode(y, 'species');
    
    // Dividir dados
    const { XTrain, XTest, yTrain, yTest } = preprocessor.trainTestSplit(
        XNormalized, yEncoded, 0.2, 42
    );
    
    // Criar e treinar modelo
    const classifier = new NeuralNetworkClassifier();
    classifier.createModel(4, 3, [8, 4]);
    
    console.log('Iniciando treinamento...');
    const history = await classifier.train(XTrain, yTrain, {
        epochs: 50,
        batchSize: 16,
        validationSplit: 0.2
    });
    
    // Avaliar modelo
    const evaluation = classifier.evaluate(XTest, yTest);
    console.log(`Acurácia no teste: ${(evaluation.accuracy * 100).toFixed(2)}%`);
    
    // Fazer predições
    const predictions = classifier.predict(XTest);
    console.log('Predições:', predictions);
    
    // Salvar modelo
    await classifier.saveModel('file://./modelo-iris');
}
```

## Aplicações Práticas Web

### **Sistema de Recomendação**:
```javascript
class RecommendationEngine {
    constructor() {
        this.userItemMatrix = new Map();
        this.itemFeatures = new Map();
        this.userProfiles = new Map();
    }
    
    addInteraction(userId, itemId, rating, timestamp = Date.now()) {
        if (!this.userItemMatrix.has(userId)) {
            this.userItemMatrix.set(userId, new Map());
        }
        
        this.userItemMatrix.get(userId).set(itemId, {
            rating,
            timestamp
        });
    }
    
    addItemFeatures(itemId, features) {
        this.itemFeatures.set(itemId, features);
    }
    
    calculateCosineSimilarity(vector1, vector2) {
        const dotProduct = vector1.reduce((sum, val, i) => sum + val * vector2[i], 0);
        const magnitude1 = Math.sqrt(vector1.reduce((sum, val) => sum + val * val, 0));
        const magnitude2 = Math.sqrt(vector2.reduce((sum, val) => sum + val * val, 0));
        
        if (magnitude1 === 0 || magnitude2 === 0) return 0;
        
        return dotProduct / (magnitude1 * magnitude2);
    }
    
    findSimilarUsers(targetUserId, k = 5) {
        const targetUserItems = this.userItemMatrix.get(targetUserId);
        if (!targetUserItems) return [];
        
        const similarities = [];
        
        for (const [userId, userItems] of this.userItemMatrix) {
            if (userId === targetUserId) continue;
            
            // Encontrar itens em comum
            const commonItems = [];
            const targetVector = [];
            const userVector = [];
            
            for (const [itemId, interaction] of targetUserItems) {
                if (userItems.has(itemId)) {
                    commonItems.push(itemId);
                    targetVector.push(interaction.rating);
                    userVector.push(userItems.get(itemId).rating);
                }
            }
            
            if (commonItems.length > 0) {
                const similarity = this.calculateCosineSimilarity(targetVector, userVector);
                similarities.push({ userId, similarity, commonItems: commonItems.length });
            }
        }
        
        return similarities
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, k);
    }
    
    recommendItemsCollaborative(userId, k = 10) {
        const similarUsers = this.findSimilarUsers(userId);
        const userItems = this.userItemMatrix.get(userId) || new Map();
        const recommendations = new Map();
        
        for (const { userId: similarUserId, similarity } of similarUsers) {
            const similarUserItems = this.userItemMatrix.get(similarUserId);
            
            for (const [itemId, interaction] of similarUserItems) {
                // Apenas recomendar itens que o usuário ainda não interagiu
                if (!userItems.has(itemId)) {
                    const currentScore = recommendations.get(itemId) || 0;
                    const weightedRating = interaction.rating * similarity;
                    recommendations.set(itemId, currentScore + weightedRating);
                }
            }
        }
        
        return Array.from(recommendations.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, k)
            .map(([itemId, score]) => ({ itemId, score }));
    }
    
    recommendItemsContentBased(userId, k = 10) {
        const userItems = this.userItemMatrix.get(userId);
        if (!userItems) return [];
        
        // Construir perfil do usuário baseado em features dos itens
        const userProfile = new Map();
        let totalRating = 0;
        
        for (const [itemId, interaction] of userItems) {
            const features = this.itemFeatures.get(itemId);
            if (!features) continue;
            
            totalRating += interaction.rating;
            
            for (const [feature, value] of Object.entries(features)) {
                const currentValue = userProfile.get(feature) || 0;
                userProfile.set(feature, currentValue + interaction.rating * value);
            }
        }
        
        // Normalizar perfil do usuário
        for (const [feature, value] of userProfile) {
            userProfile.set(feature, value / totalRating);
        }
        
        this.userProfiles.set(userId, userProfile);
        
        // Calcular similaridade com itens não interagidos
        const recommendations = [];
        
        for (const [itemId, features] of this.itemFeatures) {
            if (userItems.has(itemId)) continue;
            
            // Calcular similaridade entre perfil do usuário e item
            const itemVector = Object.values(features);
            const userVector = Object.keys(features).map(feature => 
                userProfile.get(feature) || 0
            );
            
            const similarity = this.calculateCosineSimilarity(userVector, itemVector);
            recommendations.push({ itemId, score: similarity });
        }
        
        return recommendations
            .sort((a, b) => b.score - a.score)
            .slice(0, k);
    }
    
    hybridRecommendation(userId, k = 10, collaborativeWeight = 0.6) {
        const collaborative = this.recommendItemsCollaborative(userId, k * 2);
        const contentBased = this.recommendItemsContentBased(userId, k * 2);
        
        const hybridScores = new Map();
        
        // Combinar scores colaborativo
        collaborative.forEach(({ itemId, score }) => {
            hybridScores.set(itemId, score * collaborativeWeight);
        });
        
        // Combinar scores baseado em conteúdo
        contentBased.forEach(({ itemId, score }) => {
            const currentScore = hybridScores.get(itemId) || 0;
            hybridScores.set(itemId, currentScore + score * (1 - collaborativeWeight));
        });
        
        return Array.from(hybridScores.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, k)
            .map(([itemId, score]) => ({ itemId, score }));
    }
    
    getRecommendationExplanation(userId, itemId) {
        const similarUsers = this.findSimilarUsers(userId, 3);
        const userProfile = this.userProfiles.get(userId);
        const itemFeatures = this.itemFeatures.get(itemId);
        
        let explanation = `Recomendação para o item ${itemId}:\n`;
        
        // Explicação colaborativa
        if (similarUsers.length > 0) {
            explanation += '\nBaseado em usuários similares:\n';
            similarUsers.forEach(({ userId: simUserId, similarity }) => {
                const simUserItems = this.userItemMatrix.get(simUserId);
                if (simUserItems.has(itemId)) {
                    const rating = simUserItems.get(itemId).rating;
                    explanation += `- Usuário ${simUserId} (similaridade: ${similarity.toFixed(2)}) avaliou com ${rating}\n`;
                }
            });
        }
        
        // Explicação baseada em conteúdo
        if (userProfile && itemFeatures) {
            explanation += '\nBaseado no seu perfil:\n';
            for (const [feature, value] of Object.entries(itemFeatures)) {
                const userPreference = userProfile.get(feature) || 0;
                if (userPreference > 0.5 && value > 0.5) {
                    explanation += `- Você gosta de ${feature} (${userPreference.toFixed(2)}) e este item tem ${value.toFixed(2)}\n`;
                }
            }
        }
        
        return explanation;
    }
}

// Exemplo de uso do sistema de recomendação
const engine = new RecommendationEngine();

// Adicionar interações de usuários
engine.addInteraction('user1', 'item1', 5);
engine.addInteraction('user1', 'item2', 3);
engine.addInteraction('user2', 'item1', 4);
engine.addInteraction('user2', 'item3', 5);

// Adicionar features de itens
engine.addItemFeatures('item1', { action: 0.8, comedy: 0.2, drama: 0.1 });
engine.addItemFeatures('item2', { action: 0.1, comedy: 0.9, drama: 0.2 });
engine.addItemFeatures('item3', { action: 0.7, comedy: 0.1, drama: 0.8 });

// Obter recomendações
const recommendations = engine.hybridRecommendation('user1', 5);
console.log('Recomendações:', recommendations);
```

### **Análise de Sentimentos em Tempo Real**:
```javascript
class SentimentAnalyzer {
    constructor() {
        this.positiveWords = new Set(['bom', 'ótimo', 'excelente', 'amor', 'feliz', 'incrível']);
        this.negativeWords = new Set(['ruim', 'péssimo', 'ódio', 'triste', 'terrível', 'horrível']);
        this.intensifiers = new Map([
            ['muito', 1.5],
            ['extremamente', 2.0],
            ['super', 1.3],
            ['não', -1.0]
        ]);
        
        this.emoticons = new Map([
            ['😊', 1], ['😄', 1], ['😍', 2], ['❤️', 2],
            ['😢', -1], ['😠', -2], ['😡', -2], ['💔', -2]
        ]);
    }
    
    preprocessText(text) {
        return text
            .toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
    }
    
    analyzeSentiment(text) {
        const originalText = text;
        const cleanText = this.preprocessText(text);
        const words = cleanText.split(' ');
        
        let score = 0;
        let positiveCount = 0;
        let negativeCount = 0;
        let currentIntensifier = 1;
        
        // Analisar emoticons
        for (const [emoticon, value] of this.emoticons) {
            const count = (originalText.match(new RegExp(emoticon, 'g')) || []).length;
            score += count * value;
        }
        
        // Analisar palavras
        for (let i = 0; i < words.length; i++) {
            const word = words[i];
            
            // Verificar intensificadores
            if (this.intensifiers.has(word)) {
                currentIntensifier = this.intensifiers.get(word);
                continue;
            }
            
            // Analisar sentimento
            if (this.positiveWords.has(word)) {
                score += 1 * currentIntensifier;
                positiveCount++;
            } else if (this.negativeWords.has(word)) {
                score += -1 * currentIntensifier;
                negativeCount++;
            }
            
            // Reset intensifier
            currentIntensifier = 1;
        }
        
        // Normalizar score
        const totalWords = words.length;
        const normalizedScore = totalWords > 0 ? score / totalWords : 0;
        
        // Classificar sentimento
        let sentiment;
        if (normalizedScore > 0.1) sentiment = 'positivo';
        else if (normalizedScore < -0.1) sentiment = 'negativo';
        else sentiment = 'neutro';
        
        return {
            text: originalText,
            sentiment,
            score: normalizedScore,
            confidence: Math.abs(normalizedScore),
            details: {
                positiveWords: positiveCount,
                negativeWords: negativeCount,
                totalWords,
                rawScore: score
            }
        };
    }
    
    analyzeBatch(texts) {
        return texts.map(text => this.analyzeSentiment(text));
    }
    
    getAggregateStats(analyses) {
        const total = analyses.length;
        const positive = analyses.filter(a => a.sentiment === 'positivo').length;
        const negative = analyses.filter(a => a.sentiment === 'negativo').length;
        const neutral = analyses.filter(a => a.sentiment === 'neutro').length;
        
        const avgScore = analyses.reduce((sum, a) => sum + a.score, 0) / total;
        const avgConfidence = analyses.reduce((sum, a) => sum + a.confidence, 0) / total;
        
        return {
            total,
            distribution: {
                positive: (positive / total * 100).toFixed(1) + '%',
                negative: (negative / total * 100).toFixed(1) + '%',
                neutral: (neutral / total * 100).toFixed(1) + '%'
            },
            averageScore: avgScore.toFixed(3),
            averageConfidence: avgConfidence.toFixed(3),
            mostPositive: analyses.reduce((max, current) => 
                current.score > max.score ? current : max
            ),
            mostNegative: analyses.reduce((min, current) => 
                current.score < min.score ? current : min
            )
        };
    }
}

// Sistema de monitoramento em tempo real
class RealTimeSentimentMonitor {
    constructor() {
        this.analyzer = new SentimentAnalyzer();
        this.buffer = [];
        this.maxBufferSize = 1000;
        this.subscribers = [];
    }
    
    subscribe(callback) {
        this.subscribers.push(callback);
    }
    
    unsubscribe(callback) {
        this.subscribers = this.subscribers.filter(sub => sub !== callback);
    }
    
    notify(data) {
        this.subscribers.forEach(callback => callback(data));
    }
    
    processText(text, metadata = {}) {
        const analysis = this.analyzer.analyzeSentiment(text);
        
        const entry = {
            ...analysis,
            timestamp: new Date(),
            metadata
        };
        
        this.buffer.push(entry);
        
        // Manter buffer dentro do limite
        if (this.buffer.length > this.maxBufferSize) {
            this.buffer.shift();
        }
        
        // Notificar subscribers
        this.notify({
            type: 'newAnalysis',
            data: entry,
            stats: this.getRecentStats()
        });
        
        return entry;
    }
    
    getRecentStats(minutes = 10) {
        const cutoff = new Date(Date.now() - minutes * 60 * 1000);
        const recent = this.buffer.filter(entry => entry.timestamp > cutoff);
        
        if (recent.length === 0) return null;
        
        return this.analyzer.getAggregateStats(recent);
    }
    
    exportData(format = 'json') {
        switch (format) {
            case 'csv':
                const headers = 'timestamp,text,sentiment,score,confidence\n';
                const rows = this.buffer.map(entry => 
                    `"${entry.timestamp.toISOString()}","${entry.text}","${entry.sentiment}",${entry.score},${entry.confidence}`
                ).join('\n');
                return headers + rows;
            
            case 'json':
            default:
                return JSON.stringify(this.buffer, null, 2);
        }
    }
}

// Exemplo de uso
const monitor = new RealTimeSentimentMonitor();

// Subscriber para alertas
monitor.subscribe((update) => {
    if (update.data.sentiment === 'negativo' && update.data.confidence > 0.7) {
        console.log('ALERTA: Sentimento muito negativo detectado!', update.data.text);
    }
});

// Subscriber para dashboard
monitor.subscribe((update) => {
    const stats = update.stats;
    if (stats) {
        console.log(`Dashboard: ${stats.distribution.positive} positivo, ${stats.distribution.negative} negativo`);
    }
});

// Processar textos
monitor.processText('Eu amo este produto! 😍');
monitor.processText('Esse serviço é péssimo 😠');
monitor.processText('Produto ok, nada especial');
```

## Métricas de Avaliação

### **Sistema de Métricas ML**:
```javascript
class MLMetrics {
    static confusionMatrix(yTrue, yPred, labels) {
        const matrix = {};
        labels.forEach(trueLabel => {
            matrix[trueLabel] = {};
            labels.forEach(predLabel => {
                matrix[trueLabel][predLabel] = 0;
            });
        });
        
        yTrue.forEach((trueLabel, i) => {
            const predLabel = yPred[i];
            matrix[trueLabel][predLabel]++;
        });
        
        return matrix;
    }
    
    static accuracy(yTrue, yPred) {
        const correct = yTrue.filter((val, i) => val === yPred[i]).length;
        return correct / yTrue.length;
    }
    
    static precision(yTrue, yPred, positiveLabel) {
        const tp = yTrue.filter((val, i) => val === positiveLabel && yPred[i] === positiveLabel).length;
        const fp = yTrue.filter((val, i) => val !== positiveLabel && yPred[i] === positiveLabel).length;
        
        return tp + fp === 0 ? 0 : tp / (tp + fp);
    }
    
    static recall(yTrue, yPred, positiveLabel) {
        const tp = yTrue.filter((val, i) => val === positiveLabel && yPred[i] === positiveLabel).length;
        const fn = yTrue.filter((val, i) => val === positiveLabel && yPred[i] !== positiveLabel).length;
        
        return tp + fn === 0 ? 0 : tp / (tp + fn);
    }
    
    static f1Score(yTrue, yPred, positiveLabel) {
        const prec = this.precision(yTrue, yPred, positiveLabel);
        const rec = this.recall(yTrue, yPred, positiveLabel);
        
        return prec + rec === 0 ? 0 : 2 * (prec * rec) / (prec + rec);
    }
    
    static classificationReport(yTrue, yPred, labels) {
        const report = {};
        
        labels.forEach(label => {
            report[label] = {
                precision: this.precision(yTrue, yPred, label),
                recall: this.recall(yTrue, yPred, label),
                f1Score: this.f1Score(yTrue, yPred, label),
                support: yTrue.filter(val => val === label).length
            };
        });
        
        // Métricas macro
        const macroAvg = {
            precision: labels.reduce((sum, label) => sum + report[label].precision, 0) / labels.length,
            recall: labels.reduce((sum, label) => sum + report[label].recall, 0) / labels.length,
            f1Score: labels.reduce((sum, label) => sum + report[label].f1Score, 0) / labels.length
        };
        
        report.macroAvg = macroAvg;
        report.accuracy = this.accuracy(yTrue, yPred);
        
        return report;
    }
    
    static meanSquaredError(yTrue, yPred) {
        const mse = yTrue.reduce((sum, val, i) => {
            return sum + Math.pow(val - yPred[i], 2);
        }, 0) / yTrue.length;
        
        return mse;
    }
    
    static meanAbsoluteError(yTrue, yPred) {
        const mae = yTrue.reduce((sum, val, i) => {
            return sum + Math.abs(val - yPred[i]);
        }, 0) / yTrue.length;
        
        return mae;
    }
    
    static rSquared(yTrue, yPred) {
        const yMean = yTrue.reduce((sum, val) => sum + val, 0) / yTrue.length;
        
        const totalSumSquares = yTrue.reduce((sum, val) => {
            return sum + Math.pow(val - yMean, 2);
        }, 0);
        
        const residualSumSquares = yTrue.reduce((sum, val, i) => {
            return sum + Math.pow(val - yPred[i], 2);
        }, 0);
        
        return 1 - (residualSumSquares / totalSumSquares);
    }
}

// Exemplo de uso das métricas
const yTrue = ['A', 'B', 'A', 'B', 'A'];
const yPred = ['A', 'A', 'A', 'B', 'B'];
const labels = ['A', 'B'];

const report = MLMetrics.classificationReport(yTrue, yPred, labels);
console.log('Relatório de Classificação:', report);

const matrix = MLMetrics.confusionMatrix(yTrue, yPred, labels);
console.log('Matriz de Confusão:', matrix);
```

## Deployment e Produção

### **MLOps para Web**:
```javascript
class MLModelManager {
    constructor() {
        this.models = new Map();
        this.metrics = new Map();
        this.versions = new Map();
    }
    
    async deployModel(modelName, modelUrl, version = '1.0.0') {
        try {
            const model = await tf.loadLayersModel(modelUrl);
            
            this.models.set(modelName, {
                model,
                version,
                deployedAt: new Date(),
                requestCount: 0,
                errorCount: 0
            });
            
            this.versions.set(modelName, version);
            
            console.log(`Modelo ${modelName} v${version} implantado com sucesso`);
            return true;
        } catch (error) {
            console.error('Erro ao implantar modelo:', error);
            return false;
        }
    }
    
    async predict(modelName, inputData) {
        const modelInfo = this.models.get(modelName);
        
        if (!modelInfo) {
            throw new Error(`Modelo ${modelName} não encontrado`);
        }
        
        const startTime = performance.now();
        
        try {
            modelInfo.requestCount++;
            
            const tensor = tf.tensor(inputData);
            const prediction = modelInfo.model.predict(tensor);
            const result = await prediction.data();
            
            // Cleanup
            tensor.dispose();
            prediction.dispose();
            
            const endTime = performance.now();
            const latency = endTime - startTime;
            
            this.recordMetrics(modelName, latency, true);
            
            return Array.from(result);
        } catch (error) {
            modelInfo.errorCount++;
            this.recordMetrics(modelName, 0, false);
            throw error;
        }
    }
    
    recordMetrics(modelName, latency, success) {
        if (!this.metrics.has(modelName)) {
            this.metrics.set(modelName, {
                totalRequests: 0,
                totalErrors: 0,
                totalLatency: 0,
                avgLatency: 0,
                uptime: Date.now()
            });
        }
        
        const metrics = this.metrics.get(modelName);
        metrics.totalRequests++;
        
        if (success) {
            metrics.totalLatency += latency;
            metrics.avgLatency = metrics.totalLatency / (metrics.totalRequests - metrics.totalErrors);
        } else {
            metrics.totalErrors++;
        }
    }
    
    getModelStats(modelName) {
        const modelInfo = this.models.get(modelName);
        const metrics = this.metrics.get(modelName);
        
        if (!modelInfo || !metrics) {
            return null;
        }
        
        return {
            version: modelInfo.version,
            deployedAt: modelInfo.deployedAt,
            totalRequests: metrics.totalRequests,
            errorRate: (metrics.totalErrors / metrics.totalRequests * 100).toFixed(2) + '%',
            avgLatency: metrics.avgLatency.toFixed(2) + 'ms',
            uptime: Math.floor((Date.now() - metrics.uptime) / 1000) + 's'
        };
    }
    
    async rollback(modelName, targetVersion) {
        const modelUrl = `/models/${modelName}/${targetVersion}/model.json`;
        return await this.deployModel(modelName, modelUrl, targetVersion);
    }
    
    healthCheck() {
        const report = {};
        
        for (const [modelName, modelInfo] of this.models) {
            const metrics = this.metrics.get(modelName);
            const errorRate = metrics.totalErrors / metrics.totalRequests;
            
            report[modelName] = {
                status: errorRate < 0.05 ? 'healthy' : 'degraded',
                errorRate: (errorRate * 100).toFixed(2) + '%',
                avgLatency: metrics.avgLatency.toFixed(2) + 'ms'
            };
        }
        
        return report;
    }
}

// Sistema de monitoramento de drift
class ModelDriftDetector {
    constructor(windowSize = 1000) {
        this.windowSize = windowSize;
        this.referenceData = [];
        this.productionData = [];
    }
    
    setReferenceData(data) {
        this.referenceData = data;
    }
    
    addProductionData(data) {
        this.productionData.push(data);
        
        if (this.productionData.length > this.windowSize) {
            this.productionData.shift();
        }
    }
    
    detectDrift() {
        if (this.referenceData.length === 0 || this.productionData.length < 100) {
            return { drift: false, reason: 'Dados insuficientes' };
        }
        
        // Test de Kolmogorov-Smirnov simplificado
        const refMean = this.calculateMean(this.referenceData);
        const refStd = this.calculateStd(this.referenceData, refMean);
        
        const prodMean = this.calculateMean(this.productionData);
        const prodStd = this.calculateStd(this.productionData, prodMean);
        
        const meanDiff = Math.abs(refMean - prodMean) / refStd;
        const stdDiff = Math.abs(refStd - prodStd) / refStd;
        
        const threshold = 2.0; // 2 desvios padrão
        
        if (meanDiff > threshold || stdDiff > 0.5) {
            return {
                drift: true,
                meanDrift: meanDiff,
                stdDrift: stdDiff,
                severity: meanDiff > 3.0 ? 'high' : 'medium'
            };
        }
        
        return { drift: false };
    }
    
    calculateMean(data) {
        return data.reduce((sum, val) => sum + val, 0) / data.length;
    }
    
    calculateStd(data, mean) {
        const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
        return Math.sqrt(variance);
    }
}

// Exemplo de uso completo
async function exemploMLOps() {
    const manager = new MLModelManager();
    const driftDetector = new ModelDriftDetector();
    
    // Deploy de modelo
    await manager.deployModel('classifier', '/models/classifier/model.json', '1.0.0');
    
    // Simulação de requests
    for (let i = 0; i < 100; i++) {
        const input = [Math.random(), Math.random(), Math.random()];
        
        try {
            const prediction = await manager.predict('classifier', [input]);
            driftDetector.addProductionData(input[0]); // Monitorar primeira feature
        } catch (error) {
            console.error('Erro na predição:', error);
        }
    }
    
    // Verificar status
    console.log('Status do modelo:', manager.getModelStats('classifier'));
    console.log('Health check:', manager.healthCheck());
    console.log('Drift detection:', driftDetector.detectDrift());
}
```

## Conclusão

Machine Learning para web representa uma convergência transformadora entre inteligência artificial e tecnologias web, oferecendo:

### **Capacidades Fundamentais**:
- **Aprendizado Automatizado**: Sistemas que melhoram com experiência
- **Processamento Inteligente**: Análise avançada de dados em tempo real
- **Personalização**: Experiências adaptadas ao usuário
- **Automação Inteligente**: Decisões automatizadas baseadas em dados

### **Impacto Tecnológico**:
- **Frontend Inteligente**: Interfaces que se adaptam ao comportamento do usuário
- **Backend Inteligente**: APIs que otimizam performance automaticamente
- **Experiência do Usuário**: Interações mais naturais e eficientes
- **Tomada de Decisão**: Insights baseados em análise de grandes volumes de dados

O domínio de Machine Learning para web é essencial para desenvolvedores modernos que buscam criar aplicações verdadeiramente inteligentes e competitivas no mercado digital atual.
