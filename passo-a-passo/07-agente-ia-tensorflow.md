# 🧠 Passo a Passo: Criando um Agente de IA com TensorFlow.js

## Objetivo
Criar um agente inteligente que aprende a jogar um jogo simples usando reinforcement learning no navegador.

## Pré-requisitos
- Conhecimento básico de JavaScript
- Conceitos de machine learning
- Navegador moderno

## Tecnologias
- TensorFlow.js
- HTML5 Canvas
- JavaScript ES6+

## Implementação

### 1. Estrutura HTML
```html
<!DOCTYPE html>
<html>
<head>
  <title>Agente IA - Jogo Snake</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
  <style>
    canvas { border: 2px solid #333; }
    .controls { margin: 20px 0; }
    .stats { margin: 10px 0; font-family: monospace; }
  </style>
</head>
<body>
  <h1>🐍 Agente IA aprendendo Snake</h1>
  <canvas id="gameCanvas" width="400" height="400"></canvas>
  <div class="controls">
    <button onclick="startTraining()">Iniciar Treinamento</button>
    <button onclick="stopTraining()">Parar</button>
    <button onclick="resetAgent()">Reset</button>
  </div>
  <div class="stats">
    <div>Geração: <span id="generation">0</span></div>
    <div>Melhor Score: <span id="bestScore">0</span></div>
    <div>Score Atual: <span id="currentScore">0</span></div>
    <div>Epsilon: <span id="epsilon">1.0</span></div>
  </div>
</body>
</html>
```

### 2. Configuração do Ambiente
```javascript
class SnakeGame {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.gridSize = 20;
    this.tileCount = canvas.width / this.gridSize;
    this.reset();
  }
  
  reset() {
    this.snake = [{x: 10, y: 10}];
    this.food = this.generateFood();
    this.dx = 0;
    this.dy = 0;
    this.score = 0;
    this.gameOver = false;
  }
  
  generateFood() {
    return {
      x: Math.floor(Math.random() * this.tileCount),
      y: Math.floor(Math.random() * this.tileCount)
    };
  }
  
  getState() {
    const head = this.snake[0];
    const food = this.food;
    
    // Estado simplificado: posição relativa da comida e obstáculos
    return [
      // Direção para a comida
      food.x > head.x ? 1 : 0,  // comida à direita
      food.x < head.x ? 1 : 0,  // comida à esquerda
      food.y > head.y ? 1 : 0,  // comida abaixo
      food.y < head.y ? 1 : 0,  // comida acima
      
      // Detecção de obstáculos
      this.willCollide(head.x - 1, head.y) ? 1 : 0,  // parede/corpo à esquerda
      this.willCollide(head.x + 1, head.y) ? 1 : 0,  // parede/corpo à direita
      this.willCollide(head.x, head.y - 1) ? 1 : 0,  // parede/corpo acima
      this.willCollide(head.x, head.y + 1) ? 1 : 0,  // parede/corpo abaixo
    ];
  }
  
  willCollide(x, y) {
    // Verifica colisão com paredes
    if (x < 0 || x >= this.tileCount || y < 0 || y >= this.tileCount) {
      return true;
    }
    
    // Verifica colisão com o próprio corpo
    return this.snake.some(segment => segment.x === x && segment.y === y);
  }
  
  step(action) {
    // Ações: 0=esquerda, 1=reto, 2=direita (relativo à direção atual)
    const directions = [
      {dx: -1, dy: 0}, // esquerda
      {dx: 1, dy: 0},  // direita
      {dx: 0, dy: -1}, // cima
      {dx: 0, dy: 1}   // baixo
    ];
    
    let currentDir = directions.findIndex(d => d.dx === this.dx && d.dy === this.dy);
    if (currentDir === -1) currentDir = 1; // direita como padrão
    
    // Calcular nova direção baseada na ação
    if (action === 0) currentDir = (currentDir + 3) % 4; // vira à esquerda
    else if (action === 2) currentDir = (currentDir + 1) % 4; // vira à direita
    
    this.dx = directions[currentDir].dx;
    this.dy = directions[currentDir].dy;
    
    // Mover snake
    const head = {x: this.snake[0].x + this.dx, y: this.snake[0].y + this.dy};
    
    // Verificar colisões
    if (this.willCollide(head.x, head.y)) {
      this.gameOver = true;
      return {reward: -10, done: true}; // Penalidade por morrer
    }
    
    this.snake.unshift(head);
    
    // Verificar se comeu a comida
    if (head.x === this.food.x && head.y === this.food.y) {
      this.score += 10;
      this.food = this.generateFood();
      return {reward: 10, done: false}; // Recompensa por comer
    } else {
      this.snake.pop();
      return {reward: -0.1, done: false}; // Pequena penalidade por tempo
    }
  }
  
  render() {
    // Limpar canvas
    this.ctx.fillStyle = 'black';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Desenhar snake
    this.ctx.fillStyle = 'lime';
    this.snake.forEach(segment => {
      this.ctx.fillRect(segment.x * this.gridSize, segment.y * this.gridSize, 
                       this.gridSize - 2, this.gridSize - 2);
    });
    
    // Desenhar comida
    this.ctx.fillStyle = 'red';
    this.ctx.fillRect(this.food.x * this.gridSize, this.food.y * this.gridSize, 
                     this.gridSize - 2, this.gridSize - 2);
  }
}
```

### 3. Rede Neural com TensorFlow.js
```javascript
class DQNAgent {
  constructor(stateSize = 8, actionSize = 3) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.memory = [];
    this.maxMemory = 2000;
    this.epsilon = 1.0;
    this.epsilonDecay = 0.995;
    this.epsilonMin = 0.01;
    this.learningRate = 0.001;
    this.gamma = 0.95;
    this.batchSize = 32;
    
    this.model = this.buildModel();
  }
  
  buildModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.dense({
      units: 24,
      activation: 'relu',
      inputShape: [this.stateSize]
    }));
    
    model.add(tf.layers.dense({
      units: 24,
      activation: 'relu'
    }));
    
    model.add(tf.layers.dense({
      units: this.actionSize,
      activation: 'linear'
    }));
    
    model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: 'meanSquaredError'
    });
    
    return model;
  }
  
  remember(state, action, reward, nextState, done) {
    this.memory.push({state, action, reward, nextState, done});
    if (this.memory.length > this.maxMemory) {
      this.memory.shift();
    }
  }
  
  act(state) {
    if (Math.random() <= this.epsilon) {
      return Math.floor(Math.random() * this.actionSize); // Exploração
    }
    
    const prediction = this.model.predict(tf.tensor2d([state]));
    const action = prediction.argMax(1).dataSync()[0];
    prediction.dispose();
    return action;
  }
  
  async replay() {
    if (this.memory.length < this.batchSize) return;
    
    // Amostra aleatória da memória
    const batch = this.memory.sort(() => 0.5 - Math.random()).slice(0, this.batchSize);
    
    const states = batch.map(e => e.state);
    const nextStates = batch.map(e => e.nextState);
    
    const currentQs = this.model.predict(tf.tensor2d(states));
    const nextQs = this.model.predict(tf.tensor2d(nextStates));
    
    const currentQsData = await currentQs.data();
    const nextQsData = await nextQs.data();
    
    const xs = [];
    const ys = [];
    
    batch.forEach((experience, index) => {
      const target = currentQsData.slice(index * this.actionSize, (index + 1) * this.actionSize);
      
      if (experience.done) {
        target[experience.action] = experience.reward;
      } else {
        const maxNextQ = Math.max(...nextQsData.slice(index * this.actionSize, (index + 1) * this.actionSize));
        target[experience.action] = experience.reward + this.gamma * maxNextQ;
      }
      
      xs.push(experience.state);
      ys.push(target);
    });
    
    await this.model.fit(tf.tensor2d(xs), tf.tensor2d(ys), {
      epochs: 1,
      verbose: 0
    });
    
    currentQs.dispose();
    nextQs.dispose();
    
    if (this.epsilon > this.epsilonMin) {
      this.epsilon *= this.epsilonDecay;
    }
  }
}
```

### 4. Loop de Treinamento
```javascript
let game, agent, training = false;
let generation = 0, bestScore = 0;

function initializeAgent() {
  const canvas = document.getElementById('gameCanvas');
  game = new SnakeGame(canvas);
  agent = new DQNAgent();
}

async function trainEpisode() {
  game.reset();
  let state = game.getState();
  let totalReward = 0;
  let steps = 0;
  const maxSteps = 1000;
  
  while (!game.gameOver && steps < maxSteps) {
    const action = agent.act(state);
    const result = game.step(action);
    const nextState = game.getState();
    
    agent.remember(state, action, result.reward, nextState, result.done);
    state = nextState;
    totalReward += result.reward;
    steps++;
    
    // Renderizar ocasionalmente
    if (steps % 10 === 0) {
      game.render();
      updateStats();
      await new Promise(resolve => setTimeout(resolve, 1));
    }
  }
  
  // Treinar o agente
  await agent.replay();
  
  generation++;
  if (game.score > bestScore) {
    bestScore = game.score;
  }
  
  updateStats();
  return totalReward;
}

async function startTraining() {
  if (!agent) initializeAgent();
  training = true;
  
  while (training) {
    await trainEpisode();
    await new Promise(resolve => setTimeout(resolve, 10));
  }
}

function stopTraining() {
  training = false;
}

function resetAgent() {
  initializeAgent();
  generation = 0;
  bestScore = 0;
  updateStats();
}

function updateStats() {
  document.getElementById('generation').textContent = generation;
  document.getElementById('bestScore').textContent = bestScore;
  document.getElementById('currentScore').textContent = game.score;
  document.getElementById('epsilon').textContent = agent.epsilon.toFixed(3);
}

// Inicializar quando a página carregar
window.onload = function() {
  initializeAgent();
  updateStats();
};
```

## Como Usar
1. Abra o arquivo HTML no navegador
2. Clique em "Iniciar Treinamento"
3. Observe o agente aprendendo a jogar
4. O epsilon diminui gradualmente (menos exploração, mais aproveitamento)

## Melhorias Possíveis
- Adicionar mais features ao estado (distância das paredes, tamanho da cobra)
- Implementar Double DQN ou Dueling DQN
- Usar redes convolucionais para processar toda a tela
- Adicionar experience replay prioritizado
- Salvar e carregar modelos treinados

Veja o exemplo completo na pasta `exemplos/`!
