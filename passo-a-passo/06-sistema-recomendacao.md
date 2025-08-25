# 🔧 Passo a Passo: Criando um Sistema de Recomendação Simples

## Objetivo
Criar um sistema que recomenda produtos baseado no histórico de compras do usuário.

## Tecnologias
- HTML, CSS, JavaScript (Frontend)
- Algoritmo de colaboração simples

## Implementação

### 1. Estrutura de Dados
```javascript
// Base de dados simulada
const usuarios = {
  'usuario1': ['produto1', 'produto2', 'produto3'],
  'usuario2': ['produto2', 'produto4', 'produto5'],
  'usuario3': ['produto1', 'produto4', 'produto6']
};

const produtos = [
  { id: 'produto1', nome: 'Smartphone', categoria: 'Eletrônicos' },
  { id: 'produto2', nome: 'Notebook', categoria: 'Eletrônicos' },
  { id: 'produto3', nome: 'Fone Bluetooth', categoria: 'Acessórios' },
  { id: 'produto4', nome: 'Mouse Gamer', categoria: 'Acessórios' },
  { id: 'produto5', nome: 'Teclado Mecânico', categoria: 'Acessórios' },
  { id: 'produto6', nome: 'Monitor 4K', categoria: 'Eletrônicos' }
];
```

### 2. Algoritmo de Recomendação
```javascript
function recomendar(usuarioAtual) {
  const comprasUsuario = usuarios[usuarioAtual] || [];
  const recomendacoes = new Set();
  
  // Encontrar usuários similares
  for (let outroUsuario in usuarios) {
    if (outroUsuario !== usuarioAtual) {
      const comprasOutro = usuarios[outroUsuario];
      const produtosComuns = comprasUsuario.filter(p => comprasOutro.includes(p));
      
      // Se há produtos em comum, recomendar outros produtos do usuário similar
      if (produtosComuns.length > 0) {
        comprasOutro.forEach(produto => {
          if (!comprasUsuario.includes(produto)) {
            recomendacoes.add(produto);
          }
        });
      }
    }
  }
  
  return Array.from(recomendacoes);
}
```

### 3. Interface HTML
```html
<!DOCTYPE html>
<html>
<head>
  <title>Sistema de Recomendação</title>
  <style>
    .produto { border: 1px solid #ccc; margin: 10px; padding: 10px; }
    .recomendacao { background: #e8f5e8; }
  </style>
</head>
<body>
  <h1>Sistema de Recomendação</h1>
  <select id="usuario" onchange="mostrarRecomendacoes()">
    <option value="">Selecione um usuário</option>
    <option value="usuario1">Usuário 1</option>
    <option value="usuario2">Usuário 2</option>
    <option value="usuario3">Usuário 3</option>
  </select>
  
  <div id="historico">
    <h3>Histórico de Compras</h3>
    <div id="compras"></div>
  </div>
  
  <div id="recomendacoes">
    <h3>Recomendações para Você</h3>
    <div id="sugestoes"></div>
  </div>
</body>
</html>
```

### 4. Lógica JavaScript Completa
```javascript
function mostrarRecomendacoes() {
  const usuarioSelecionado = document.getElementById('usuario').value;
  if (!usuarioSelecionado) return;
  
  // Mostrar histórico
  const compras = usuarios[usuarioSelecionado] || [];
  const comprasDiv = document.getElementById('compras');
  comprasDiv.innerHTML = '';
  compras.forEach(produtoId => {
    const produto = produtos.find(p => p.id === produtoId);
    comprasDiv.innerHTML += `<div class="produto">${produto.nome} - ${produto.categoria}</div>`;
  });
  
  // Mostrar recomendações
  const recomendacoes = recomendar(usuarioSelecionado);
  const sugestoesDiv = document.getElementById('sugestoes');
  sugestoesDiv.innerHTML = '';
  recomendacoes.forEach(produtoId => {
    const produto = produtos.find(p => p.id === produtoId);
    sugestoesDiv.innerHTML += `<div class="produto recomendacao">📌 ${produto.nome} - ${produto.categoria}</div>`;
  });
}
```

## Como Melhorar
- Adicionar pesos por categoria
- Implementar algoritmos mais sofisticados (colaborativo, baseado em conteúdo)
- Usar machine learning real com bibliotecas como TensorFlow.js
- Conectar com banco de dados real
- Implementar feedback do usuário

Veja o exemplo completo funcionando em `exemplos/`!
