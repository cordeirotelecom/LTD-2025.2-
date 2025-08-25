# 🐙 GitHub para Projetos de IA: Guia Completo

## Por que GitHub é crucial para IA?
GitHub não é apenas um repositório de código - é uma plataforma completa para colaboração, versionamento, deploy automático e showcase de projetos de IA.

## 🚀 GitHub Copilot: Seu Par de Programação IA

### O que é o GitHub Copilot?
Assistente de IA que sugere código em tempo real, treinado em bilhões de linhas de código público.

### Configuração
```bash
# Instalar extensão no VS Code
# Fazer login com conta GitHub Pro/Team/Enterprise
```

### Prompts Eficazes
```python
# ✅ BOM: Seja específico
# Criar modelo de classificação de imagens com CNN usando TensorFlow
def create_cnn_model(input_shape, num_classes):

# ✅ BOM: Use comentários descritivos
# Função para preprocessar imagens: redimensionar para 224x224, normalizar pixels
def preprocess_image(image_path):

# ❌ RUIM: Muito vago
# fazer modelo
```

## 📁 Estrutura de Repositório para Projetos de IA

```
projeto-ia/
├── README.md                 # Documentação principal
├── requirements.txt          # Dependências Python
├── environment.yml           # Ambiente Conda
├── .gitignore               # Arquivos a ignorar
├── LICENSE                  # Licença do projeto
├── data/                    # Dados (usar Git LFS)
│   ├── raw/                 # Dados brutos
│   ├── processed/           # Dados processados
│   └── external/            # Dados externos
├── models/                  # Modelos treinados
│   ├── trained/             # Modelos finais
│   └── checkpoints/         # Checkpoints
├── notebooks/               # Jupyter notebooks
│   ├── 01-exploratory.ipynb
│   ├── 02-modeling.ipynb
│   └── 03-evaluation.ipynb
├── src/                     # Código fonte
│   ├── __init__.py
│   ├── data/                # Scripts de dados
│   ├── features/            # Feature engineering
│   ├── models/              # Modelos
│   └── visualization/       # Visualizações
├── tests/                   # Testes
├── docs/                    # Documentação
├── docker/                  # Containerização
└── scripts/                 # Scripts utilitários
```

## 🔧 GitHub Actions para IA

### CI/CD Pipeline
```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/
    
    - name: Lint code
      run: |
        flake8 src/
    
    - name: Train model
      run: |
        python src/train_model.py
    
    - name: Evaluate model
      run: |
        python src/evaluate_model.py
```

### Deploy Automático
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Heroku
      uses: akhileshns/heroku-deploy@v3.12.12
      with:
        heroku_api_key: ${{secrets.HEROKU_API_KEY}}
        heroku_app_name: "meu-app-ia"
        heroku_email: "email@example.com"
```

## 📊 Git LFS para Dados Grandes

```bash
# Instalar Git LFS
git lfs install

# Rastrear arquivos grandes
git lfs track "*.csv"
git lfs track "*.h5"
git lfs track "*.pkl"
git lfs track "data/**"

# Committar .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## 🏷️ Versionamento de Modelos

### Tags semânticas
```bash
# Versionar releases
git tag -a v1.0.0 -m "Primeiro modelo em produção"
git tag -a v1.1.0 -m "Modelo com melhor acurácia"
git push origin --tags
```

### Branch strategy
```bash
# Feature branches
git checkout -b feature/novo-algoritmo

# Development branch
git checkout -b develop

# Hotfix branches
git checkout -b hotfix/corrigir-preprocessamento
```

## 📝 README.md Perfeito para IA

```markdown
# 🤖 Nome do Projeto

## 🎯 Objetivo
Descrição clara do problema que resolve.

## 📊 Dataset
- **Fonte**: Onde obteve os dados
- **Tamanho**: Número de amostras
- **Features**: Principais características

## 🧠 Modelo
- **Algoritmo**: RandomForest/CNN/Transformer
- **Acurácia**: 95.2%
- **Métricas**: Precision, Recall, F1-score

## 🚀 Como usar
\`\`\`bash
pip install -r requirements.txt
python src/predict.py --input data/test.csv
\`\`\`

## 📈 Resultados
- Baseline: 85%
- Nosso modelo: 95.2%
- Melhoria: +10.2%

## 🔧 Tecnologias
- Python 3.9
- TensorFlow 2.8
- Pandas, NumPy
- Scikit-learn

## 📄 Licença
MIT License
```

## 🌟 GitHub Pages para Portfolio

### Ativar GitHub Pages
1. Settings > Pages
2. Source: Deploy from a branch
3. Branch: main / docs folder

### Portfolio HTML simples
```html
<!DOCTYPE html>
<html>
<head>
    <title>Meu Portfolio de IA</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-center mb-8">Portfolio de IA</h1>
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-2">Chatbot Inteligente</h2>
                <p class="text-gray-600 mb-4">Sistema de atendimento com NLP</p>
                <a href="projeto1/" class="bg-blue-500 text-white px-4 py-2 rounded">Ver Demo</a>
            </div>
        </div>
    </div>
</body>
</html>
```

## 🔍 Issues e Project Management

### Templates de Issues
```markdown
<!-- .github/ISSUE_TEMPLATE/bug_report.md -->
---
name: Bug Report
about: Reportar um problema
---

**Descrição do bug**
Descrição clara do problema.

**Para reproduzir**
1. Execute o comando '...'
2. Veja erro '...'

**Comportamento esperado**
O que deveria acontecer.

**Ambiente**
- OS: [Windows/Mac/Linux]
- Python: [versão]
- TensorFlow: [versão]
```

### GitHub Projects
- Criar boards Kanban
- Automatizar movimentação de cards
- Integrar com Issues e PRs

Veja exemplos na pasta `exemplos/github/`
