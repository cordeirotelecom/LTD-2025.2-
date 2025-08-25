# 💻 Visual Studio Code para Desenvolvimento com IA

## Por que VS Code é essencial para IA?
O VS Code é o editor mais popular para desenvolvimento com IA, oferecendo extensões poderosas, integração com GitHub Copilot, terminais integrados e suporte nativo para múltiplas linguagens.

## 🔧 Extensões Essenciais para IA

### Assistentes de IA
- **GitHub Copilot**: Autocompletar código com IA
- **CodeGPT**: Integração com ChatGPT, Claude, Gemini
- **Tabnine**: Autocompletar inteligente
- **IntelliCode**: IA da Microsoft para sugestões

### Python & Machine Learning
- **Python**: Suporte oficial Python
- **Jupyter**: Notebooks integrados
- **Pylance**: IntelliSense avançado
- **Python Docstring Generator**: Documentação automática

### Web Development
- **Live Server**: Servidor local instantâneo
- **Auto Rename Tag**: Renomear tags HTML automaticamente
- **Prettier**: Formatação automática de código
- **ES7+ React/Redux/React-Native snippets**: Snippets para React

### Git & GitHub
- **GitLens**: Git supercharged
- **GitHub Pull Requests**: Gerenciar PRs no VS Code
- **Git Graph**: Visualizar histórico do Git

## 🛠️ Configuração Perfeita para IA

### settings.json
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "jupyter.askForKernelRestart": false,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "github.copilot.enable": {
    "*": true,
    "yaml": false,
    "plaintext": false
  },
  "editor.inlineSuggest.enabled": true,
  "editor.suggest.preview": true
}
```

### Snippets Personalizados
```json
// python.json
{
  "TensorFlow Import": {
    "prefix": "tf",
    "body": [
      "import tensorflow as tf",
      "import numpy as np",
      "import matplotlib.pyplot as plt"
    ]
  },
  "Flask API": {
    "prefix": "flaskapi",
    "body": [
      "from flask import Flask, request, jsonify",
      "",
      "app = Flask(__name__)",
      "",
      "@app.route('/api/predict', methods=['POST'])",
      "def predict():",
      "    data = request.json",
      "    # Implementar predição",
      "    return jsonify({'result': 'success'})",
      "",
      "if __name__ == '__main__':",
      "    app.run(debug=True)"
    ]
  }
}
```

## 🚀 Fluxo de Trabalho com IA

### 1. Setup do Projeto
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install tensorflow numpy pandas matplotlib flask
```

### 2. Usar GitHub Copilot
- Digite comentários descrevendo o que quer fazer
- Aceite sugestões com Tab
- Use Ctrl+Enter para ver múltiplas opções

### 3. Notebooks Integrados
- Crie arquivos .ipynb diretamente no VS Code
- Execute células com Shift+Enter
- Visualize gráficos inline

### 4. Debugging com IA
- Use breakpoints visuais
- Inspecione variáveis em tempo real
- Debug tanto código Python quanto JavaScript

## 🎯 Shortcuts Essenciais
- `Ctrl+Shift+P`: Command Palette
- `Ctrl+` ` : Terminal integrado
- `Ctrl+Shift+\` : Split editor
- `F5`: Start debugging
- `Ctrl+/`: Comentar linha
- `Alt+Shift+F`: Formatar documento

## 🔍 Debugging de Modelos de IA

### launch.json para Python
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: ML Training",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/train_model.py",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      },
      "console": "integratedTerminal"
    }
  ]
}
```

## 📊 Integração com Jupyter
- Instale extensão Jupyter
- Conecte com kernels remotos
- Visualize dados com plotly/matplotlib
- Export notebooks para Python/HTML

Veja exemplos práticos na pasta `exemplos/vscode/`
