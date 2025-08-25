# 🔧 Ferramentas de Desenvolvimento Moderno

## Introdução

Este guia apresenta as ferramentas essenciais para desenvolvimento web moderno, focando na integração entre editores de código, controle de versão e inteligência artificial.

## 📝 Visual Studio Code (VS Code)

### Por que VS Code?
- **Gratuito e Open Source**
- **Extensibilidade** com milhares de extensões
- **Integração Git** nativa
- **IntelliSense** avançado
- **Terminal integrado**
- **Suporte multi-linguagem**

### Configuração Essencial

#### 1. Extensões Fundamentais

```json
{
  "recommendations": [
    "ms-vscode.vscode-github-copilot",
    "github.vscode-github-actions",
    "eamodio.gitlens",
    "bradlc.vscode-tailwindcss",
    "ms-vscode.vscode-json",
    "ms-python.python",
    "ms-vscode.vscode-typescript-next",
    "ritwickdey.liveserver",
    "formulahendry.auto-rename-tag",
    "christian-kohler.path-intellisense"
  ]
}
```

#### 2. Settings.json Otimizado

```json
{
  "editor.fontSize": 14,
  "editor.fontFamily": "'Fira Code', 'Cascadia Code', Consolas, monospace",
  "editor.fontLigatures": true,
  "editor.lineHeight": 1.5,
  "editor.tabSize": 2,
  "editor.insertSpaces": true,
  "editor.wordWrap": "on",
  "editor.minimap.enabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll": true,
    "source.organizeImports": true
  },
  
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000,
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  
  "git.enableSmartCommit": true,
  "git.confirmSync": false,
  "git.autofetch": true,
  
  "workbench.colorTheme": "GitHub Dark Default",
  "workbench.iconTheme": "material-icon-theme",
  "workbench.startupEditor": "welcomePage",
  
  "terminal.integrated.defaultProfile.windows": "PowerShell",
  "terminal.integrated.fontSize": 12,
  
  "emmet.includeLanguages": {
    "javascript": "javascriptreact",
    "typescript": "typescriptreact"
  },
  
  "prettier.semi": true,
  "prettier.singleQuote": true,
  "prettier.tabWidth": 2,
  
  "eslint.format.enable": true,
  "eslint.workingDirectories": ["./"],
  
  "github.copilot.enable": {
    "*": true,
    "yaml": false,
    "plaintext": false,
    "markdown": true
  }
}
```

#### 3. Keybindings Personalizados

```json
[
  {
    "key": "ctrl+shift+p",
    "command": "workbench.action.showCommands"
  },
  {
    "key": "ctrl+shift+`",
    "command": "workbench.action.terminal.new"
  },
  {
    "key": "ctrl+shift+e",
    "command": "workbench.view.explorer"
  },
  {
    "key": "ctrl+shift+g",
    "command": "workbench.view.scm"
  },
  {
    "key": "ctrl+shift+d",
    "command": "workbench.view.debug"
  },
  {
    "key": "ctrl+k ctrl+s",
    "command": "workbench.action.openGlobalKeybindings"
  },
  {
    "key": "alt+z",
    "command": "editor.action.toggleWordWrap"
  },
  {
    "key": "ctrl+shift+a",
    "command": "editor.action.blockComment"
  }
]
```

### Extensões por Categoria

#### Git/GitHub
```bash
# GitLens - Superpoderes para Git
Name: GitLens — Git supercharged
Id: eamodio.gitlens
Description: Visualização avançada de Git

# GitHub Integration
Name: GitHub Pull Requests and Issues
Id: github.vscode-pull-request-github
Description: Gerenciar PRs e Issues direto no VS Code

# GitHub Actions
Name: GitHub Actions
Id: github.vscode-github-actions
Description: Sintaxe e validação para workflows
```

#### Inteligência Artificial
```bash
# GitHub Copilot
Name: GitHub Copilot
Id: github.copilot
Description: Par de programação com IA

# GitHub Copilot Chat
Name: GitHub Copilot Chat
Id: github.copilot-chat
Description: Chat integrado com IA

# Tabnine
Name: Tabnine AI Autocomplete
Id: tabnine.tabnine-vscode
Description: Autocompletar inteligente
```

#### Desenvolvimento Web
```bash
# Live Server
Name: Live Server
Id: ritwickdey.liveserver
Description: Servidor local com live reload

# Auto Rename Tag
Name: Auto Rename Tag
Id: formulahendry.auto-rename-tag
Description: Renomear tags HTML automaticamente

# Bracket Pair Colorizer
Name: Bracket Pair Colorizer 2
Id: coenraads.bracket-pair-colorizer-2
Description: Colorir parênteses e chaves

# Path Intellisense
Name: Path Intellisense
Id: christian-kohler.path-intellisense
Description: Autocompletar caminhos de arquivos
```

## 🌊 Git - Controle de Versão

### Configuração Global

```bash
# Identidade
git config --global user.name "Seu Nome"
git config --global user.email "seu.email@gmail.com"

# Editor padrão
git config --global core.editor "code --wait"

# Merge tool
git config --global merge.tool vscode
git config --global mergetool.vscode.cmd 'code --wait $MERGED'

# Aliases úteis
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'
git config --global alias.lg "log --oneline --decorate --graph --all"
git config --global alias.adog "log --all --decorate --oneline --graph"

# Configurações de push
git config --global push.default simple
git config --global pull.rebase false

# Configurações de linha
git config --global core.autocrlf true  # Windows
git config --global core.autocrlf input # Mac/Linux

# Melhorias visuais
git config --global color.ui auto
git config --global color.branch auto
git config --global color.diff auto
git config --global color.status auto
```

### .gitconfig Completo

```ini
[user]
    name = Seu Nome
    email = seu.email@gmail.com

[core]
    editor = code --wait
    autocrlf = true
    excludesfile = ~/.gitignore_global

[push]
    default = simple

[pull]
    rebase = false

[merge]
    tool = vscode

[mergetool "vscode"]
    cmd = code --wait $MERGED

[color]
    ui = auto
    branch = auto
    diff = auto
    status = auto

[alias]
    st = status
    co = checkout
    br = branch
    ci = commit
    unstage = reset HEAD --
    last = log -1 HEAD
    visual = !gitk
    lg = log --oneline --decorate --graph --all
    adog = log --all --decorate --oneline --graph
    amend = commit --amend --no-edit
    force = push --force-with-lease
    cleanup = "!git branch --merged | grep -v '\\*\\|master\\|main\\|develop' | xargs -n 1 git branch -d"

[diff]
    tool = vscode

[difftool "vscode"]
    cmd = code --wait --diff $LOCAL $REMOTE

[init]
    defaultBranch = main
```

### .gitignore Global

```gitignore
# Sistema Operacional
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Editores
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Dependências
node_modules/
bower_components/

# Build
dist/
build/
*.tgz
*.tar.gz

# Ambiente
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Cache
.cache/
.parcel-cache/
.next/
.nuxt/

# Temporários
*.tmp
*.temp
.tmp/
temp/
```

## 🚀 GitHub - Colaboração

### GitHub CLI Setup

```bash
# Instalar GitHub CLI
winget install GitHub.cli  # Windows
brew install gh           # macOS
sudo apt install gh       # Ubuntu

# Autenticar
gh auth login

# Configurar
gh config set git_protocol https
gh config set editor "code --wait"
gh config set pager cat
```

### Comandos Essenciais

```bash
# Repositório
gh repo create meu-projeto --public --clone
gh repo view --web
gh repo fork usuario/repo
gh repo clone usuario/repo

# Issues
gh issue create --title "Bug no sistema" --body "Descrição detalhada"
gh issue list --state open
gh issue view 123
gh issue close 123

# Pull Requests
gh pr create --title "feat: nova funcionalidade" --body "Descrição"
gh pr list --state open
gh pr view 456
gh pr checkout 456
gh pr merge 456 --merge
gh pr review 456 --approve

# Releases
gh release create v1.0.0 --title "Primeira versão" --notes "Funcionalidades iniciais"
gh release list
gh release download v1.0.0

# Gists
gh gist create arquivo.js --public
gh gist list
gh gist view abc123

# Actions
gh workflow list
gh workflow run deploy.yml
gh run list
gh run view 123456
```

### Templates de Issue

```markdown
<!-- .github/ISSUE_TEMPLATE/bug_report.md -->
---
name: Bug Report
about: Reportar um bug
title: '[BUG] '
labels: bug
assignees: ''
---

## 🐛 Descrição do Bug
Uma descrição clara e concisa do bug.

## 🔄 Como Reproduzir
Passos para reproduzir o comportamento:
1. Vá para '...'
2. Clique em '....'
3. Role para baixo até '....'
4. Veja o erro

## ✅ Comportamento Esperado
Uma descrição clara do que você esperava que acontecesse.

## 📱 Screenshots
Se aplicável, adicione screenshots para explicar o problema.

## 🖥️ Ambiente
- OS: [ex: iOS]
- Browser: [ex: chrome, safari]
- Version: [ex: 22]

## 📝 Contexto Adicional
Adicione qualquer contexto adicional sobre o problema aqui.
```

### Template de Pull Request

```markdown
<!-- .github/pull_request_template.md -->
## 📝 Descrição
Breve descrição das mudanças realizadas.

## 🎯 Tipo de Mudança
- [ ] Bug fix (mudança que corrige um problema)
- [ ] Nova funcionalidade (mudança que adiciona funcionalidade)
- [ ] Breaking change (mudança que pode quebrar funcionalidades existentes)
- [ ] Documentação (mudança apenas na documentação)

## 🧪 Como Foi Testado?
Descreva os testes que você executou para verificar suas mudanças.

## ✅ Checklist
- [ ] Meu código segue o style guide do projeto
- [ ] Eu fiz uma auto-revisão do meu código
- [ ] Eu comentei meu código, especialmente em áreas difíceis de entender
- [ ] Eu fiz as mudanças correspondentes na documentação
- [ ] Minhas mudanças não geram novos warnings
- [ ] Eu adicionei testes que provam que minha correção é efetiva ou que minha funcionalidade funciona
- [ ] Testes unitários novos e existentes passam localmente com minhas mudanças

## 📸 Screenshots (se aplicável)
Adicione screenshots para demonstrar as mudanças visuais.

## 🔗 Issue Relacionada
Fixes #(número da issue)
```

## 🤖 Inteligência Artificial

### GitHub Copilot

#### Configuração Avançada
```json
{
  "github.copilot.enable": {
    "*": true,
    "yaml": false,
    "plaintext": false,
    "markdown": true
  },
  "github.copilot.inlineSuggest.enable": true,
  "github.copilot.editor.enableAutoCompletions": true,
  "editor.inlineSuggest.enabled": true,
  "editor.suggestSelection": "first"
}
```

#### Prompts Efetivos

```javascript
// Exemplo de prompt para função
// Função para validar email com regex
function validateEmail(email) {
  // Copilot vai gerar automaticamente
}

// Comentário descritivo gera melhor código
// Criar um sistema de cache em memória com TTL (time to live)
// que expira entradas após um tempo determinado
class MemoryCache {
  // Copilot irá implementar toda a classe
}

// Prompt para algoritmo específico
// Implementar busca binária recursiva em array ordenado
function binarySearch(arr, target, left = 0, right = arr.length - 1) {
  // Implementação automática
}
```

#### Técnicas Avançadas

```javascript
// 1. Context-aware prompting
// Considerando o contexto da aplicação de e-commerce, 
// criar função para calcular desconto progressivo baseado na quantidade
function calculateProgressiveDiscount(quantity, basePrice) {
  // Copilot entende o contexto e sugere lógica de e-commerce
}

// 2. Pattern-based prompting
// Seguindo o padrão Repository, criar classe para gerenciar usuários
class UserRepository {
  // Copilot reconhece o padrão e implementa métodos CRUD
}

// 3. Framework-specific prompting
// React hook personalizado para gerenciar estado de formulário
function useFormState(initialState) {
  // Copilot gera hook seguindo padrões React
}
```

### Outras Ferramentas de IA

#### ChatGPT/Claude Integration
```javascript
// Extensão para VS Code que integra IA
// Instalar: Code GPT, Tabnine, Codeium

// Exemplo de uso com Code GPT
// 1. Selecionar código
// 2. Ctrl+Shift+P -> "Ask CodeGPT"
// 3. Fazer pergunta específica
```

#### Ferramentas de Terminal
```bash
# GitHub Copilot CLI
npm install -g @githubnext/github-copilot-cli

# Aliases para facilitar uso
echo 'eval "$(github-copilot-cli alias -- "$0")"' >> ~/.bashrc

# Uso
?? "como listar arquivos por tamanho"
git? "como desfazer último commit"
gh? "como criar release"
```

## 🔧 Workflow Integrado

### Setup de Projeto Completo

```bash
# 1. Criar projeto
mkdir meu-projeto
cd meu-projeto

# 2. Inicializar Git
git init
git branch -M main

# 3. Configurar VS Code
code .

# 4. Estrutura inicial
touch README.md .gitignore
mkdir src docs tests

# 5. Configurar package.json
npm init -y

# 6. Instalar dependências de desenvolvimento
npm install -D prettier eslint husky lint-staged

# 7. Configurar Husky (Git hooks)
npx husky install
npx husky add .husky/pre-commit "lint-staged"

# 8. Criar repositório GitHub
gh repo create --source=. --public --push
```

### Configuração de Qualidade

#### .prettierrc
```json
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false
}
```

#### .eslintrc.json
```json
{
  "env": {
    "browser": true,
    "es2021": true,
    "node": true
  },
  "extends": [
    "eslint:recommended"
  ],
  "parserOptions": {
    "ecmaVersion": 12,
    "sourceType": "module"
  },
  "rules": {
    "indent": ["error", 2],
    "linebreak-style": ["error", "unix"],
    "quotes": ["error", "single"],
    "semi": ["error", "always"]
  }
}
```

#### package.json scripts
```json
{
  "scripts": {
    "dev": "live-server src",
    "build": "npm run lint && npm run format",
    "lint": "eslint . --ext .js,.jsx,.ts,.tsx",
    "lint:fix": "eslint . --ext .js,.jsx,.ts,.tsx --fix",
    "format": "prettier --write .",
    "format:check": "prettier --check .",
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "lint-staged": {
    "*.{js,jsx,ts,tsx}": [
      "eslint --fix",
      "prettier --write",
      "git add"
    ],
    "*.{json,css,md}": [
      "prettier --write",
      "git add"
    ]
  }
}
```

## 📊 Métricas e Monitoramento

### VS Code Telemetry
```json
{
  "telemetry.telemetryLevel": "error",
  "githubPullRequests.telemetry.enabled": false,
  "github.copilot.advanced": {
    "length": 500,
    "temperature": "",
    "top_p": "",
    "stops": {
      "*": ["\n\n\n"]
    }
  }
}
```

### Git Estatísticas
```bash
# Estatísticas do repositório
git log --stat
git log --oneline --graph --decorate --all
git shortlog -sn --all
git log --since="2 weeks ago" --pretty=format:"%h %an %s" --graph

# Análise de contribuições
git log --pretty=format:"%an" | sort | uniq -c | sort -nr

# Linhas de código por autor
git log --pretty=format:"%an" --numstat | awk '
/^[0-9]+/ { add += $1; subs += $2; loc += $1 - $2 }
/^Author/ { author = $0 }
END { printf "added lines: %s removed lines: %s total lines: %s\n", add, subs, loc }'
```

## 🎯 Dicas de Produtividade

### Snippets Personalizados

```json
// snippets.json no VS Code
{
  "React Functional Component": {
    "prefix": "rfc",
    "body": [
      "import React from 'react';",
      "",
      "const ${1:ComponentName} = () => {",
      "  return (",
      "    <div>",
      "      $0",
      "    </div>",
      "  );",
      "};",
      "",
      "export default ${1:ComponentName};"
    ],
    "description": "Create React Functional Component"
  },
  "Console Log": {
    "prefix": "clg",
    "body": [
      "console.log('$1:', $1);$0"
    ],
    "description": "Console log with label"
  },
  "Try Catch Block": {
    "prefix": "tryc",
    "body": [
      "try {",
      "  $1",
      "} catch (error) {",
      "  console.error('Error:', error);",
      "  $0",
      "}"
    ],
    "description": "Try catch block"
  }
}
```

### Macros e Automações

```json
// tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Sync with main",
      "type": "shell",
      "command": "git",
      "args": ["pull", "origin", "main"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Deploy to GitHub Pages",
      "type": "shell",
      "command": "npm",
      "args": ["run", "deploy"],
      "group": "build",
      "dependsOn": "Sync with main"
    }
  ]
}
```

## 🔮 Tendências Futuras

### IA Coding Assistants
- **GitHub Copilot X**: Chat, CLI, Docs
- **Amazon CodeWhisperer**: Alternativa ao Copilot
- **Tabnine**: IA local e privada
- **Codeium**: Gratuito e open source

### Next-Gen Tools
- **GitHub Codespaces**: Desenvolvimento na nuvem
- **VS Code for Web**: Editor no navegador
- **Dev Containers**: Ambientes isolados
- **Remote Development**: Desenvolvimento remoto

### Emerging Technologies
- **WebAssembly**: Performance nativa no web
- **Edge Computing**: Deploy distribuído
- **Serverless**: Arquitetura sem servidor
- **JAMstack**: JavaScript, APIs, Markup

---

## 🎉 Conclusão

Este guia cobriu as ferramentas essenciais para desenvolvimento moderno. A combinação de VS Code + Git + GitHub + IA cria um ambiente de desenvolvimento poderoso e produtivo.

**Próximos passos:**
1. Configure seu ambiente seguindo este guia
2. Pratique com projetos reais
3. Explore extensões avançadas
4. Mantenha-se atualizado com novas ferramentas

**Recursos adicionais:**
- [VS Code Documentation](https://code.visualstudio.com/docs)
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [GitHub Copilot Documentation](https://docs.github.com/copilot)
