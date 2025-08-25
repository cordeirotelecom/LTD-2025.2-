# 🔗 Guia Completo: GitHub + VS Code Passo a Passo

## 📋 Índice
1. [Configuração Inicial](#configuracao-inicial)
2. [Criando seu Primeiro Repositório](#primeiro-repositorio)
3. [Comandos Git Básicos no VS Code](#comandos-basicos)
4. [Fluxo de Trabalho Completo](#fluxo-trabalho)
5. [Trabalhando com Branches](#branches)
6. [Resolvendo Conflitos](#conflitos)
7. [GitHub Actions](#github-actions)
8. [Dicas Avançadas](#dicas-avancadas)

## 🚀 Configuração Inicial

### 1. Instalar Git
```bash
# Windows (usando winget)
winget install Git.Git

# Ou baixe de: https://git-scm.com/download/win
```

### 2. Configurar Git no Terminal
```bash
# Configurar seu nome
git config --global user.name "Seu Nome"

# Configurar seu email (use o mesmo do GitHub)
git config --global user.email "seuemail@gmail.com"

# Verificar configurações
git config --list
```

### 3. Configurar VS Code
Instale essas extensões essenciais:

#### GitLens - Git supercharged
```
Ctrl + Shift + P > Extensions: Install Extensions
Busque: "GitLens"
```

#### GitHub Pull Requests and Issues
```
Busque: "GitHub Pull Requests and Issues"
```

#### Git Graph
```
Busque: "Git Graph"
```

### 4. Conectar VS Code com GitHub

#### Método 1: GitHub CLI (Recomendado)
```bash
# Instalar GitHub CLI
winget install GitHub.cli

# Fazer login
gh auth login
# Selecione: GitHub.com > HTTPS > Yes > Login with a web browser
```

#### Método 2: Token Personal
1. GitHub.com > Settings > Developer settings > Personal access tokens
2. Generate new token (classic)
3. Selecione scopes: `repo`, `workflow`, `write:packages`
4. Copie o token gerado

## 📁 Criando seu Primeiro Repositório

### Método 1: VS Code + GitHub (Mais Fácil)

#### Passo 1: Criar Projeto Local
```bash
# Abrir terminal no VS Code (Ctrl + `)
mkdir meu-primeiro-projeto
cd meu-primeiro-projeto
```

#### Passo 2: Inicializar Git
```bash
# Inicializar repositório Git
git init

# Criar arquivo README
echo "# Meu Primeiro Projeto" > README.md
```

#### Passo 3: Fazer Primeiro Commit
```bash
# Adicionar arquivos
git add .

# Fazer commit
git commit -m "Primeiro commit: projeto inicial"
```

#### Passo 4: Publicar no GitHub
```bash
# Usando GitHub CLI
gh repo create meu-primeiro-projeto --public --push

# Ou manual:
# 1. Crie repositório no GitHub.com
# 2. Copie URL do repositório
git remote add origin https://github.com/seuusuario/meu-primeiro-projeto.git
git branch -M main
git push -u origin main
```

### Método 2: Clone de Repositório Existente

```bash
# Clonar repositório
git clone https://github.com/seuusuario/repositorio.git

# Abrir no VS Code
cd repositorio
code .
```

## 🔧 Comandos Git Básicos no VS Code

### Interface Gráfica (Source Control)

#### Visualizar Mudanças
1. **Ctrl + Shift + G** - Abrir Source Control
2. Ver arquivos modificados na aba lateral
3. Clicar em arquivo para ver diff

#### Fazer Commit
1. Escrever mensagem na caixa de texto
2. **Ctrl + Enter** - Commit
3. Ou clicar no ✓ (check)

#### Push/Pull
1. Clicar nos 3 pontos (...) em Source Control
2. Selecionar "Push" ou "Pull"

### Comandos Terminal Integrado

```bash
# Status dos arquivos
git status

# Ver diferenças
git diff nome-arquivo.js

# Adicionar arquivos específicos
git add index.html style.css

# Adicionar todos os arquivos
git add .

# Commit com mensagem
git commit -m "Adicionar página de login"

# Push para GitHub
git push

# Pull do GitHub
git pull

# Ver histórico de commits
git log --oneline

# Ver branches
git branch
```

## 🎯 Fluxo de Trabalho Completo

### Exemplo Prático: Desenvolvendo uma Página Web

#### Passo 1: Setup do Projeto
```bash
# Criar estrutura do projeto
mkdir website-pessoal
cd website-pessoal

# Inicializar Git
git init

# Criar arquivos iniciais
touch index.html style.css script.js README.md
```

#### Passo 2: Criar .gitignore
```bash
# Criar arquivo .gitignore
echo "node_modules/" > .gitignore
echo ".env" >> .gitignore
echo "*.log" >> .gitignore
echo ".DS_Store" >> .gitignore
```

#### Passo 3: Primeiro Commit
```html
<!-- index.html -->
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meu Website Pessoal</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <h1>Olá, eu sou [Seu Nome]</h1>
        <p>Desenvolvedor Web</p>
    </header>
    
    <main>
        <section id="sobre">
            <h2>Sobre Mim</h2>
            <p>Estudante de Sistemas de Informação, apaixonado por tecnologia.</p>
        </section>
        
        <section id="projetos">
            <h2>Projetos</h2>
            <div class="projeto">
                <h3>Website Pessoal</h3>
                <p>Meu primeiro projeto usando Git e GitHub!</p>
            </div>
        </section>
    </main>
    
    <script src="script.js"></script>
</body>
</html>
```

```css
/* style.css */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
}

header {
    background: #007bff;
    color: white;
    text-align: center;
    padding: 2rem;
}

main {
    max-width: 800px;
    margin: 2rem auto;
    padding: 0 1rem;
}

section {
    margin-bottom: 2rem;
}

.projeto {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 5px;
    margin-bottom: 1rem;
}
```

```javascript
// script.js
console.log('Website carregado!');

// Adicionar interatividade simples
document.addEventListener('DOMContentLoaded', function() {
    const header = document.querySelector('header h1');
    
    header.addEventListener('click', function() {
        alert('Olá! Bem-vindo ao meu website!');
    });
});
```

```markdown
# Website Pessoal

Meu primeiro projeto web usando Git e GitHub.

## Tecnologias
- HTML5
- CSS3
- JavaScript
- Git/GitHub

## Como executar
1. Clone o repositório
2. Abra `index.html` no navegador

## Autor
[Seu Nome] - [Seu Email]
```

#### Passo 4: Primeiro Commit e Push
```bash
# Adicionar todos os arquivos
git add .

# Verificar o que será commitado
git status

# Fazer commit
git commit -m "Primeiro commit: estrutura básica do website"

# Criar repositório no GitHub e push
gh repo create website-pessoal --public --push
```

#### Passo 5: Adicionar Funcionalidade Nova

```javascript
// Adicionar no script.js
// Sistema de tema escuro/claro
function toggleTheme() {
    const body = document.body;
    body.classList.toggle('dark-theme');
    
    const theme = body.classList.contains('dark-theme') ? 'dark' : 'light';
    localStorage.setItem('theme', theme);
}

// Carregar tema salvo
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
    }
    
    // Adicionar botão de tema
    const themeButton = document.createElement('button');
    themeButton.textContent = '🌓 Trocar Tema';
    themeButton.onclick = toggleTheme;
    themeButton.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    `;
    document.body.appendChild(themeButton);
});
```

```css
/* Adicionar no style.css */
.dark-theme {
    background-color: #121212;
    color: #ffffff;
}

.dark-theme header {
    background: #1f1f1f;
}

.dark-theme .projeto {
    background: #2d2d2d;
}
```

#### Passo 6: Commit da Nova Funcionalidade
```bash
# Ver arquivos modificados
git status

# Ver diferenças específicas
git diff script.js
git diff style.css

# Adicionar arquivos modificados
git add script.js style.css

# Commit com mensagem descritiva
git commit -m "Adicionar sistema de tema escuro/claro

- Implementar toggle de tema
- Salvar preferência no localStorage
- Adicionar estilos para tema escuro"

# Push para GitHub
git push
```

## 🌿 Trabalhando com Branches

### Criando e Usando Branches

#### No Terminal
```bash
# Criar e mudar para nova branch
git checkout -b nova-funcionalidade

# Ou usar comando mais novo
git switch -c nova-funcionalidade

# Ver todas as branches
git branch

# Mudar para branch existente
git switch main
git switch nova-funcionalidade

# Push da nova branch
git push -u origin nova-funcionalidade
```

#### No VS Code
1. **Ctrl + Shift + P** > "Git: Create Branch"
2. Digite nome da branch
3. Ou clique no nome da branch na barra inferior

### Exemplo: Adicionando Seção de Contato

#### Passo 1: Criar Branch
```bash
git switch -c secao-contato
```

#### Passo 2: Implementar Funcionalidade
```html
<!-- Adicionar no index.html antes do </main> -->
<section id="contato">
    <h2>Contato</h2>
    <form id="formContato">
        <div>
            <label for="nome">Nome:</label>
            <input type="text" id="nome" required>
        </div>
        <div>
            <label for="email">Email:</label>
            <input type="email" id="email" required>
        </div>
        <div>
            <label for="mensagem">Mensagem:</label>
            <textarea id="mensagem" required></textarea>
        </div>
        <button type="submit">Enviar</button>
    </form>
</section>
```

```css
/* Adicionar no style.css */
form {
    max-width: 400px;
}

form div {
    margin-bottom: 1rem;
}

label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: bold;
}

input, textarea {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid #ddd;
    border-radius: 3px;
}

button {
    background: #007bff;
    color: white;
    padding: 0.7rem 1.5rem;
    border: none;
    border-radius: 3px;
    cursor: pointer;
}

button:hover {
    background: #0056b3;
}

.dark-theme input,
.dark-theme textarea {
    background: #2d2d2d;
    color: white;
    border-color: #555;
}
```

```javascript
// Adicionar no script.js
// Sistema de contato
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('formContato');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const nome = document.getElementById('nome').value;
            const email = document.getElementById('email').value;
            const mensagem = document.getElementById('mensagem').value;
            
            // Simular envio
            alert(`Obrigado ${nome}! Sua mensagem foi enviada com sucesso!`);
            form.reset();
        });
    }
});
```

#### Passo 3: Commit na Branch
```bash
# Adicionar mudanças
git add .

# Commit
git commit -m "Adicionar seção de contato

- Criar formulário de contato
- Implementar validação básica
- Adicionar estilos responsivos
- Suporte ao tema escuro"

# Push da branch
git push -u origin secao-contato
```

#### Passo 4: Merge via GitHub (Pull Request)

**Opção 1: Via Web (Recomendado)**
1. Ir para github.com/seuusuario/website-pessoal
2. Clicar em "Compare & pull request"
3. Adicionar título e descrição
4. Clicar em "Create pull request"
5. Fazer merge

**Opção 2: Via Terminal**
```bash
# Voltar para main
git switch main

# Fazer merge
git merge secao-contato

# Push do merge
git push

# Deletar branch local (opcional)
git branch -d secao-contato

# Deletar branch remota (opcional)
git push origin --delete secao-contato
```

## ⚔️ Resolvendo Conflitos

### Simulando um Conflito

#### Cenário: Duas pessoas editando o mesmo arquivo

**Pessoa 1:**
```bash
# Na branch main
git switch main

# Editar README.md
echo "## Funcionalidades
- Tema escuro/claro
- Design responsivo" >> README.md

git add README.md
git commit -m "Atualizar README com funcionalidades"
git push
```

**Pessoa 2 (simulando):**
```bash
# Criar nova branch
git switch -c melhorar-readme

# Editar o mesmo README.md
echo "## Características
- Interface moderna
- Fácil navegação" >> README.md

git add README.md
git commit -m "Melhorar README com características"
git push -u origin melhorar-readme
```

### Resolvendo o Conflito

#### Passo 1: Tentar Merge
```bash
git switch main
git pull  # Atualizar main
git merge melhorar-readme
```

#### Passo 2: Resolver Conflito no VS Code
O VS Code mostrará o conflito assim:
```markdown
# Website Pessoal

Meu primeiro projeto web usando Git e GitHub.

<<<<<<< HEAD
## Funcionalidades
- Tema escuro/claro
- Design responsivo
=======
## Características
- Interface moderna
- Fácil navegação
>>>>>>> melhorar-readme
```

#### Passo 3: Escolher Resolução
No VS Code:
1. Clique em "Accept Current Change" (manter main)
2. Ou "Accept Incoming Change" (usar branch)
3. Ou "Accept Both Changes" (combinar)
4. Ou editar manualmente

**Resolução combinada:**
```markdown
# Website Pessoal

Meu primeiro projeto web usando Git e GitHub.

## Funcionalidades
- Tema escuro/claro
- Design responsivo
- Interface moderna
- Fácil navegação
```

#### Passo 4: Finalizar Merge
```bash
# Adicionar arquivo resolvido
git add README.md

# Finalizar merge
git commit -m "Resolver conflito no README: combinar funcionalidades e características"

# Push
git push
```

## 🤖 GitHub Actions

### CI/CD Básico para Website

#### Criar .github/workflows/deploy.yml
```yaml
name: Deploy Website

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
    
    - name: Validar HTML
      run: |
        # Instalar validador HTML
        npm install -g html-validate
        
        # Validar arquivos HTML
        html-validate *.html
    
    - name: Testar JavaScript
      run: |
        # Verificar sintaxe JavaScript
        node -c script.js
    
    - name: Lint CSS
      run: |
        # Instalar stylelint
        npm install -g stylelint stylelint-config-standard
        
        # Criar config
        echo '{"extends": "stylelint-config-standard"}' > .stylelintrc.json
        
        # Lint CSS
        stylelint "*.css"

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./
```

### Auto-deploy para Vercel
```yaml
name: Deploy to Vercel

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Vercel
      uses: amondnet/vercel-action@v20
      with:
        vercel-token: ${{ secrets.VERCEL_TOKEN }}
        vercel-org-id: ${{ secrets.ORG_ID }}
        vercel-project-id: ${{ secrets.PROJECT_ID }}
```

## 💡 Dicas Avançadas

### 1. Aliases Úteis
```bash
# Configurar aliases globais
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.cm commit
git config --global alias.unstage 'reset HEAD --'

# Agora você pode usar:
git st      # em vez de git status
git co main # em vez de git checkout main
git cm -m "mensagem" # em vez de git commit -m "mensagem"
```

### 2. VS Code Settings para Git
```json
// settings.json
{
  "git.autofetch": true,
  "git.confirmSync": false,
  "git.enableSmartCommit": true,
  "git.postCommitCommand": "push",
  "gitlens.hovers.currentLine.over": "line",
  "gitlens.currentLine.enabled": true,
  "gitlens.codeLens.enabled": true
}
```

### 3. Snippets para Commit Messages
```json
// commit-snippets.json
{
  "Feat commit": {
    "prefix": "feat",
    "body": "feat: ${1:description}"
  },
  "Fix commit": {
    "prefix": "fix", 
    "body": "fix: ${1:description}"
  },
  "Docs commit": {
    "prefix": "docs",
    "body": "docs: ${1:description}"
  }
}
```

### 4. Git Hooks com Husky
```bash
# Instalar husky
npm install --save-dev husky

# Configurar pre-commit hook
npx husky add .husky/pre-commit "npm run lint"
npx husky add .husky/pre-commit "npm run test"
```

### 5. Templates de Commit
```bash
# Criar template
echo "# Tipo(escopo): Descrição breve

# Corpo da mensagem (opcional)

# Footer (opcional)
# Fixes #issue-number" > ~/.gitmessage

# Configurar template
git config --global commit.template ~/.gitmessage
```

### 6. Stash para Salvar Trabalho Temporário
```bash
# Salvar trabalho atual sem commit
git stash

# Ver stashes
git stash list

# Recuperar último stash
git stash pop

# Recuperar stash específico
git stash pop stash@{0}

# Criar stash com nome
git stash save "trabalho em progresso na funcionalidade X"
```

### 7. Revert e Reset (Cuidado!)
```bash
# Reverter último commit (cria novo commit)
git revert HEAD

# Reset soft (mantém mudanças staged)
git reset --soft HEAD~1

# Reset hard (APAGA mudanças!)
git reset --hard HEAD~1
```

## 🏁 Exercício Prático Final

### Projeto: Todo List com Git/GitHub

1. **Criar repositório** "todo-list-js"
2. **Implementar** HTML básico
3. **Criar branch** "adicionar-todos"
4. **Implementar** JavaScript para adicionar tarefas
5. **Merge** via Pull Request
6. **Criar branch** "marcar-completo"
7. **Implementar** funcionalidade de completar
8. **Resolver conflito** simulado
9. **Deploy** com GitHub Actions
10. **Adicionar** badges no README

Este é o fluxo completo que todo desenvolvedor precisa dominar! 🚀

## 📚 Recursos Adicionais

- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [VS Code Git Tutorial](https://code.visualstudio.com/docs/editor/versioncontrol)
- [GitHub Learning Lab](https://lab.github.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

**Próximo passo:** Pratique criando seus próprios projetos e usando este fluxo! 🎯
