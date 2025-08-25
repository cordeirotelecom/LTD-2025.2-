# 🤖 Google Gemini: API Completa para Projetos Web

## O que é o Google Gemini?
Gemini é o modelo de IA mais avançado do Google, capaz de entender e gerar texto, código, imagens e muito mais. Substituto direto do ChatGPT para aplicações profissionais.

## 🔑 Configuração da API

### 1. Obter Chave da API
```bash
# Acesse: https://makersuite.google.com/app/apikey
# Crie uma chave de API gratuita
# Guarde em arquivo .env
```

### 2. Instalação
```bash
# JavaScript/Node.js
npm install @google/generative-ai

# Python
pip install google-generativeai
```

## 💻 Implementação JavaScript (Web)

### Configuração Básica
```javascript
import { GoogleGenerativeAI } from "@google/generative-ai";

const API_KEY = "SUA_API_KEY_AQUI";
const genAI = new GoogleGenerativeAI(API_KEY);

// Usar modelo Gemini Pro
const model = genAI.getGenerativeModel({ model: "gemini-pro" });
```

### Chat Simples
```html
<!DOCTYPE html>
<html>
<head>
    <title>Chat com Gemini</title>
    <style>
        .chat-container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
        .user { background: #e3f2fd; text-align: right; }
        .bot { background: #f3e5f5; }
        #input { width: 100%; padding: 10px; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>🤖 Chat com Google Gemini</h1>
        <div id="messages"></div>
        <input type="text" id="input" placeholder="Digite sua mensagem..." onkeypress="handleEnter(event)">
        <button onclick="sendMessage()">Enviar</button>
    </div>

    <script type="module">
        import { GoogleGenerativeAI } from "https://esm.run/@google/generative-ai";
        
        const API_KEY = "AIzaSyBLwXyour_api_key_here"; // Substitua pela sua chave
        const genAI = new GoogleGenerativeAI(API_KEY);
        const model = genAI.getGenerativeModel({ model: "gemini-pro" });

        window.sendMessage = async function() {
            const input = document.getElementById('input');
            const message = input.value.trim();
            if (!message) return;

            // Adicionar mensagem do usuário
            addMessage('user', message);
            input.value = '';

            try {
                // Gerar resposta do Gemini
                const result = await model.generateContent(message);
                const response = await result.response;
                const text = response.text();
                
                addMessage('bot', text);
            } catch (error) {
                addMessage('bot', 'Erro: ' + error.message);
            }
        };

        function addMessage(sender, text) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            div.innerHTML = `<strong>${sender === 'user' ? 'Você' : 'Gemini'}:</strong> ${text}`;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }

        window.handleEnter = function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        };
    </script>
</body>
</html>
```

### Chat com Histórico (Conversação Contínua)
```javascript
class GeminiChat {
    constructor(apiKey) {
        this.genAI = new GoogleGenerativeAI(apiKey);
        this.model = this.genAI.getGenerativeModel({ model: "gemini-pro" });
        this.chat = this.model.startChat({
            history: [],
            generationConfig: {
                maxOutputTokens: 1000,
                temperature: 0.7,
            },
        });
    }

    async sendMessage(message) {
        try {
            const result = await this.chat.sendMessage(message);
            const response = await result.response;
            return response.text();
        } catch (error) {
            throw new Error(`Erro ao enviar mensagem: ${error.message}`);
        }
    }

    getHistory() {
        return this.chat.getHistory();
    }
}

// Uso
const chat = new GeminiChat("SUA_API_KEY");

// Enviar mensagens
const resposta1 = await chat.sendMessage("Olá! Como você está?");
const resposta2 = await chat.sendMessage("Me explique sobre machine learning");
```

## 🐍 Implementação Python

### Configuração
```python
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
```

### Chat Básico
```python
def chat_gemini():
    model = genai.GenerativeModel('gemini-pro')
    
    print("Chat com Gemini (digite 'sair' para encerrar)")
    
    while True:
        user_input = input("Você: ")
        if user_input.lower() == 'sair':
            break
            
        try:
            response = model.generate_content(user_input)
            print(f"Gemini: {response.text}")
        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    chat_gemini()
```

### API Flask com Gemini
```python
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

genai.configure(api_key="SUA_API_KEY")
model = genai.GenerativeModel('gemini-pro')

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Mensagem não pode estar vazia'}), 400
        
        response = model.generate_content(message)
        
        return jsonify({
            'response': response.text,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Chat com streaming de resposta"""
    try:
        data = request.json
        message = data.get('message', '')
        
        def generate():
            response = model.generate_content(
                message,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield f"data: {chunk.text}\n\n"
        
        return Response(generate(), mimetype='text/plain')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## 🔧 Funcionalidades Avançadas

### 1. Análise de Imagens (Gemini Pro Vision)
```javascript
// Upload e análise de imagem
async function analyzeImage(imageFile) {
    const model = genAI.getGenerativeModel({ model: "gemini-pro-vision" });
    
    const imageParts = [
        {
            inlineData: {
                data: await fileToBase64(imageFile),
                mimeType: imageFile.type
            }
        }
    ];
    
    const result = await model.generateContent([
        "Descreva esta imagem em detalhes",
        ...imageParts
    ]);
    
    return result.response.text();
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = error => reject(error);
    });
}
```

### 2. Geração de Código
```javascript
async function generateCode(prompt) {
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });
    
    const codePrompt = `
        Gere código Python para: ${prompt}
        
        Requisitos:
        - Código limpo e comentado
        - Incluir tratamento de erro
        - Usar boas práticas
        - Adicionar docstrings
    `;
    
    const result = await model.generateContent(codePrompt);
    return result.response.text();
}

// Exemplo de uso
const code = await generateCode("criar um web scraper para extrair preços de produtos");
```

### 3. Sistema de Moderação de Conteúdo
```python
def moderate_content(text):
    """Verifica se o conteúdo é apropriado"""
    
    prompt = f"""
    Analise o seguinte texto e classifique como:
    - SEGURO: Conteúdo apropriado
    - SUSPEITO: Conteúdo questionável
    - PERIGOSO: Conteúdo inapropriado
    
    Texto: "{text}"
    
    Responda apenas com: SEGURO, SUSPEITO ou PERIGOSO
    """
    
    response = model.generate_content(prompt)
    return response.text.strip()

# Uso
resultado = moderate_content("Como fazer uma bomba?")
if resultado == "PERIGOSO":
    print("Conteúdo bloqueado")
```

## 🎯 Casos de Uso Práticos

### 1. Assistente de Código
```javascript
class CodeAssistant {
    constructor(apiKey) {
        this.genAI = new GoogleGenerativeAI(apiKey);
        this.model = this.genAI.getGenerativeModel({ model: "gemini-pro" });
    }
    
    async explainCode(code, language) {
        const prompt = `Explique este código ${language} de forma didática:\n\n${code}`;
        const result = await this.model.generateContent(prompt);
        return result.response.text();
    }
    
    async optimizeCode(code, language) {
        const prompt = `Otimize este código ${language} mantendo a funcionalidade:\n\n${code}`;
        const result = await this.model.generateContent(prompt);
        return result.response.text();
    }
    
    async findBugs(code, language) {
        const prompt = `Identifique possíveis bugs neste código ${language}:\n\n${code}`;
        const result = await this.model.generateContent(prompt);
        return result.response.text();
    }
}
```

### 2. Gerador de Conteúdo para Blog
```python
class ContentGenerator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_blog_post(self, topic, tone="profissional", length="médio"):
        prompt = f"""
        Escreva um artigo de blog sobre: {topic}
        
        Características:
        - Tom: {tone}
        - Tamanho: {length}
        - Incluir introdução, desenvolvimento e conclusão
        - Usar subtítulos
        - Incluir exemplos práticos
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def generate_social_media_posts(self, topic, platform="instagram"):
        prompt = f"""
        Crie 5 posts para {platform} sobre: {topic}
        
        Cada post deve ter:
        - Texto envolvente
        - Hashtags relevantes
        - Call to action
        """
        
        response = self.model.generate_content(prompt)
        return response.text
```

## ⚡ Dicas de Performance

### Rate Limiting
```javascript
class RateLimitedGemini {
    constructor(apiKey, requestsPerMinute = 60) {
        this.genAI = new GoogleGenerativeAI(apiKey);
        this.model = this.genAI.getGenerativeModel({ model: "gemini-pro" });
        this.requests = [];
        this.maxRequests = requestsPerMinute;
    }
    
    async generateContent(prompt) {
        await this.waitForRateLimit();
        this.requests.push(Date.now());
        return await this.model.generateContent(prompt);
    }
    
    async waitForRateLimit() {
        const now = Date.now();
        const minute = 60 * 1000;
        
        // Remove requests older than 1 minute
        this.requests = this.requests.filter(time => now - time < minute);
        
        if (this.requests.length >= this.maxRequests) {
            const oldestRequest = this.requests[0];
            const waitTime = minute - (now - oldestRequest);
            await new Promise(resolve => setTimeout(resolve, waitTime));
        }
    }
}
```

### Cache de Respostas
```javascript
class CachedGemini {
    constructor(apiKey) {
        this.gemini = new RateLimitedGemini(apiKey);
        this.cache = new Map();
    }
    
    async generateContent(prompt) {
        const hash = this.hashPrompt(prompt);
        
        if (this.cache.has(hash)) {
            return this.cache.get(hash);
        }
        
        const result = await this.gemini.generateContent(prompt);
        this.cache.set(hash, result);
        
        return result;
    }
    
    hashPrompt(prompt) {
        // Simple hash function
        let hash = 0;
        for (let i = 0; i < prompt.length; i++) {
            const char = prompt.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return hash.toString();
    }
}
```

Veja exemplos completos funcionais na pasta `exemplos/gemini/`!
