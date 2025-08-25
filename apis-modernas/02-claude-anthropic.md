# 🌟 Claude (Anthropic): API Avançada para IA Conversacional

## O que é Claude?
Claude é o modelo de IA da Anthropic, conhecido pela segurança, precisão e capacidade de manter conversas longas e contextuais. Excelente para análise de documentos e tarefas complexas.

## 🔑 Configuração da API

### Obter Chave da API
```bash
# Acesse: https://console.anthropic.com/
# Crie uma conta e gere sua API key
# Adicione no .env: ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Instalação
```bash
# Python
pip install anthropic

# JavaScript/Node.js
npm install @anthropic-ai/sdk
```

## 💻 Implementação JavaScript

### Configuração Básica
```javascript
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
    apiKey: 'sk-ant-api03-your-key-here',
});
```

### Chat Simples
```javascript
async function chatWithClaude(message) {
    try {
        const response = await anthropic.messages.create({
            model: 'claude-3-opus-20240229',
            max_tokens: 1000,
            temperature: 0.7,
            messages: [
                {
                    role: 'user',
                    content: message
                }
            ]
        });
        
        return response.content[0].text;
    } catch (error) {
        console.error('Erro:', error);
        throw error;
    }
}

// Uso
const resposta = await chatWithClaude("Explique machine learning em termos simples");
console.log(resposta);
```

### Interface Web Completa
```html
<!DOCTYPE html>
<html>
<head>
    <title>Chat com Claude</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
        .user { background: #007bff; color: white; text-align: right; }
        .assistant { background: #f8f9fa; border-left: 4px solid #007bff; }
        .input-area { display: flex; gap: 10px; }
        #messageInput { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        #sendButton { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>🤖 Chat com Claude (Anthropic)</h1>
    <div id="chatContainer" class="chat-container"></div>
    <div class="input-area">
        <input type="text" id="messageInput" placeholder="Digite sua mensagem..." onkeypress="handleEnter(event)">
        <button id="sendButton" onclick="sendMessage()">Enviar</button>
    </div>

    <script type="module">
        // Nota: Em produção, use um proxy backend para proteger a API key
        import Anthropic from 'https://esm.sh/@anthropic-ai/sdk';
        
        const anthropic = new Anthropic({
            apiKey: 'sk-ant-api03-your-key-here', // Substitua pela sua chave
            dangerouslyAllowBrowser: true // Apenas para demo - use backend em produção
        });

        let conversationHistory = [];

        window.sendMessage = async function() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;

            // Adicionar mensagem do usuário
            addMessage('user', message);
            conversationHistory.push({ role: 'user', content: message });
            input.value = '';

            // Mostrar indicador de digitação
            const typingDiv = addMessage('assistant', 'Claude está digitando...');

            try {
                const response = await anthropic.messages.create({
                    model: 'claude-3-sonnet-20240229',
                    max_tokens: 1000,
                    temperature: 0.7,
                    messages: conversationHistory
                });

                // Remover indicador de digitação
                typingDiv.remove();

                const assistantMessage = response.content[0].text;
                addMessage('assistant', assistantMessage);
                conversationHistory.push({ role: 'assistant', content: assistantMessage });

            } catch (error) {
                typingDiv.remove();
                addMessage('assistant', `Erro: ${error.message}`);
            }
        };

        function addMessage(role, content) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const roleLabel = role === 'user' ? 'Você' : 'Claude';
            messageDiv.innerHTML = `<strong>${roleLabel}:</strong><br>${content.replace(/\n/g, '<br>')}`;
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            return messageDiv;
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

## 🐍 Implementação Python

### Chat Básico
```python
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY')
)

def chat_with_claude():
    conversation = []
    
    print("Chat com Claude (digite 'sair' para encerrar)")
    print("=" * 50)
    
    while True:
        user_input = input("\nVocê: ")
        
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("Até logo!")
            break
        
        conversation.append({"role": "user", "content": user_input})
        
        try:
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=conversation
            )
            
            assistant_message = response.content[0].text
            print(f"\nClaude: {assistant_message}")
            
            conversation.append({"role": "assistant", "content": assistant_message})
            
        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    chat_with_claude()
```

### API Flask com Claude
```python
from flask import Flask, request, jsonify, render_template, stream_with_context, Response
from flask_cors import CORS
import anthropic
import os
import json

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

@app.route('/')
def index():
    return render_template('claude_chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        conversation_history = data.get('history', [])
        
        if not message:
            return jsonify({'error': 'Mensagem não pode estar vazia'}), 400
        
        # Adicionar nova mensagem ao histórico
        conversation_history.append({"role": "user", "content": message})
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.7,
            messages=conversation_history
        )
        
        assistant_message = response.content[0].text
        conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return jsonify({
            'response': assistant_message,
            'history': conversation_history,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/analyze-document', methods=['POST'])
def analyze_document():
    """Analisar documentos longos - especialidade do Claude"""
    try:
        data = request.json
        document_text = data.get('document', '')
        analysis_type = data.get('analysis_type', 'summary')
        
        if not document_text:
            return jsonify({'error': 'Documento não pode estar vazio'}), 400
        
        prompts = {
            'summary': f"Faça um resumo executivo do seguinte documento:\n\n{document_text}",
            'key_points': f"Liste os pontos principais do seguinte documento:\n\n{document_text}",
            'questions': f"Gere 5 perguntas importantes baseadas neste documento:\n\n{document_text}",
            'critique': f"Faça uma análise crítica construtiva deste documento:\n\n{document_text}"
        }
        
        prompt = prompts.get(analysis_type, prompts['summary'])
        
        response = client.messages.create(
            model="claude-3-opus-20240229",  # Opus para análises complexas
            max_tokens=2000,
            temperature=0.3,  # Menos criativo para análises
            messages=[{"role": "user", "content": prompt}]
        )
        
        return jsonify({
            'analysis': response.content[0].text,
            'analysis_type': analysis_type,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/code-review', methods=['POST'])
def code_review():
    """Revisão de código com Claude"""
    try:
        data = request.json
        code = data.get('code', '')
        language = data.get('language', 'python')
        
        if not code:
            return jsonify({'error': 'Código não pode estar vazio'}), 400
        
        prompt = f"""
        Por favor, revise este código {language} e forneça:
        
        1. **Análise Geral**: Qualidade e estrutura do código
        2. **Problemas Identificados**: Bugs, vulnerabilidades, problemas de performance
        3. **Sugestões de Melhoria**: Como otimizar e melhorar o código
        4. **Boas Práticas**: Recomendações seguindo padrões da linguagem
        
        Código para revisão:
        ```{language}
        {code}
        ```
        """
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return jsonify({
            'review': response.content[0].text,
            'language': language,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## 🎯 Casos de Uso Avançados

### 1. Análise de Sentimentos em Lote
```python
class SentimentAnalyzer:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def analyze_reviews(self, reviews):
        """Analisa sentimentos de múltiplas reviews"""
        
        reviews_text = "\n".join([f"{i+1}. {review}" for i, review in enumerate(reviews)])
        
        prompt = f"""
        Analise o sentimento de cada uma das seguintes reviews de produto.
        Para cada review, forneça:
        - Sentimento: Positivo/Neutro/Negativo
        - Confiança: 1-10
        - Temas principais mencionados
        
        Reviews:
        {reviews_text}
        
        Formate a resposta como JSON.
        """
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# Uso
analyzer = SentimentAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
reviews = [
    "Produto excelente, recomendo!",
    "Chegou quebrado, péssimo atendimento",
    "OK, nada excepcional mas atende"
]
resultado = analyzer.analyze_reviews(reviews)
```

### 2. Gerador de Testes Unitários
```python
def generate_unit_tests(code, language="python"):
    """Gera testes unitários para código fornecido"""
    
    prompt = f"""
    Gere testes unitários completos para o seguinte código {language}.
    
    Requisitos:
    - Use pytest (para Python) ou framework apropriado
    - Inclua casos de teste positivos e negativos
    - Teste casos edge
    - Inclua mocks quando necessário
    - Comente os testes
    
    Código para testar:
    ```{language}
    {code}
    ```
    
    Forneça apenas o código dos testes, bem formatado.
    """
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=2000,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# Exemplo de uso
code_to_test = """
def calculate_discount(price, discount_percent):
    if price < 0 or discount_percent < 0 or discount_percent > 100:
        raise ValueError("Invalid input")
    return price * (1 - discount_percent / 100)
"""

tests = generate_unit_tests(code_to_test)
print(tests)
```

### 3. Assistente de Documentação
```python
class DocumentationAssistant:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate_api_docs(self, code):
        """Gera documentação de API automaticamente"""
        
        prompt = f"""
        Analise este código de API e gere documentação completa no formato Markdown.
        
        Inclua:
        - Descrição geral da API
        - Endpoints disponíveis
        - Parâmetros de entrada
        - Exemplos de resposta
        - Códigos de erro possíveis
        - Exemplos de uso em diferentes linguagens
        
        Código da API:
        ```python
        {code}
        ```
        """
        
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=3000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_readme(self, project_structure, description):
        """Gera README.md baseado na estrutura do projeto"""
        
        prompt = f"""
        Crie um README.md profissional para este projeto.
        
        Descrição do projeto: {description}
        
        Estrutura do projeto:
        {project_structure}
        
        Inclua:
        - Badge do projeto
        - Descrição clara
        - Instalação
        - Uso básico
        - Exemplos
        - Contribuição
        - Licença
        - Contato
        """
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
```

## 📊 Comparação Claude vs Outros Modelos

| Característica | Claude | ChatGPT | Gemini |
|----------------|--------|---------|--------|
| Tamanho do contexto | 200K tokens | 128K tokens | 1M tokens |
| Análise de documentos | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Segurança | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Código | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Criatividade | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Preço | Médio | Alto | Baixo |

## 🔧 Dicas de Otimização

### Rate Limiting e Cache
```python
import time
from functools import lru_cache
import hashlib

class OptimizedClaude:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.last_request = 0
        self.min_interval = 1  # 1 segundo entre requests
    
    def _wait_for_rate_limit(self):
        """Implementa rate limiting básico"""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()
    
    @lru_cache(maxsize=100)
    def cached_completion(self, prompt_hash, model, max_tokens, temperature):
        """Versão com cache das completions"""
        self._wait_for_rate_limit()
        
        # Reconstruct prompt from hash (simplified)
        # In production, use proper cache with prompt storage
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": "cached_prompt"}]
        )
        
        return response.content[0].text
    
    def generate_with_cache(self, prompt, model="claude-3-sonnet-20240229", 
                          max_tokens=1000, temperature=0.7):
        """Gera resposta com cache"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return self.cached_completion(prompt_hash, model, max_tokens, temperature)
```

Veja exemplos completos funcionais na pasta `exemplos/claude/`!
