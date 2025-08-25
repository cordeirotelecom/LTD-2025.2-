# APIs de Inteligência Artificial para Web: Guia Técnico Completo

## Introdução às APIs de IA

APIs (Application Programming Interfaces) de Inteligência Artificial representam pontes tecnológicas que democratizam o acesso a capacidades de IA avançadas, permitindo que desenvolvedores integrem funcionalidades inteligentes em aplicações web sem necessidade de expertise profunda em machine learning ou recursos computacionais massivos para treinamento de modelos.

## Fundamentos Técnicos

### **Arquitetura de APIs de IA**:

```javascript
// Padrão de Integração com APIs de IA
class AIAPIClient {
    constructor(config) {
        this.baseURL = config.baseURL;
        this.apiKey = config.apiKey;
        this.timeout = config.timeout || 30000;
        this.retryAttempts = config.retryAttempts || 3;
        this.rateLimiter = new RateLimiter(config.rateLimit);
    }
    
    async makeRequest(endpoint, data, options = {}) {
        const requestConfig = {
            method: options.method || 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
                'User-Agent': 'WebApp/1.0',
                ...options.headers
            },
            body: JSON.stringify(data),
            timeout: this.timeout
        };
        
        return await this.executeWithRetry(endpoint, requestConfig);
    }
    
    async executeWithRetry(endpoint, config, attempt = 1) {
        try {
            // Rate limiting
            await this.rateLimiter.waitForToken();
            
            const response = await fetch(`${this.baseURL}${endpoint}`, config);
            
            if (!response.ok) {
                throw new APIError(response.status, await response.text());
            }
            
            return await response.json();
        } catch (error) {
            if (attempt < this.retryAttempts && this.shouldRetry(error)) {
                await this.backoffDelay(attempt);
                return this.executeWithRetry(endpoint, config, attempt + 1);
            }
            throw error;
        }
    }
    
    shouldRetry(error) {
        // Retry on network errors and 5xx server errors
        return error.code === 'NETWORK_ERROR' || 
               (error.status >= 500 && error.status < 600) ||
               error.status === 429; // Rate limit
    }
    
    async backoffDelay(attempt) {
        const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
        await new Promise(resolve => setTimeout(resolve, delay));
    }
}

// Rate Limiter Implementation
class RateLimiter {
    constructor(requestsPerSecond = 10) {
        this.requestsPerSecond = requestsPerSecond;
        this.tokens = requestsPerSecond;
        this.lastRefill = Date.now();
    }
    
    async waitForToken() {
        this.refillTokens();
        
        if (this.tokens >= 1) {
            this.tokens--;
            return;
        }
        
        // Wait until next token is available
        const waitTime = (1 / this.requestsPerSecond) * 1000;
        await new Promise(resolve => setTimeout(resolve, waitTime));
        return this.waitForToken();
    }
    
    refillTokens() {
        const now = Date.now();
        const timePassed = (now - this.lastRefill) / 1000;
        const tokensToAdd = timePassed * this.requestsPerSecond;
        
        this.tokens = Math.min(this.requestsPerSecond, this.tokens + tokensToAdd);
        this.lastRefill = now;
    }
}

// Custom Error Classes
class APIError extends Error {
    constructor(status, message) {
        super(message);
        this.name = 'APIError';
        this.status = status;
    }
}

class RateLimitError extends APIError {
    constructor(message, resetTime) {
        super(429, message);
        this.name = 'RateLimitError';
        this.resetTime = resetTime;
    }
}
```

## OpenAI APIs: GPT e DALL-E

### **Implementação Completa com GPT-4**:

```javascript
class OpenAIClient extends AIAPIClient {
    constructor(apiKey) {
        super({
            baseURL: 'https://api.openai.com/v1',
            apiKey,
            rateLimit: { requestsPerSecond: 20 }
        });
        
        this.models = {
            gpt4: 'gpt-4',
            gpt35Turbo: 'gpt-3.5-turbo',
            whisper: 'whisper-1',
            dalle3: 'dall-e-3'
        };
    }
    
    async generateText(prompt, options = {}) {
        const defaultOptions = {
            model: this.models.gpt35Turbo,
            max_tokens: 1000,
            temperature: 0.7,
            top_p: 1,
            frequency_penalty: 0,
            presence_penalty: 0,
            stream: false
        };
        
        const requestData = {
            messages: this.formatMessages(prompt, options.context),
            ...defaultOptions,
            ...options
        };
        
        try {
            const response = await this.makeRequest('/chat/completions', requestData);
            
            return {
                text: response.choices[0].message.content,
                usage: response.usage,
                model: response.model,
                id: response.id
            };
        } catch (error) {
            throw new OpenAIError('Text generation failed', error);
        }
    }
    
    async generateStreamingText(prompt, options = {}, onChunk) {
        const requestData = {
            messages: this.formatMessages(prompt, options.context),
            model: options.model || this.models.gpt35Turbo,
            max_tokens: options.max_tokens || 1000,
            temperature: options.temperature || 0.7,
            stream: true
        };
        
        const response = await fetch(`${this.baseURL}/chat/completions`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n').filter(line => line.trim());
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        
                        if (data === '[DONE]') return;
                        
                        try {
                            const parsed = JSON.parse(data);
                            const content = parsed.choices[0]?.delta?.content;
                            
                            if (content) {
                                onChunk(content);
                            }
                        } catch (e) {
                            console.warn('Failed to parse streaming chunk:', e);
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }
    
    async generateImage(prompt, options = {}) {
        const defaultOptions = {
            model: this.models.dalle3,
            size: '1024x1024',
            quality: 'standard',
            n: 1
        };
        
        const requestData = {
            prompt,
            ...defaultOptions,
            ...options
        };
        
        try {
            const response = await this.makeRequest('/images/generations', requestData);
            
            return {
                images: response.data.map(img => ({
                    url: img.url,
                    revised_prompt: img.revised_prompt
                })),
                created: response.created
            };
        } catch (error) {
            throw new OpenAIError('Image generation failed', error);
        }
    }
    
    async transcribeAudio(audioFile, options = {}) {
        const formData = new FormData();
        formData.append('file', audioFile);
        formData.append('model', options.model || this.models.whisper);
        formData.append('language', options.language || 'pt');
        formData.append('response_format', options.format || 'json');
        
        if (options.prompt) {
            formData.append('prompt', options.prompt);
        }
        
        try {
            const response = await fetch(`${this.baseURL}/audio/transcriptions`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`
                },
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            throw new OpenAIError('Audio transcription failed', error);
        }
    }
    
    async createEmbedding(text, model = 'text-embedding-ada-002') {
        const requestData = {
            input: text,
            model
        };
        
        try {
            const response = await this.makeRequest('/embeddings', requestData);
            
            return {
                embedding: response.data[0].embedding,
                usage: response.usage,
                model: response.model
            };
        } catch (error) {
            throw new OpenAIError('Embedding creation failed', error);
        }
    }
    
    formatMessages(prompt, context = []) {
        const messages = [...context];
        
        if (typeof prompt === 'string') {
            messages.push({ role: 'user', content: prompt });
        } else if (Array.isArray(prompt)) {
            messages.push(...prompt);
        }
        
        return messages;
    }
    
    // Function calling capabilities
    async callFunction(prompt, functions, options = {}) {
        const requestData = {
            messages: this.formatMessages(prompt, options.context),
            model: options.model || this.models.gpt35Turbo,
            functions,
            function_call: options.function_call || 'auto',
            temperature: options.temperature || 0.3
        };
        
        try {
            const response = await this.makeRequest('/chat/completions', requestData);
            const message = response.choices[0].message;
            
            if (message.function_call) {
                return {
                    type: 'function_call',
                    function_name: message.function_call.name,
                    arguments: JSON.parse(message.function_call.arguments),
                    usage: response.usage
                };
            } else {
                return {
                    type: 'text_response',
                    text: message.content,
                    usage: response.usage
                };
            }
        } catch (error) {
            throw new OpenAIError('Function calling failed', error);
        }
    }
}

// Usage Example: AI-Powered Code Generator
class AICodeGenerator {
    constructor(openaiApiKey) {
        this.openai = new OpenAIClient(openaiApiKey);
        this.codeTemplates = new Map();
        this.setupTemplates();
    }
    
    setupTemplates() {
        this.codeTemplates.set('react_component', {
            prompt: `Generate a React functional component that {description}. 
                    Include proper TypeScript types, error handling, and modern React patterns.`,
            functions: [{
                name: 'generate_react_component',
                description: 'Generate a React component with TypeScript',
                parameters: {
                    type: 'object',
                    properties: {
                        componentName: { type: 'string' },
                        props: { type: 'object' },
                        code: { type: 'string' }
                    },
                    required: ['componentName', 'code']
                }
            }]
        });
        
        this.codeTemplates.set('api_endpoint', {
            prompt: `Create a {framework} API endpoint that {description}. 
                    Include proper error handling, validation, and documentation.`,
            functions: [{
                name: 'generate_api_endpoint',
                description: 'Generate an API endpoint',
                parameters: {
                    type: 'object',
                    properties: {
                        endpoint: { type: 'string' },
                        method: { type: 'string' },
                        code: { type: 'string' },
                        documentation: { type: 'string' }
                    },
                    required: ['endpoint', 'method', 'code']
                }
            }]
        });
    }
    
    async generateCode(type, description, framework = 'javascript') {
        const template = this.codeTemplates.get(type);
        
        if (!template) {
            throw new Error(`Unknown code template: ${type}`);
        }
        
        const prompt = template.prompt
            .replace('{description}', description)
            .replace('{framework}', framework);
        
        try {
            const response = await this.openai.callFunction(
                prompt,
                template.functions,
                { model: 'gpt-4', temperature: 0.3 }
            );
            
            if (response.type === 'function_call') {
                return {
                    success: true,
                    code: response.arguments.code,
                    metadata: response.arguments,
                    usage: response.usage
                };
            } else {
                return {
                    success: false,
                    error: 'No function call generated',
                    fallback: response.text
                };
            }
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    async explainCode(code, language) {
        const prompt = `Explain this ${language} code step by step, including:
            1. What it does
            2. How it works
            3. Best practices used
            4. Potential improvements
            
            Code:
            \`\`\`${language}
            ${code}
            \`\`\``;
        
        try {
            const response = await this.openai.generateText(prompt, {
                model: 'gpt-4',
                max_tokens: 1500,
                temperature: 0.3
            });
            
            return {
                explanation: response.text,
                usage: response.usage
            };
        } catch (error) {
            throw new OpenAIError('Code explanation failed', error);
        }
    }
    
    async optimizeCode(code, language, optimization_goal = 'performance') {
        const prompt = `Optimize this ${language} code for ${optimization_goal}. 
            Provide the optimized version and explain the improvements made.
            
            Original code:
            \`\`\`${language}
            ${code}
            \`\`\``;
        
        try {
            const response = await this.openai.generateText(prompt, {
                model: 'gpt-4',
                max_tokens: 2000,
                temperature: 0.2
            });
            
            return {
                optimized_code: this.extractCodeFromResponse(response.text),
                explanation: response.text,
                usage: response.usage
            };
        } catch (error) {
            throw new OpenAIError('Code optimization failed', error);
        }
    }
    
    extractCodeFromResponse(text) {
        const codeBlockRegex = /```[\w]*\n([\s\S]*?)\n```/g;
        const matches = text.match(codeBlockRegex);
        
        if (matches) {
            return matches[0].replace(/```[\w]*\n|```/g, '').trim();
        }
        
        return text;
    }
}

class OpenAIError extends Error {
    constructor(message, originalError) {
        super(message);
        this.name = 'OpenAIError';
        this.originalError = originalError;
    }
}
```

## Google Cloud AI APIs

### **Visão Computacional e Linguagem Natural**:

```javascript
class GoogleCloudAI {
    constructor(apiKey, projectId) {
        this.apiKey = apiKey;
        this.projectId = projectId;
        this.baseURL = 'https://googleapis.com';
    }
    
    async analyzeImage(imageData, features = ['LABEL_DETECTION', 'TEXT_DETECTION']) {
        const requestData = {
            requests: [{
                image: {
                    content: imageData // Base64 encoded image
                },
                features: features.map(feature => ({
                    type: feature,
                    maxResults: 10
                }))
            }]
        };
        
        try {
            const response = await fetch(
                `${this.baseURL}/vision/v1/images:annotate?key=${this.apiKey}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                }
            );
            
            const result = await response.json();
            
            return {
                labels: result.responses[0].labelAnnotations || [],
                text: result.responses[0].textAnnotations || [],
                faces: result.responses[0].faceAnnotations || [],
                objects: result.responses[0].localizedObjectAnnotations || []
            };
        } catch (error) {
            throw new GoogleCloudError('Image analysis failed', error);
        }
    }
    
    async analyzeSentiment(text, language = 'pt') {
        const requestData = {
            document: {
                type: 'PLAIN_TEXT',
                content: text,
                language
            },
            encodingType: 'UTF8'
        };
        
        try {
            const response = await fetch(
                `${this.baseURL}/language/v1/documents:analyzeSentiment?key=${this.apiKey}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                }
            );
            
            const result = await response.json();
            
            return {
                overallSentiment: {
                    score: result.documentSentiment.score,
                    magnitude: result.documentSentiment.magnitude
                },
                sentences: result.sentences || []
            };
        } catch (error) {
            throw new GoogleCloudError('Sentiment analysis failed', error);
        }
    }
    
    async translateText(text, targetLanguage, sourceLanguage = 'auto') {
        const requestData = {
            q: text,
            target: targetLanguage,
            source: sourceLanguage !== 'auto' ? sourceLanguage : undefined,
            format: 'text'
        };
        
        try {
            const response = await fetch(
                `${this.baseURL}/language/translate/v2?key=${this.apiKey}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                }
            );
            
            const result = await response.json();
            
            return {
                translatedText: result.data.translations[0].translatedText,
                detectedSourceLanguage: result.data.translations[0].detectedSourceLanguage
            };
        } catch (error) {
            throw new GoogleCloudError('Translation failed', error);
        }
    }
    
    async extractEntities(text, language = 'pt') {
        const requestData = {
            document: {
                type: 'PLAIN_TEXT',
                content: text,
                language
            },
            encodingType: 'UTF8'
        };
        
        try {
            const response = await fetch(
                `${this.baseURL}/language/v1/documents:analyzeEntities?key=${this.apiKey}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                }
            );
            
            const result = await response.json();
            
            return {
                entities: result.entities.map(entity => ({
                    name: entity.name,
                    type: entity.type,
                    salience: entity.salience,
                    mentions: entity.mentions
                }))
            };
        } catch (error) {
            throw new GoogleCloudError('Entity extraction failed', error);
        }
    }
}

class GoogleCloudError extends Error {
    constructor(message, originalError) {
        super(message);
        this.name = 'GoogleCloudError';
        this.originalError = originalError;
    }
}
```

## Microsoft Azure Cognitive Services

### **Integração Completa**:

```javascript
class AzureCognitiveServices {
    constructor(subscriptionKey, endpoint, region = 'eastus') {
        this.subscriptionKey = subscriptionKey;
        this.endpoint = endpoint;
        this.region = region;
    }
    
    async analyzeFace(imageData) {
        const faceAttributes = [
            'age', 'gender', 'smile', 'emotion',
            'glasses', 'hair', 'makeup', 'accessories'
        ].join(',');
        
        try {
            const response = await fetch(
                `${this.endpoint}/face/v1.0/detect?returnFaceAttributes=${faceAttributes}`,
                {
                    method: 'POST',
                    headers: {
                        'Ocp-Apim-Subscription-Key': this.subscriptionKey,
                        'Content-Type': 'application/octet-stream'
                    },
                    body: imageData
                }
            );
            
            const faces = await response.json();
            
            return faces.map(face => ({
                id: face.faceId,
                rectangle: face.faceRectangle,
                attributes: face.faceAttributes
            }));
        } catch (error) {
            throw new AzureError('Face analysis failed', error);
        }
    }
    
    async speechToText(audioBlob, language = 'pt-BR') {
        const formData = new FormData();
        formData.append('audio', audioBlob);
        
        try {
            const response = await fetch(
                `${this.endpoint}/speechtotext/v3.0/transcriptions:transcribe?language=${language}`,
                {
                    method: 'POST',
                    headers: {
                        'Ocp-Apim-Subscription-Key': this.subscriptionKey
                    },
                    body: formData
                }
            );
            
            const result = await response.json();
            
            return {
                text: result.displayText,
                confidence: result.confidence,
                duration: result.duration
            };
        } catch (error) {
            throw new AzureError('Speech to text failed', error);
        }
    }
    
    async textToSpeech(text, voice = 'pt-BR-FranciscaNeural') {
        const ssml = `
            <speak version="1.0" xml:lang="pt-BR">
                <voice name="${voice}">
                    ${text}
                </voice>
            </speak>
        `;
        
        try {
            const response = await fetch(
                `${this.endpoint}/cognitiveservices/v1/text-to-speech`,
                {
                    method: 'POST',
                    headers: {
                        'Ocp-Apim-Subscription-Key': this.subscriptionKey,
                        'Content-Type': 'application/ssml+xml',
                        'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3'
                    },
                    body: ssml
                }
            );
            
            const audioBlob = await response.blob();
            return audioBlob;
        } catch (error) {
            throw new AzureError('Text to speech failed', error);
        }
    }
    
    async analyzeText(text, options = {}) {
        const requestData = {
            documents: [{
                id: '1',
                text: text,
                language: options.language || 'pt'
            }]
        };
        
        const analyses = [];
        
        // Sentiment Analysis
        if (options.sentiment !== false) {
            try {
                const sentimentResponse = await fetch(
                    `${this.endpoint}/text/analytics/v3.1/sentiment`,
                    {
                        method: 'POST',
                        headers: {
                            'Ocp-Apim-Subscription-Key': this.subscriptionKey,
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(requestData)
                    }
                );
                
                const sentimentResult = await sentimentResponse.json();
                analyses.push({
                    type: 'sentiment',
                    result: sentimentResult.documents[0]
                });
            } catch (error) {
                console.error('Sentiment analysis failed:', error);
            }
        }
        
        // Key Phrases
        if (options.keyPhrases !== false) {
            try {
                const keyPhrasesResponse = await fetch(
                    `${this.endpoint}/text/analytics/v3.1/keyPhrases`,
                    {
                        method: 'POST',
                        headers: {
                            'Ocp-Apim-Subscription-Key': this.subscriptionKey,
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(requestData)
                    }
                );
                
                const keyPhrasesResult = await keyPhrasesResponse.json();
                analyses.push({
                    type: 'keyPhrases',
                    result: keyPhrasesResult.documents[0]
                });
            } catch (error) {
                console.error('Key phrases extraction failed:', error);
            }
        }
        
        return analyses;
    }
}

class AzureError extends Error {
    constructor(message, originalError) {
        super(message);
        this.name = 'AzureError';
        this.originalError = originalError;
    }
}
```

## Implementação Prática: Plataforma de Conteúdo IA

### **Sistema Completo**:

```javascript
// AI Content Platform
class AIContentPlatform {
    constructor(apiKeys) {
        this.openai = new OpenAIClient(apiKeys.openai);
        this.googleCloud = new GoogleCloudAI(apiKeys.googleCloud, apiKeys.projectId);
        this.azure = new AzureCognitiveServices(
            apiKeys.azure.key, 
            apiKeys.azure.endpoint,
            apiKeys.azure.region
        );
        
        this.contentCache = new Map();
        this.analytics = new AnalyticsEngine();
    }
    
    async createArticle(topic, style = 'informative', length = 'medium') {
        const cacheKey = `article_${topic}_${style}_${length}`;
        
        if (this.contentCache.has(cacheKey)) {
            return this.contentCache.get(cacheKey);
        }
        
        try {
            // Generate outline
            const outline = await this.generateOutline(topic, style);
            
            // Generate content sections
            const sections = await Promise.all(
                outline.sections.map(section => 
                    this.generateSection(section, style, length)
                )
            );
            
            // Generate images
            const images = await this.generateImages(topic, sections.length);
            
            // Analyze sentiment and optimize
            const sentiment = await this.googleCloud.analyzeSentiment(
                sections.join('\n\n')
            );
            
            const article = {
                title: outline.title,
                introduction: outline.introduction,
                sections: sections.map((content, index) => ({
                    title: outline.sections[index],
                    content,
                    image: images[index]
                })),
                conclusion: await this.generateConclusion(topic, sections),
                metadata: {
                    topic,
                    style,
                    length,
                    sentiment: sentiment.overallSentiment,
                    wordCount: this.countWords(sections.join(' ')),
                    generatedAt: new Date()
                }
            };
            
            this.contentCache.set(cacheKey, article);
            this.analytics.trackGeneration('article', topic);
            
            return article;
        } catch (error) {
            throw new ContentGenerationError('Article creation failed', error);
        }
    }
    
    async generateOutline(topic, style) {
        const prompt = `Create a detailed outline for a ${style} article about "${topic}". 
            Include:
            1. An engaging title
            2. A compelling introduction
            3. 4-6 main sections with descriptive titles
            4. Key points for each section
            
            Format as JSON with the structure:
            {
                "title": "...",
                "introduction": "...",
                "sections": ["Section 1", "Section 2", ...]
            }`;
        
        const response = await this.openai.generateText(prompt, {
            model: 'gpt-4',
            temperature: 0.7,
            max_tokens: 800
        });
        
        try {
            return JSON.parse(response.text);
        } catch (e) {
            // Fallback parsing
            return this.parseOutlineFromText(response.text);
        }
    }
    
    async generateSection(sectionTitle, style, length) {
        const wordCounts = {
            short: 150,
            medium: 300,
            long: 500
        };
        
        const targetWords = wordCounts[length] || 300;
        
        const prompt = `Write a ${style} section titled "${sectionTitle}". 
            Target length: approximately ${targetWords} words.
            Make it engaging, informative, and well-structured with examples.`;
        
        const response = await this.openai.generateText(prompt, {
            model: 'gpt-3.5-turbo',
            temperature: 0.6,
            max_tokens: Math.ceil(targetWords * 1.5)
        });
        
        return response.text;
    }
    
    async generateImages(topic, count) {
        const images = [];
        
        for (let i = 0; i < Math.min(count, 3); i++) {
            try {
                const imagePrompt = `Professional illustration about ${topic}, 
                    modern style, clean design, suitable for article illustration`;
                
                const imageResponse = await this.openai.generateImage(imagePrompt, {
                    size: '1024x1024',
                    quality: 'standard'
                });
                
                images.push({
                    url: imageResponse.images[0].url,
                    alt: `Illustration about ${topic}`,
                    caption: `Visual representation related to ${topic}`
                });
            } catch (error) {
                console.error('Image generation failed:', error);
                images.push(null);
            }
        }
        
        return images;
    }
    
    async generateConclusion(topic, sections) {
        const prompt = `Write a compelling conclusion for an article about "${topic}".
            Summarize the key points and provide actionable insights.
            Section summaries: ${sections.slice(0, 2).join(' ... ')}`;
        
        const response = await this.openai.generateText(prompt, {
            model: 'gpt-3.5-turbo',
            temperature: 0.5,
            max_tokens: 200
        });
        
        return response.text;
    }
    
    async optimizeContent(content, target = 'engagement') {
        const optimizations = {
            engagement: 'Make this content more engaging and interactive',
            seo: 'Optimize this content for search engines',
            readability: 'Improve the readability and clarity of this content',
            conversion: 'Optimize this content for conversions and calls-to-action'
        };
        
        const prompt = `${optimizations[target] || optimizations.engagement}:
            
            ${content}
            
            Provide the optimized version:`;
        
        const response = await this.openai.generateText(prompt, {
            model: 'gpt-4',
            temperature: 0.3,
            max_tokens: 1500
        });
        
        return response.text;
    }
    
    async translateContent(content, targetLanguage) {
        try {
            const translation = await this.googleCloud.translateText(
                content, 
                targetLanguage
            );
            
            return {
                translatedText: translation.translatedText,
                detectedLanguage: translation.detectedSourceLanguage,
                confidence: 0.95 // Google Translate is generally high confidence
            };
        } catch (error) {
            throw new TranslationError('Content translation failed', error);
        }
    }
    
    async analyzeContentQuality(content) {
        try {
            // Sentiment analysis
            const sentiment = await this.googleCloud.analyzeSentiment(content);
            
            // Extract entities
            const entities = await this.googleCloud.extractEntities(content);
            
            // Azure text analysis
            const azureAnalysis = await this.azure.analyzeText(content, {
                sentiment: true,
                keyPhrases: true
            });
            
            // Calculate readability (simplified)
            const readability = this.calculateReadability(content);
            
            return {
                sentiment: sentiment.overallSentiment,
                entities: entities.entities,
                keyPhrases: azureAnalysis.find(a => a.type === 'keyPhrases')?.result?.keyPhrases || [],
                readability,
                wordCount: this.countWords(content),
                estimatedReadingTime: Math.ceil(this.countWords(content) / 200),
                qualityScore: this.calculateQualityScore(sentiment, readability, entities)
            };
        } catch (error) {
            throw new AnalysisError('Content quality analysis failed', error);
        }
    }
    
    calculateReadability(text) {
        const sentences = text.split(/[.!?]+/).length;
        const words = this.countWords(text);
        const avgWordsPerSentence = words / sentences;
        
        // Simplified readability score (0-100)
        let score = 100;
        if (avgWordsPerSentence > 20) score -= 20;
        if (avgWordsPerSentence > 30) score -= 30;
        
        return Math.max(0, Math.min(100, score));
    }
    
    calculateQualityScore(sentiment, readability, entities) {
        let score = 50; // Base score
        
        // Sentiment contribution
        if (Math.abs(sentiment.score) > 0.5) score += 15;
        
        // Readability contribution
        score += (readability / 100) * 20;
        
        // Entity richness contribution
        score += Math.min(entities.length * 3, 15);
        
        return Math.min(100, score);
    }
    
    countWords(text) {
        return text.trim().split(/\s+/).length;
    }
    
    parseOutlineFromText(text) {
        // Fallback parser for when JSON parsing fails
        const lines = text.split('\n').filter(line => line.trim());
        
        return {
            title: lines[0] || 'Generated Article',
            introduction: lines[1] || 'Introduction to the topic.',
            sections: lines.slice(2, 8) || ['Section 1', 'Section 2', 'Section 3']
        };
    }
}

// Analytics Engine
class AnalyticsEngine {
    constructor() {
        this.metrics = {
            generations: 0,
            tokens_used: 0,
            cost_estimate: 0,
            content_types: new Map()
        };
    }
    
    trackGeneration(type, topic) {
        this.metrics.generations++;
        
        const count = this.metrics.content_types.get(type) || 0;
        this.metrics.content_types.set(type, count + 1);
        
        // Store in localStorage for persistence
        localStorage.setItem('ai_analytics', JSON.stringify({
            ...this.metrics,
            content_types: Array.from(this.metrics.content_types.entries())
        }));
    }
    
    getUsageReport() {
        return {
            totalGenerations: this.metrics.generations,
            tokensUsed: this.metrics.tokens_used,
            estimatedCost: this.metrics.cost_estimate,
            contentBreakdown: Object.fromEntries(this.metrics.content_types),
            averageCostPerGeneration: this.metrics.cost_estimate / this.metrics.generations || 0
        };
    }
}

// Error Classes
class ContentGenerationError extends Error {
    constructor(message, originalError) {
        super(message);
        this.name = 'ContentGenerationError';
        this.originalError = originalError;
    }
}

class TranslationError extends Error {
    constructor(message, originalError) {
        super(message);
        this.name = 'TranslationError';
        this.originalError = originalError;
    }
}

class AnalysisError extends Error {
    constructor(message, originalError) {
        super(message);
        this.name = 'AnalysisError';
        this.originalError = originalError;
    }
}

// Usage Example
const platform = new AIContentPlatform({
    openai: 'your-openai-key',
    googleCloud: 'your-google-key',
    projectId: 'your-project-id',
    azure: {
        key: 'your-azure-key',
        endpoint: 'your-azure-endpoint',
        region: 'eastus'
    }
});

// Generate comprehensive article
async function createComprehensiveArticle() {
    try {
        const article = await platform.createArticle(
            'Inteligência Artificial na Programação Web',
            'technical',
            'long'
        );
        
        console.log('Article generated:', article.title);
        console.log('Quality analysis:', await platform.analyzeContentQuality(
            article.sections.map(s => s.content).join('\n\n')
        ));
        
        return article;
    } catch (error) {
        console.error('Article generation failed:', error);
    }
}
```

## Boas Práticas e Otimização

### **Estratégias de Implementação**:

1. **Gerenciamento de Chaves API**:
```javascript
class APIKeyManager {
    constructor() {
        this.keys = new Map();
        this.rotationSchedule = new Map();
    }
    
    addKey(service, key, config = {}) {
        this.keys.set(service, {
            key,
            usage: 0,
            rateLimit: config.rateLimit || 1000,
            cost: config.cost || 0,
            lastUsed: null
        });
    }
    
    getKey(service) {
        const keyData = this.keys.get(service);
        if (!keyData) throw new Error(`No key found for ${service}`);
        
        keyData.usage++;
        keyData.lastUsed = new Date();
        
        return keyData.key;
    }
    
    trackUsage(service, tokens, cost) {
        const keyData = this.keys.get(service);
        if (keyData) {
            keyData.cost += cost;
            keyData.usage += tokens;
        }
    }
    
    getUsageReport() {
        return Array.from(this.keys.entries()).map(([service, data]) => ({
            service,
            usage: data.usage,
            cost: data.cost,
            lastUsed: data.lastUsed
        }));
    }
}
```

2. **Cache Inteligente**:
```javascript
class IntelligentCache {
    constructor(maxSize = 100) {
        this.cache = new Map();
        this.maxSize = maxSize;
        this.accessTimes = new Map();
    }
    
    set(key, value, ttl = 3600000) { // 1 hour default TTL
        if (this.cache.size >= this.maxSize) {
            this.evictLRU();
        }
        
        this.cache.set(key, {
            value,
            expires: Date.now() + ttl,
            hits: 0
        });
        
        this.accessTimes.set(key, Date.now());
    }
    
    get(key) {
        const item = this.cache.get(key);
        
        if (!item) return null;
        
        if (Date.now() > item.expires) {
            this.cache.delete(key);
            this.accessTimes.delete(key);
            return null;
        }
        
        item.hits++;
        this.accessTimes.set(key, Date.now());
        return item.value;
    }
    
    evictLRU() {
        let oldestKey = null;
        let oldestTime = Date.now();
        
        for (const [key, time] of this.accessTimes) {
            if (time < oldestTime) {
                oldestTime = time;
                oldestKey = key;
            }
        }
        
        if (oldestKey) {
            this.cache.delete(oldestKey);
            this.accessTimes.delete(oldestKey);
        }
    }
}
```

## Considerações de Segurança e Compliance

### **Implementação de Segurança**:

```javascript
class AISecurityManager {
    constructor() {
        this.sensitivePatterns = [
            /\b\d{3}-\d{2}-\d{4}\b/, // SSN
            /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/, // Credit card
            /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/ // Email
        ];
    }
    
    sanitizeInput(input) {
        // Remove sensitive information
        let sanitized = input;
        
        this.sensitivePatterns.forEach(pattern => {
            sanitized = sanitized.replace(pattern, '[REDACTED]');
        });
        
        // Remove potential injection attempts
        sanitized = sanitized.replace(/<script[^>]*>.*?<\/script>/gi, '');
        sanitized = sanitized.replace(/javascript:/gi, '');
        
        return sanitized;
    }
    
    validateAPIResponse(response, expectedStructure) {
        // Validate response structure to prevent injection
        if (typeof response !== 'object') return false;
        
        for (const key of expectedStructure) {
            if (!(key in response)) return false;
        }
        
        return true;
    }
    
    encryptAPIKey(key) {
        // Basic encryption for client-side storage
        return btoa(key);
    }
    
    decryptAPIKey(encryptedKey) {
        try {
            return atob(encryptedKey);
        } catch (e) {
            throw new Error('Invalid API key format');
        }
    }
}
```

## Monitoramento e Analytics

### **Sistema de Monitoramento**:

```javascript
class AIPerformanceMonitor {
    constructor() {
        this.metrics = {
            requestCount: 0,
            averageResponseTime: 0,
            errorRate: 0,
            tokensConsumed: 0,
            costAccumulated: 0
        };
        
        this.responseTimes = [];
        this.errors = [];
    }
    
    recordRequest(startTime, endTime, tokens, cost, error = null) {
        this.metrics.requestCount++;
        
        const responseTime = endTime - startTime;
        this.responseTimes.push(responseTime);
        
        // Update average (rolling)
        this.metrics.averageResponseTime = 
            this.responseTimes.slice(-100).reduce((a, b) => a + b, 0) / 
            Math.min(this.responseTimes.length, 100);
        
        if (error) {
            this.errors.push({
                timestamp: new Date(),
                error: error.message,
                type: error.constructor.name
            });
        }
        
        this.metrics.errorRate = this.errors.length / this.metrics.requestCount;
        this.metrics.tokensConsumed += tokens || 0;
        this.metrics.costAccumulated += cost || 0;
    }
    
    getHealthStatus() {
        return {
            status: this.metrics.errorRate < 0.05 ? 'healthy' : 'degraded',
            metrics: this.metrics,
            recommendations: this.generateRecommendations()
        };
    }
    
    generateRecommendations() {
        const recommendations = [];
        
        if (this.metrics.averageResponseTime > 5000) {
            recommendations.push('Consider implementing caching to improve response times');
        }
        
        if (this.metrics.errorRate > 0.1) {
            recommendations.push('High error rate detected - review error logs');
        }
        
        if (this.metrics.costAccumulated > 100) {
            recommendations.push('High API costs - consider optimizing token usage');
        }
        
        return recommendations;
    }
}
```

## Conclusão

APIs de Inteligência Artificial representam uma revolução no desenvolvimento web, oferecendo:

### **Vantagens Estratégicas**:
- **Acesso a Tecnologia Avançada**: Modelos de ponta sem investimento em infraestrutura
- **Desenvolvimento Acelerado**: Implementação rápida de funcionalidades complexas
- **Escalabilidade**: Capacidade de processamento sob demanda
- **Custo-Efetividade**: Pagamento por uso versus desenvolvimento interno

### **Considerações Importantes**:
- **Dependência Externa**: Riscos de interrupção de serviço
- **Custos Variáveis**: Necessidade de monitoramento financeiro
- **Privacidade de Dados**: Cuidados com informações sensíveis
- **Latência de Rede**: Impacto na experiência do usuário

### **Futuro das APIs de IA**:
- **Modelos Especializados**: APIs focadas em domínios específicos
- **Edge AI**: Processamento local com menor latência
- **Integração Nativa**: APIs embutidas em navegadores
- **Customização Avançada**: Fine-tuning de modelos via API

O domínio das APIs de IA é essencial para desenvolvedores modernos, permitindo a criação de aplicações web inteligentes e competitivas no mercado atual.
