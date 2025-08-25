# Desenvolvimento Full-Stack com IA: Guia Completo

## Arquitetura Moderna de Aplicações IA

### **Stack Tecnológico Essencial**

#### Frontend Inteligente
```javascript
// React + TypeScript + IA Integration
import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

const AIApp: React.FC = () => {
    const [model, setModel] = useState<tf.LayersModel | null>(null);
    const [prediction, setPrediction] = useState<string>('');
    
    useEffect(() => {
        loadModel();
    }, []);
    
    const loadModel = async () => {
        try {
            const loadedModel = await tf.loadLayersModel('/models/classifier.json');
            setModel(loadedModel);
        } catch (error) {
            console.error('Erro ao carregar modelo:', error);
        }
    };
    
    const predict = async (inputData: number[]) => {
        if (!model) return;
        
        const tensor = tf.tensor2d([inputData]);
        const prediction = model.predict(tensor) as tf.Tensor;
        const result = await prediction.data();
        
        setPrediction(result[0] > 0.5 ? 'Positivo' : 'Negativo');
        
        tensor.dispose();
        prediction.dispose();
    };
    
    return (
        <div>
            <h2>Classificador IA</h2>
            <p>Predição: {prediction}</p>
        </div>
    );
};
```

#### Backend com IA
```python
# FastAPI + ML Pipeline
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict
import asyncio
import redis
import json

app = FastAPI(title="AI Backend API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache Redis
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Modelo de Classificação
class TextClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-multilingual-cased')
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# Instância global do modelo
model = None
tokenizer = None

@app.on_event("startup")
async def load_models():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = TextClassifier()
    model.load_state_dict(torch.load('models/classifier.pth', map_location='cpu'))
    model.eval()

@app.post("/api/classify-text")
async def classify_text(data: Dict[str, str]):
    text = data.get("text", "")
    
    # Verificar cache
    cache_key = f"classification:{hash(text)}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    # Tokenizar
    inputs = tokenizer(
        text,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Predição
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities.max().item()
    
    result = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "labels": ["Negativo", "Neutro", "Positivo"]
    }
    
    # Salvar no cache
    redis_client.setex(cache_key, 3600, json.dumps(result))
    
    return result

@app.post("/api/generate-content")
async def generate_content(data: Dict[str, str]):
    prompt = data.get("prompt", "")
    
    # Simular geração (integrar com OpenAI/outros)
    generated_text = f"Conteúdo gerado baseado em: {prompt}"
    
    return {"generated_text": generated_text}

# WebSocket para comunicação em tempo real
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Processar dados em tempo real
            response = {"message": f"Processado: {data}"}
            await websocket.send_text(json.dumps(response))
    except:
        pass
```

## Integração Frontend-Backend

### **Estado Global com Context API**
```javascript
// AIContext.tsx
import React, { createContext, useContext, useReducer } from 'react';

interface AIState {
    models: Record<string, any>;
    predictions: any[];
    loading: boolean;
    error: string | null;
}

interface AIAction {
    type: string;
    payload?: any;
}

const initialState: AIState = {
    models: {},
    predictions: [],
    loading: false,
    error: null
};

const aiReducer = (state: AIState, action: AIAction): AIState => {
    switch (action.type) {
        case 'SET_LOADING':
            return { ...state, loading: action.payload };
        case 'SET_ERROR':
            return { ...state, error: action.payload, loading: false };
        case 'ADD_PREDICTION':
            return { 
                ...state, 
                predictions: [...state.predictions, action.payload],
                loading: false 
            };
        case 'LOAD_MODEL':
            return {
                ...state,
                models: { ...state.models, [action.payload.name]: action.payload.model }
            };
        default:
            return state;
    }
};

const AIContext = createContext<{
    state: AIState;
    dispatch: React.Dispatch<AIAction>;
} | null>(null);

export const AIProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [state, dispatch] = useReducer(aiReducer, initialState);
    
    return (
        <AIContext.Provider value={{ state, dispatch }}>
            {children}
        </AIContext.Provider>
    );
};

export const useAI = () => {
    const context = useContext(AIContext);
    if (!context) {
        throw new Error('useAI deve ser usado dentro de AIProvider');
    }
    return context;
};
```

### **Serviços de API**
```javascript
// services/aiService.ts
class AIService {
    private baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    private wsConnection: WebSocket | null = null;
    
    async classifyText(text: string): Promise<any> {
        try {
            const response = await fetch(`${this.baseURL}/api/classify-text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Erro na classificação:', error);
            throw error;
        }
    }
    
    async generateContent(prompt: string): Promise<any> {
        const response = await fetch(`${this.baseURL}/api/generate-content`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });
        
        return await response.json();
    }
    
    connectWebSocket(onMessage: (data: any) => void): void {
        if (this.wsConnection) return;
        
        this.wsConnection = new WebSocket(`ws://localhost:8000/ws`);
        
        this.wsConnection.onopen = () => {
            console.log('WebSocket conectado');
        };
        
        this.wsConnection.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };
        
        this.wsConnection.onerror = (error) => {
            console.error('Erro WebSocket:', error);
        };
    }
    
    sendWebSocketMessage(message: any): void {
        if (this.wsConnection?.readyState === WebSocket.OPEN) {
            this.wsConnection.send(JSON.stringify(message));
        }
    }
}

export const aiService = new AIService();
```

## Deploy e DevOps

### **Docker Configuration**
```dockerfile
# Dockerfile - Backend
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# Dockerfile - Frontend
FROM node:16-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### **Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://localhost:8000

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - postgres
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/aidb
      - REDIS_URL=redis://redis:6379

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=aidb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## Monitoramento e Analytics

### **Sistema de Métricas**
```python
# monitoring.py
from prometheus_client import Counter, Histogram, generate_latest
import time
import functools

# Métricas
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total model predictions', ['model_name'])

def monitor_api(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            REQUEST_COUNT.labels(method='POST', endpoint=func.__name__).inc()
            return result
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    
    return wrapper

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## Boas Práticas de Segurança

### **Autenticação JWT**
```python
# auth.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import os

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
```

## Otimização de Performance

### **Cache Strategy**
```javascript
// cacheManager.js
class CacheManager {
    constructor() {
        this.cache = new Map();
        this.expiry = new Map();
    }
    
    set(key, value, ttl = 300000) { // 5 minutos padrão
        this.cache.set(key, value);
        this.expiry.set(key, Date.now() + ttl);
    }
    
    get(key) {
        if (this.expiry.get(key) < Date.now()) {
            this.cache.delete(key);
            this.expiry.delete(key);
            return null;
        }
        return this.cache.get(key);
    }
    
    clear() {
        this.cache.clear();
        this.expiry.clear();
    }
}

export const cacheManager = new CacheManager();
```

## Conclusão

Este guia apresenta uma arquitetura completa para desenvolvimento full-stack com IA, incluindo:

- **Frontend Reativo**: React com TypeScript e TensorFlow.js
- **Backend Escalável**: FastAPI com modelos ML integrados  
- **Infraestrutura**: Docker, Redis, PostgreSQL
- **Monitoramento**: Prometheus e métricas customizadas
- **Segurança**: JWT, validação e sanitização
- **Performance**: Cache inteligente e otimizações

A combinação dessas tecnologias permite criar aplicações web robustas e inteligentes, prontas para produção.
