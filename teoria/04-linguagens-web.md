# Linguagens Web: Fundamentos e Evolução Tecnológica

## Introdução às Linguagens Web

As linguagens web constituem o ecossistema fundamental para desenvolvimento de aplicações modernas, abrangendo desde interfaces de usuário até processamento de dados em servidor. Esta análise técnica examina a evolução, arquiteturas e implementações avançadas das principais linguagens web.

## Fundamentos da Web

### Arquitetura Cliente-Servidor

```javascript
// Modelo de comunicação HTTP/HTTPS
class WebArchitecture {
    constructor() {
        this.clientSide = {
            languages: ['HTML', 'CSS', 'JavaScript', 'TypeScript'],
            frameworks: ['React', 'Vue', 'Angular', 'Svelte'],
            buildTools: ['Webpack', 'Vite', 'Parcel', 'Rollup']
        };
        
        this.serverSide = {
            languages: ['JavaScript', 'Python', 'PHP', 'Ruby', 'Java', 'C#', 'Go', 'Rust'],
            frameworks: ['Express', 'FastAPI', 'Laravel', 'Rails', 'Spring', 'ASP.NET'],
            databases: ['MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'Elasticsearch']
        };
        
        this.protocols = ['HTTP/1.1', 'HTTP/2', 'HTTP/3', 'WebSocket', 'GraphQL'];
    }
    
    establishConnection(client, server) {
        return new Promise((resolve, reject) => {
            // Simulação de handshake TCP/IP
            const connection = {
                protocol: 'HTTPS',
                version: 'HTTP/2',
                encryption: 'TLS 1.3',
                compression: 'gzip, br',
                keepAlive: true,
                maxConnections: 100
            };
            
            if (this.validateConnection(client, server)) {
                resolve(connection);
            } else {
                reject(new Error('Connection failed'));
            }
        });
    }
    
    validateConnection(client, server) {
        return client.supports(server.protocol) && 
               server.accepts(client.userAgent);
    }
}
```

## HTML5: Estrutura Semântica Avançada

### Elementos Semânticos e Acessibilidade

```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Aplicação web com IA para programação">
    <meta name="keywords" content="IA, programação, web development, machine learning">
    
    <!-- Preload Critical Resources -->
    <link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>
    <link rel="preload" href="/js/critical.js" as="script">
    
    <!-- Progressive Enhancement -->
    <link rel="stylesheet" href="/css/base.css">
    <link rel="stylesheet" href="/css/enhanced.css" media="(min-width: 768px)">
    
    <!-- Service Worker Registration -->
    <script>
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js');
        }
    </script>
    
    <title>Laboratório de Transformação Digital - IA para Web</title>
</head>
<body>
    <!-- Skip Navigation for Accessibility -->
    <a href="#main-content" class="skip-nav">Pular para conteúdo principal</a>
    
    <header role="banner">
        <nav role="navigation" aria-label="Navegação principal">
            <ul>
                <li><a href="#teoria" aria-describedby="teoria-desc">Teoria</a></li>
                <li><a href="#pratica" aria-describedby="pratica-desc">Prática</a></li>
                <li><a href="#projetos" aria-describedby="projetos-desc">Projetos</a></li>
            </ul>
        </nav>
    </header>
    
    <main id="main-content" role="main">
        <article>
            <header>
                <h1>Agentes de IA para Desenvolvimento Web</h1>
                <p class="subtitle">Explorando o futuro da programação automatizada</p>
            </header>
            
            <section aria-labelledby="intro-heading">
                <h2 id="intro-heading">Introdução</h2>
                <p>Content here...</p>
            </section>
            
            <!-- Web Components Integration -->
            <ai-code-assistant 
                model="gpt-4"
                language="javascript"
                context="web-development">
            </ai-code-assistant>
            
            <!-- Progressive Web App Features -->
            <section class="offline-content" hidden>
                <h2>Conteúdo Offline</h2>
                <p>Este conteúdo está disponível offline.</p>
            </section>
        </article>
    </main>
    
    <aside role="complementary">
        <section aria-labelledby="related-heading">
            <h2 id="related-heading">Recursos Relacionados</h2>
            <ul>
                <li><a href="/docs/api">Documentação da API</a></li>
                <li><a href="/examples">Exemplos de Código</a></li>
            </ul>
        </section>
    </aside>
    
    <footer role="contentinfo">
        <p>&copy; 2025 Laboratório de Transformação Digital</p>
        <nav aria-label="Links do rodapé">
            <a href="/privacy">Privacidade</a>
            <a href="/terms">Termos</a>
            <a href="/contact">Contato</a>
        </nav>
    </footer>
    
    <!-- Critical JavaScript -->
    <script src="/js/app.js" async></script>
    
    <!-- Analytics with Privacy -->
    <script>
        // Privacy-focused analytics
        if (navigator.doNotTrack !== "1") {
            // Load analytics only if user hasn't opted out
            loadAnalytics();
        }
    </script>
</body>
</html>
```

### APIs Web Modernas

```javascript
// Service Worker para PWA
class ServiceWorkerManager {
    constructor() {
        this.cacheName = 'ltd-v1.0.0';
        this.staticAssets = [
            '/',
            '/css/app.css',
            '/js/app.js',
            '/images/logo.svg'
        ];
    }
    
    async install() {
        const cache = await caches.open(this.cacheName);
        return cache.addAll(this.staticAssets);
    }
    
    async fetch(request) {
        // Cache First Strategy for static assets
        if (this.isStaticAsset(request.url)) {
            return this.cacheFirst(request);
        }
        
        // Network First for API calls
        if (this.isAPICall(request.url)) {
            return this.networkFirst(request);
        }
        
        // Stale While Revalidate for content
        return this.staleWhileRevalidate(request);
    }
    
    async cacheFirst(request) {
        const cached = await caches.match(request);
        return cached || fetch(request);
    }
    
    async networkFirst(request) {
        try {
            const response = await fetch(request);
            if (response.ok) {
                const cache = await caches.open(this.cacheName);
                cache.put(request, response.clone());
            }
            return response;
        } catch (error) {
            return caches.match(request) || new Response('Offline', { status: 503 });
        }
    }
}

// Web Workers para processamento intensivo
class AIWorkerManager {
    constructor() {
        this.workers = new Map();
        this.taskQueue = [];
    }
    
    createWorker(type) {
        const workerCode = this.getWorkerCode(type);
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        const worker = new Worker(URL.createObjectURL(blob));
        
        worker.onmessage = (event) => this.handleWorkerMessage(event, type);
        worker.onerror = (error) => this.handleWorkerError(error, type);
        
        this.workers.set(type, worker);
        return worker;
    }
    
    getWorkerCode(type) {
        switch (type) {
            case 'ml':
                return `
                    importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');
                    
                    self.onmessage = async function(e) {
                        const { data, action } = e.data;
                        
                        switch (action) {
                            case 'predict':
                                const model = await tf.loadLayersModel(data.modelUrl);
                                const prediction = model.predict(tf.tensor(data.input));
                                const result = await prediction.data();
                                
                                self.postMessage({
                                    type: 'prediction',
                                    result: Array.from(result)
                                });
                                break;
                                
                            case 'train':
                                // Training logic here
                                break;
                        }
                    };
                `;
                
            case 'crypto':
                return `
                    self.onmessage = async function(e) {
                        const { data, action } = e.data;
                        
                        if (action === 'hash') {
                            const encoder = new TextEncoder();
                            const data = encoder.encode(e.data.text);
                            const hashBuffer = await crypto.subtle.digest('SHA-256', data);
                            const hashArray = Array.from(new Uint8Array(hashBuffer));
                            const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
                            
                            self.postMessage({ type: 'hash', result: hashHex });
                        }
                    };
                `;
                
            default:
                return `self.onmessage = function(e) { self.postMessage(e.data); };`;
        }
    }
    
    async processTask(type, data) {
        return new Promise((resolve, reject) => {
            let worker = this.workers.get(type);
            
            if (!worker) {
                worker = this.createWorker(type);
            }
            
            const taskId = Date.now() + Math.random();
            
            const messageHandler = (event) => {
                if (event.data.taskId === taskId) {
                    worker.removeEventListener('message', messageHandler);
                    resolve(event.data.result);
                }
            };
            
            worker.addEventListener('message', messageHandler);
            worker.postMessage({ ...data, taskId });
            
            // Timeout protection
            setTimeout(() => {
                worker.removeEventListener('message', messageHandler);
                reject(new Error('Worker timeout'));
            }, 30000);
        });
    }
}
```

## CSS3: Estilização Avançada e Performance

### Layout Moderno com Grid e Flexbox

```css
/* CSS Custom Properties (Variables) */
:root {
    /* Color Palette */
    --primary-hue: 210;
    --primary-sat: 100%;
    --primary-light: 50%;
    
    --primary-50: hsl(var(--primary-hue), var(--primary-sat), 95%);
    --primary-100: hsl(var(--primary-hue), var(--primary-sat), 90%);
    --primary-500: hsl(var(--primary-hue), var(--primary-sat), var(--primary-light));
    --primary-900: hsl(var(--primary-hue), var(--primary-sat), 10%);
    
    /* Typography Scale */
    --font-size-xs: clamp(0.75rem, 0.7rem + 0.25vw, 0.875rem);
    --font-size-sm: clamp(0.875rem, 0.8rem + 0.375vw, 1rem);
    --font-size-base: clamp(1rem, 0.9rem + 0.5vw, 1.125rem);
    --font-size-lg: clamp(1.125rem, 1rem + 0.625vw, 1.25rem);
    --font-size-xl: clamp(1.25rem, 1.1rem + 0.75vw, 1.5rem);
    
    /* Spacing Scale */
    --space-xs: clamp(0.25rem, 0.2rem + 0.25vw, 0.375rem);
    --space-sm: clamp(0.5rem, 0.4rem + 0.5vw, 0.75rem);
    --space-md: clamp(1rem, 0.8rem + 1vw, 1.5rem);
    --space-lg: clamp(1.5rem, 1.2rem + 1.5vw, 2.25rem);
    --space-xl: clamp(2rem, 1.6rem + 2vw, 3rem);
    
    /* Animation */
    --transition-fast: 150ms ease-out;
    --transition-normal: 300ms ease-out;
    --transition-slow: 500ms ease-out;
    
    /* Shadows */
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
}

/* Dark Mode Support */
@media (prefers-color-scheme: dark) {
    :root {
        --primary-light: 70%;
        --bg-primary: hsl(var(--primary-hue), 20%, 10%);
        --text-primary: hsl(var(--primary-hue), 20%, 90%);
    }
}

/* Modern CSS Grid Layout */
.app-layout {
    display: grid;
    grid-template-areas: 
        "header header"
        "sidebar main"
        "footer footer";
    grid-template-columns: minmax(250px, 1fr) 4fr;
    grid-template-rows: auto 1fr auto;
    min-height: 100vh;
    gap: var(--space-md);
}

@media (max-width: 768px) {
    .app-layout {
        grid-template-areas: 
            "header"
            "main"
            "sidebar"
            "footer";
        grid-template-columns: 1fr;
    }
}

/* Advanced Flexbox Patterns */
.flex-container {
    display: flex;
    flex-direction: column;
    gap: var(--space-md);
}

.flex-container > .header {
    flex: 0 0 auto; /* Don't grow or shrink */
}

.flex-container > .content {
    flex: 1 1 auto; /* Grow and shrink */
    overflow-y: auto;
}

.flex-container > .footer {
    flex: 0 0 auto;
}

/* Container Queries (Quando suportado) */
@container (min-width: 400px) {
    .card {
        display: flex;
        flex-direction: row;
    }
}

/* CSS Logical Properties */
.content-block {
    padding-block: var(--space-md);
    padding-inline: var(--space-lg);
    margin-block-end: var(--space-lg);
    border-inline-start: 4px solid var(--primary-500);
}

/* Modern CSS Features */
.glassmorphism {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
}

.gradient-text {
    background: linear-gradient(45deg, var(--primary-500), var(--primary-900));
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
    font-weight: bold;
}

/* CSS Animations with Performance */
@keyframes slideInUp {
    from {
        transform: translate3d(0, 100%, 0);
        opacity: 0;
    }
    to {
        transform: translate3d(0, 0, 0);
        opacity: 1;
    }
}

.animate-slide-in {
    animation: slideInUp var(--transition-normal) ease-out;
    will-change: transform, opacity;
}

/* Scroll Snap */
.scroll-container {
    scroll-snap-type: y mandatory;
    overflow-y: scroll;
    height: 100vh;
}

.scroll-section {
    scroll-snap-align: start;
    height: 100vh;
}

/* CSS Grid Advanced Patterns */
.masonry-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    grid-auto-rows: masonry; /* Future feature */
    gap: var(--space-md);
}

/* Fallback for browsers without masonry */
@supports not (grid-auto-rows: masonry) {
    .masonry-grid {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-md);
    }
    
    .masonry-grid > * {
        flex: 1 0 300px;
    }
}

/* CSS Custom Selectors (Future) */
@custom-selector :--heading h1, h2, h3, h4, h5, h6;

:--heading {
    font-family: 'Inter', system-ui, sans-serif;
    font-weight: 600;
    line-height: 1.2;
    color: var(--text-primary);
}
```

### CSS-in-JS e Styled Components

```javascript
// Styled Components com TypeScript
import styled, { createGlobalStyle, ThemeProvider } from 'styled-components';

// Global Styles
const GlobalStyle = createGlobalStyle`
    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }
    
    body {
        font-family: 'Inter', system-ui, sans-serif;
        line-height: 1.6;
        color: ${props => props.theme.colors.text.primary};
        background: ${props => props.theme.colors.background.primary};
        font-feature-settings: 'liga' 1, 'kern' 1;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
`;

// Theme Definition
const theme = {
    colors: {
        primary: {
            50: '#f0f9ff',
            100: '#e0f2fe',
            500: '#0ea5e9',
            900: '#0c4a6e'
        },
        text: {
            primary: '#1f2937',
            secondary: '#6b7280'
        },
        background: {
            primary: '#ffffff',
            secondary: '#f9fafb'
        }
    },
    spacing: {
        xs: '0.25rem',
        sm: '0.5rem',
        md: '1rem',
        lg: '1.5rem',
        xl: '2rem'
    },
    breakpoints: {
        sm: '640px',
        md: '768px',
        lg: '1024px',
        xl: '1280px'
    },
    transitions: {
        fast: '150ms ease-out',
        normal: '300ms ease-out',
        slow: '500ms ease-out'
    }
};

// Styled Components
const Container = styled.div`
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 ${props => props.theme.spacing.md};
    
    @media (min-width: ${props => props.theme.breakpoints.lg}) {
        padding: 0 ${props => props.theme.spacing.xl};
    }
`;

const Button = styled.button`
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: ${props => props.theme.spacing.sm};
    
    padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
    border: none;
    border-radius: 8px;
    
    font-size: 1rem;
    font-weight: 500;
    line-height: 1.5;
    
    background: ${props => props.variant === 'primary' 
        ? props.theme.colors.primary[500] 
        : 'transparent'};
    color: ${props => props.variant === 'primary' 
        ? 'white' 
        : props.theme.colors.primary[500]};
    
    border: 2px solid ${props => props.theme.colors.primary[500]};
    
    cursor: pointer;
    transition: all ${props => props.theme.transitions.fast};
    
    &:hover {
        background: ${props => props.variant === 'primary' 
            ? props.theme.colors.primary[600] 
            : props.theme.colors.primary[50]};
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    &:active {
        transform: translateY(0);
    }
    
    &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }
`;

const Card = styled.div`
    background: ${props => props.theme.colors.background.primary};
    border-radius: 12px;
    padding: ${props => props.theme.spacing.lg};
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all ${props => props.theme.transitions.normal};
    
    &:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
`;

// Usage in React Component
const App = () => {
    return (
        <ThemeProvider theme={theme}>
            <GlobalStyle />
            <Container>
                <Card>
                    <h1>Laboratório de Transformação Digital</h1>
                    <p>Explorando IA para desenvolvimento web</p>
                    <Button variant="primary">
                        Começar Agora
                    </Button>
                </Card>
            </Container>
        </ThemeProvider>
    );
};
```

## JavaScript/TypeScript: Programação Avançada

### TypeScript para Desenvolvimento Robusto

```typescript
// Type System Avançado
interface APIResponse<T> {
    data: T;
    status: 'success' | 'error';
    message?: string;
    timestamp: Date;
}

type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

type ExtractArrayType<T> = T extends (infer U)[] ? U : never;

// Generic Utility Types
type NonNullable<T> = T extends null | undefined ? never : T;
type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// Advanced Function Types
type AsyncFunction<T extends any[], R> = (...args: T) => Promise<R>;
type EventHandler<E extends Event = Event> = (event: E) => void;

// Conditional Types
type APIEndpoint<T> = T extends 'user' 
    ? '/api/users' 
    : T extends 'post' 
    ? '/api/posts' 
    : '/api/generic';

// Classes com Decorators
class APIClient {
    private baseURL: string;
    private cache = new Map<string, { data: any; expiry: number }>();
    
    constructor(baseURL: string) {
        this.baseURL = baseURL;
    }
    
    @cached(300000) // 5 minutes cache
    @retryOnFailure(3)
    async get<T>(endpoint: string): Promise<APIResponse<T>> {
        const response = await fetch(`${this.baseURL}${endpoint}`);
        
        if (!response.ok) {
            throw new APIError(`Request failed: ${response.status}`, response.status);
        }
        
        const data = await response.json();
        
        return {
            data,
            status: 'success',
            timestamp: new Date()
        };
    }
    
    @validateInput
    async post<T, U>(endpoint: string, payload: T): Promise<APIResponse<U>> {
        const response = await fetch(`${this.baseURL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        return {
            data,
            status: response.ok ? 'success' : 'error',
            message: response.ok ? undefined : data.message,
            timestamp: new Date()
        };
    }
}

// Decorators Implementation
function cached(duration: number) {
    return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
        const originalMethod = descriptor.value;
        
        descriptor.value = async function (...args: any[]) {
            const cacheKey = `${propertyKey}_${JSON.stringify(args)}`;
            const cached = this.cache.get(cacheKey);
            
            if (cached && Date.now() < cached.expiry) {
                return cached.data;
            }
            
            const result = await originalMethod.apply(this, args);
            
            this.cache.set(cacheKey, {
                data: result,
                expiry: Date.now() + duration
            });
            
            return result;
        };
        
        return descriptor;
    };
}

function retryOnFailure(maxAttempts: number) {
    return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
        const originalMethod = descriptor.value;
        
        descriptor.value = async function (...args: any[]) {
            let lastError: Error;
            
            for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                try {
                    return await originalMethod.apply(this, args);
                } catch (error) {
                    lastError = error as Error;
                    
                    if (attempt === maxAttempts) {
                        throw lastError;
                    }
                    
                    // Exponential backoff
                    await new Promise(resolve => 
                        setTimeout(resolve, Math.pow(2, attempt) * 1000)
                    );
                }
            }
            
            throw lastError!;
        };
        
        return descriptor;
    };
}

// Custom Error Classes
class APIError extends Error {
    constructor(message: string, public statusCode: number) {
        super(message);
        this.name = 'APIError';
    }
}

// Advanced Async Patterns
class TaskQueue {
    private queue: Array<() => Promise<any>> = [];
    private running = false;
    private concurrency: number;
    private activeCount = 0;
    
    constructor(concurrency: number = 3) {
        this.concurrency = concurrency;
    }
    
    async add<T>(task: () => Promise<T>): Promise<T> {
        return new Promise((resolve, reject) => {
            this.queue.push(async () => {
                try {
                    const result = await task();
                    resolve(result);
                } catch (error) {
                    reject(error);
                }
            });
            
            this.process();
        });
    }
    
    private async process() {
        if (this.running || this.activeCount >= this.concurrency) {
            return;
        }
        
        const task = this.queue.shift();
        if (!task) {
            return;
        }
        
        this.activeCount++;
        this.running = true;
        
        try {
            await task();
        } finally {
            this.activeCount--;
            this.running = false;
            
            // Process next task
            if (this.queue.length > 0) {
                setImmediate(() => this.process());
            }
        }
    }
}

// State Management with Observables
class Observable<T> {
    private subscribers: Set<(value: T) => void> = new Set();
    private _value: T;
    
    constructor(initialValue: T) {
        this._value = initialValue;
    }
    
    get value(): T {
        return this._value;
    }
    
    set value(newValue: T) {
        if (newValue !== this._value) {
            this._value = newValue;
            this.notify();
        }
    }
    
    subscribe(callback: (value: T) => void): () => void {
        this.subscribers.add(callback);
        
        // Return unsubscribe function
        return () => {
            this.subscribers.delete(callback);
        };
    }
    
    private notify() {
        this.subscribers.forEach(callback => callback(this._value));
    }
    
    map<U>(mapper: (value: T) => U): Observable<U> {
        const mapped = new Observable(mapper(this._value));
        
        this.subscribe(value => {
            mapped.value = mapper(value);
        });
        
        return mapped;
    }
    
    filter(predicate: (value: T) => boolean): Observable<T> {
        const filtered = new Observable(this._value);
        
        this.subscribe(value => {
            if (predicate(value)) {
                filtered.value = value;
            }
        });
        
        return filtered;
    }
}

// Usage Example
const userStore = new Observable<User | null>(null);
const isLoggedIn = userStore.map(user => user !== null);

isLoggedIn.subscribe(loggedIn => {
    if (loggedIn) {
        console.log('User logged in');
    } else {
        console.log('User logged out');
    }
});
```

### Padrões de Design Avançados

```javascript
// Module Pattern com ES6
class EventEmitter {
    constructor() {
        this.events = new Map();
    }
    
    on(event, callback) {
        if (!this.events.has(event)) {
            this.events.set(event, new Set());
        }
        this.events.get(event).add(callback);
        
        // Return unsubscribe function
        return () => this.off(event, callback);
    }
    
    off(event, callback) {
        if (this.events.has(event)) {
            this.events.get(event).delete(callback);
        }
    }
    
    emit(event, ...args) {
        if (this.events.has(event)) {
            this.events.get(event).forEach(callback => {
                try {
                    callback(...args);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }
    
    once(event, callback) {
        const wrapper = (...args) => {
            callback(...args);
            this.off(event, wrapper);
        };
        
        this.on(event, wrapper);
    }
}

// Singleton Pattern
class ConfigManager {
    constructor() {
        if (ConfigManager.instance) {
            return ConfigManager.instance;
        }
        
        this.config = new Map();
        ConfigManager.instance = this;
    }
    
    set(key, value) {
        this.config.set(key, value);
    }
    
    get(key, defaultValue = null) {
        return this.config.get(key) ?? defaultValue;
    }
    
    has(key) {
        return this.config.has(key);
    }
}

// Observer Pattern
class Store {
    constructor(initialState = {}) {
        this.state = initialState;
        this.observers = new Set();
        this.middleware = [];
    }
    
    subscribe(observer) {
        this.observers.add(observer);
        
        return () => {
            this.observers.delete(observer);
        };
    }
    
    dispatch(action) {
        // Apply middleware
        let processedAction = action;
        
        for (const middleware of this.middleware) {
            processedAction = middleware(processedAction, this.state);
        }
        
        // Update state
        const newState = this.reducer(this.state, processedAction);
        
        if (newState !== this.state) {
            this.state = newState;
            this.notify();
        }
    }
    
    use(middleware) {
        this.middleware.push(middleware);
    }
    
    reducer(state, action) {
        // Override in subclasses
        return state;
    }
    
    notify() {
        this.observers.forEach(observer => {
            try {
                observer(this.state);
            } catch (error) {
                console.error('Error in observer:', error);
            }
        });
    }
}

// Factory Pattern
class ComponentFactory {
    static components = new Map();
    
    static register(name, component) {
        this.components.set(name, component);
    }
    
    static create(name, props = {}) {
        const Component = this.components.get(name);
        
        if (!Component) {
            throw new Error(`Component ${name} not found`);
        }
        
        return new Component(props);
    }
    
    static list() {
        return Array.from(this.components.keys());
    }
}

// Command Pattern
class CommandManager {
    constructor() {
        this.history = [];
        this.currentIndex = -1;
    }
    
    execute(command) {
        // Remove any commands after current index
        this.history = this.history.slice(0, this.currentIndex + 1);
        
        // Execute command
        command.execute();
        
        // Add to history
        this.history.push(command);
        this.currentIndex++;
    }
    
    undo() {
        if (this.canUndo()) {
            const command = this.history[this.currentIndex];
            command.undo();
            this.currentIndex--;
            return true;
        }
        return false;
    }
    
    redo() {
        if (this.canRedo()) {
            this.currentIndex++;
            const command = this.history[this.currentIndex];
            command.execute();
            return true;
        }
        return false;
    }
    
    canUndo() {
        return this.currentIndex >= 0;
    }
    
    canRedo() {
        return this.currentIndex < this.history.length - 1;
    }
}

// Strategy Pattern
class ValidationStrategy {
    validate(value) {
        throw new Error('validate method must be implemented');
    }
}

class EmailValidation extends ValidationStrategy {
    validate(value) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return {
            isValid: emailRegex.test(value),
            message: emailRegex.test(value) ? null : 'Invalid email format'
        };
    }
}

class PasswordValidation extends ValidationStrategy {
    constructor(options = {}) {
        super();
        this.minLength = options.minLength || 8;
        this.requireUppercase = options.requireUppercase !== false;
        this.requireNumbers = options.requireNumbers !== false;
        this.requireSpecialChars = options.requireSpecialChars !== false;
    }
    
    validate(value) {
        const errors = [];
        
        if (value.length < this.minLength) {
            errors.push(`Password must be at least ${this.minLength} characters`);
        }
        
        if (this.requireUppercase && !/[A-Z]/.test(value)) {
            errors.push('Password must contain uppercase letters');
        }
        
        if (this.requireNumbers && !/\d/.test(value)) {
            errors.push('Password must contain numbers');
        }
        
        if (this.requireSpecialChars && !/[!@#$%^&*]/.test(value)) {
            errors.push('Password must contain special characters');
        }
        
        return {
            isValid: errors.length === 0,
            message: errors.length > 0 ? errors.join(', ') : null
        };
    }
}

class Validator {
    constructor() {
        this.strategies = new Map();
    }
    
    addStrategy(field, strategy) {
        this.strategies.set(field, strategy);
    }
    
    validate(data) {
        const results = {};
        
        for (const [field, strategy] of this.strategies) {
            if (data.hasOwnProperty(field)) {
                results[field] = strategy.validate(data[field]);
            }
        }
        
        return results;
    }
    
    isValid(data) {
        const results = this.validate(data);
        return Object.values(results).every(result => result.isValid);
    }
}
```

## Frameworks Frontend Modernos

### React com Hooks Avançados

```jsx
import React, { useState, useEffect, useCallback, useMemo, useContext, useReducer } from 'react';

// Custom Hooks
const useLocalStorage = (key, initialValue) => {
    const [storedValue, setStoredValue] = useState(() => {
        try {
            const item = window.localStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        } catch (error) {
            console.error(`Error reading localStorage key "${key}":`, error);
            return initialValue;
        }
    });
    
    const setValue = useCallback((value) => {
        try {
            const valueToStore = value instanceof Function ? value(storedValue) : value;
            setStoredValue(valueToStore);
            window.localStorage.setItem(key, JSON.stringify(valueToStore));
        } catch (error) {
            console.error(`Error setting localStorage key "${key}":`, error);
        }
    }, [key, storedValue]);
    
    return [storedValue, setValue];
};

const useAPI = (url, options = {}) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    const fetchData = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);
            
            const response = await fetch(url, options);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            setData(result);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [url, JSON.stringify(options)]);
    
    useEffect(() => {
        fetchData();
    }, [fetchData]);
    
    return { data, loading, error, refetch: fetchData };
};

const useDebounce = (value, delay) => {
    const [debouncedValue, setDebouncedValue] = useState(value);
    
    useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);
        
        return () => {
            clearTimeout(handler);
        };
    }, [value, delay]);
    
    return debouncedValue;
};

// Context for State Management
const AppContext = React.createContext();

const appReducer = (state, action) => {
    switch (action.type) {
        case 'SET_USER':
            return { ...state, user: action.payload };
        case 'SET_THEME':
            return { ...state, theme: action.payload };
        case 'TOGGLE_SIDEBAR':
            return { ...state, sidebarOpen: !state.sidebarOpen };
        default:
            return state;
    }
};

const AppProvider = ({ children }) => {
    const [state, dispatch] = useReducer(appReducer, {
        user: null,
        theme: 'light',
        sidebarOpen: false
    });
    
    const value = useMemo(() => ({
        ...state,
        dispatch
    }), [state]);
    
    return (
        <AppContext.Provider value={value}>
            {children}
        </AppContext.Provider>
    );
};

// High-Order Component for Error Boundaries
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }
    
    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }
    
    componentDidCatch(error, errorInfo) {
        console.error('Error caught by boundary:', error, errorInfo);
        
        // Send to error reporting service
        this.props.onError?.(error, errorInfo);
    }
    
    render() {
        if (this.state.hasError) {
            return this.props.fallback || (
                <div className="error-boundary">
                    <h2>Something went wrong.</h2>
                    <details>
                        {this.state.error?.toString()}
                    </details>
                </div>
            );
        }
        
        return this.props.children;
    }
}

// React Component with Performance Optimizations
const AICodeAssistant = React.memo(({ language, context, onCodeGenerate }) => {
    const [code, setCode] = useState('');
    const [prompt, setPrompt] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);
    const debouncedPrompt = useDebounce(prompt, 500);
    
    const { data: suggestions, loading } = useAPI(
        debouncedPrompt ? `/api/suggestions?prompt=${encodeURIComponent(debouncedPrompt)}&lang=${language}` : null
    );
    
    const generateCode = useCallback(async () => {
        if (!prompt.trim()) return;
        
        setIsGenerating(true);
        
        try {
            const response = await fetch('/api/generate-code', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt,
                    language,
                    context
                })
            });
            
            const result = await response.json();
            setCode(result.code);
            onCodeGenerate?.(result.code);
        } catch (error) {
            console.error('Code generation failed:', error);
        } finally {
            setIsGenerating(false);
        }
    }, [prompt, language, context, onCodeGenerate]);
    
    return (
        <div className="ai-code-assistant">
            <div className="input-section">
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe what you want to code..."
                    rows={3}
                />
                
                <button 
                    onClick={generateCode}
                    disabled={!prompt.trim() || isGenerating}
                    className="generate-btn"
                >
                    {isGenerating ? 'Generating...' : 'Generate Code'}
                </button>
            </div>
            
            {loading && <div className="suggestions-loading">Loading suggestions...</div>}
            
            {suggestions && suggestions.length > 0 && (
                <div className="suggestions">
                    <h4>Suggestions:</h4>
                    <ul>
                        {suggestions.map((suggestion, index) => (
                            <li key={index} onClick={() => setPrompt(suggestion)}>
                                {suggestion}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
            
            {code && (
                <div className="generated-code">
                    <h4>Generated Code:</h4>
                    <pre><code>{code}</code></pre>
                </div>
            )}
        </div>
    );
});
```

## Backend com Node.js

### Express.js Avançado com TypeScript

```typescript
import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import { body, validationResult } from 'express-validator';

// Types
interface AuthenticatedRequest extends Request {
    user?: {
        id: string;
        email: string;
        role: string;
    };
}

interface APIError extends Error {
    statusCode: number;
    isOperational: boolean;
}

// Error Handling
class AppError extends Error implements APIError {
    statusCode: number;
    isOperational: boolean;
    
    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
        this.isOperational = true;
        
        Error.captureStackTrace(this, this.constructor);
    }
}

// Middleware
const errorHandler = (err: APIError, req: Request, res: Response, next: NextFunction) => {
    const { statusCode = 500, message, stack } = err;
    
    // Log error
    console.error(`[ERROR] ${message}`, {
        statusCode,
        stack: process.env.NODE_ENV === 'development' ? stack : undefined,
        url: req.url,
        method: req.method,
        ip: req.ip,
        userAgent: req.get('User-Agent')
    });
    
    // Send response
    res.status(statusCode).json({
        status: 'error',
        message: process.env.NODE_ENV === 'production' && statusCode === 500 
            ? 'Internal server error' 
            : message,
        ...(process.env.NODE_ENV === 'development' && { stack })
    });
};

const asyncHandler = (fn: Function) => {
    return (req: Request, res: Response, next: NextFunction) => {
        Promise.resolve(fn(req, res, next)).catch(next);
    };
};

const authenticate = asyncHandler(async (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
        throw new AppError('Access token required', 401);
    }
    
    try {
        // Verify JWT token (implementation depends on your auth strategy)
        const decoded = verifyToken(token);
        req.user = decoded;
        next();
    } catch (error) {
        throw new AppError('Invalid token', 401);
    }
});

const authorize = (...roles: string[]) => {
    return (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
        if (!req.user) {
            throw new AppError('Authentication required', 401);
        }
        
        if (!roles.includes(req.user.role)) {
            throw new AppError('Insufficient permissions', 403);
        }
        
        next();
    };
};

// Rate Limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // Limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP',
    standardHeaders: true,
    legacyHeaders: false,
});

const strictLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 10,
    message: 'Too many requests for this endpoint',
});

// Validation Middleware
const validateRequest = (req: Request, res: Response, next: NextFunction) => {
    const errors = validationResult(req);
    
    if (!errors.isEmpty()) {
        const errorMessages = errors.array().map(error => ({
            field: error.param,
            message: error.msg,
            value: error.value
        }));
        
        return res.status(400).json({
            status: 'error',
            message: 'Validation failed',
            errors: errorMessages
        });
    }
    
    next();
};

// Application Setup
const app = express();

// Security Middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'", "fonts.googleapis.com"],
            fontSrc: ["'self'", "fonts.gstatic.com"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "https:"],
            connectSrc: ["'self'", "wss:"]
        }
    }
}));

app.use(cors({
    origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
    credentials: true,
    optionsSuccessStatus: 200
}));

app.use(compression());
app.use(limiter);
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// API Routes
app.post('/api/auth/login', [
    body('email').isEmail().normalizeEmail(),
    body('password').isLength({ min: 6 }),
    validateRequest
], asyncHandler(async (req: Request, res: Response) => {
    const { email, password } = req.body;
    
    // Authenticate user (implementation depends on your auth strategy)
    const user = await authenticateUser(email, password);
    
    if (!user) {
        throw new AppError('Invalid credentials', 401);
    }
    
    const token = generateToken(user);
    
    res.json({
        status: 'success',
        data: {
            user: {
                id: user.id,
                email: user.email,
                role: user.role
            },
            token
        }
    });
}));

app.get('/api/users/profile', 
    authenticate,
    asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
        const user = await getUserById(req.user!.id);
        
        if (!user) {
            throw new AppError('User not found', 404);
        }
        
        res.json({
            status: 'success',
            data: { user }
        });
    })
);

app.post('/api/ai/generate-code',
    authenticate,
    strictLimiter,
    [
        body('prompt').isLength({ min: 10, max: 1000 }),
        body('language').isIn(['javascript', 'python', 'typescript', 'html', 'css']),
        validateRequest
    ],
    asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
        const { prompt, language, context } = req.body;
        
        // Call AI service to generate code
        const generatedCode = await generateCodeWithAI({
            prompt,
            language,
            context,
            userId: req.user!.id
        });
        
        res.json({
            status: 'success',
            data: {
                code: generatedCode,
                language,
                generatedAt: new Date()
            }
        });
    })
);

// Health Check
app.get('/health', (req: Request, res: Response) => {
    res.json({
        status: 'healthy',
        timestamp: new Date(),
        uptime: process.uptime(),
        version: process.env.npm_package_version
    });
});

// 404 Handler
app.all('*', (req: Request, res: Response, next: NextFunction) => {
    next(new AppError(`Route ${req.originalUrl} not found`, 404));
});

// Error Handler
app.use(errorHandler);

// Database Connection and Server Start
const startServer = async () => {
    try {
        // Connect to database
        await connectToDatabase();
        
        const PORT = process.env.PORT || 3000;
        
        app.listen(PORT, () => {
            console.log(`Server running on port ${PORT}`);
            console.log(`Environment: ${process.env.NODE_ENV}`);
        });
    } catch (error) {
        console.error('Failed to start server:', error);
        process.exit(1);
    }
};

// Graceful Shutdown
process.on('SIGTERM', async () => {
    console.log('SIGTERM received, shutting down gracefully');
    
    try {
        await disconnectFromDatabase();
        process.exit(0);
    } catch (error) {
        console.error('Error during shutdown:', error);
        process.exit(1);
    }
});

startServer();

// Utility Functions (would be in separate modules)
function verifyToken(token: string): any {
    // JWT verification implementation
    throw new Error('Not implemented');
}

function generateToken(user: any): string {
    // JWT generation implementation
    throw new Error('Not implemented');
}

async function authenticateUser(email: string, password: string): Promise<any> {
    // User authentication implementation
    throw new Error('Not implemented');
}

async function getUserById(id: string): Promise<any> {
    // Database query implementation
    throw new Error('Not implemented');
}

async function generateCodeWithAI(params: any): Promise<string> {
    // AI service integration
    throw new Error('Not implemented');
}

async function connectToDatabase(): Promise<void> {
    // Database connection implementation
    throw new Error('Not implemented');
}

async function disconnectFromDatabase(): Promise<void> {
    // Database disconnection implementation
    throw new Error('Not implemented');
}
```

## Conclusão

As linguagens web modernas formam um ecossistema complexo e em constante evolução, caracterizado por:

### **Tendências Fundamentais**:
1. **TypeScript Adoption**: Tipagem estática para JavaScript
2. **Component-Based Architecture**: React, Vue, Angular
3. **JAMstack**: JavaScript, APIs, Markup
4. **Progressive Web Apps**: Experiência nativa na web
5. **Serverless Computing**: Functions as a Service

### **Paradigmas Emergentes**:
- **Static Site Generation**: Next.js, Nuxt.js, Gatsby
- **Server-Side Rendering**: Performance e SEO otimizados
- **Micro-Frontends**: Arquitetura modular para grandes aplicações
- **Edge Computing**: Processamento próximo ao usuário

### **Ferramentas de Desenvolvimento**:
- **Build Tools**: Vite, Webpack, Parcel
- **Testing**: Jest, Cypress, Testing Library
- **DevOps**: Docker, Kubernetes, CI/CD
- **Monitoring**: Sentry, LogRocket, New Relic

### **Integração com IA**:
- **Code Generation**: GitHub Copilot, Tabnine
- **Automated Testing**: AI-powered test creation
- **Performance Optimization**: Intelligent bundling
- **User Experience**: Personalization and recommendations

O futuro das linguagens web aponta para maior automação, melhor performance e experiências de usuário mais ricas, com IA desempenhando papel central na evolução do desenvolvimento web.
