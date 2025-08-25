# Passo a Passo: Criando um Agente de IA Simples (Chatbot)

## 1. Usando JavaScript para um chatbot básico
```html
<input type="text" id="entrada" placeholder="Digite algo...">
<button onclick="responder()">Enviar</button>
<p id="resposta"></p>
<script>
function responder() {
    var entrada = document.getElementById('entrada').value.toLowerCase();
    var resposta = 'Não entendi.';
    if (entrada.includes('oi')) resposta = 'Olá! Como posso ajudar?';
    if (entrada.includes('tchau')) resposta = 'Até logo!';
    document.getElementById('resposta').innerText = resposta;
}
</script>
```

## 2. Explicação
Esse é um exemplo simples de agente baseado em regras, útil para entender o conceito de agentes reativos.
