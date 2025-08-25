# Passo a Passo: Chatbot Web Completo

## 1. Estrutura HTML
```html
<div id="chat">
  <div id="mensagens"></div>
  <input type="text" id="entrada" placeholder="Digite sua mensagem...">
  <button onclick="enviarMensagem()">Enviar</button>
</div>
```

## 2. CSS Básico
```html
<style>
#chat { width: 300px; margin: 0 auto; background: #fff; padding: 10px; border-radius: 8px; box-shadow: 0 0 10px #ccc; }
#mensagens { height: 200px; overflow-y: auto; border: 1px solid #eee; margin-bottom: 10px; padding: 5px; }
</style>
```

## 3. JavaScript para lógica do chatbot
```html
<script>
function enviarMensagem() {
  var entrada = document.getElementById('entrada').value;
  if (!entrada) return;
  adicionarMensagem('Você: ' + entrada);
  var resposta = gerarResposta(entrada);
  adicionarMensagem('Bot: ' + resposta);
  document.getElementById('entrada').value = '';
}
function adicionarMensagem(msg) {
  var mensagens = document.getElementById('mensagens');
  mensagens.innerHTML += '<div>' + msg + '</div>';
  mensagens.scrollTop = mensagens.scrollHeight;
}
function gerarResposta(texto) {
  texto = texto.toLowerCase();
  if (texto.includes('oi')) return 'Olá! Como posso ajudar?';
  if (texto.includes('preço')) return 'Consulte nosso site para preços atualizados.';
  if (texto.includes('tchau')) return 'Até logo!';
  return 'Desculpe, não entendi.';
}
</script>
```

## 4. Teste e melhorias
- Adicione mais respostas.
- Melhore o layout.
- Integre com backend se desejar respostas mais inteligentes.
