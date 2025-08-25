# Passo a Passo: Usando a API do ChatGPT (OpenAI) em uma Página Web

## 1. Crie uma conta em https://platform.openai.com/
## 2. Gere uma chave de API (API Key)
## 3. No seu código JavaScript, use `fetch` para enviar perguntas e receber respostas:

```js
fetch('https://api.openai.com/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer SUA_API_KEY'
  },
  body: JSON.stringify({
    model: 'gpt-3.5-turbo',
    messages: [{role: 'user', content: 'Olá, IA!'}]
  })
})
.then(res => res.json())
.then(data => console.log(data.choices[0].message.content));
```

## 4. Substitua `SUA_API_KEY` pela sua chave real.
## 5. Veja o exemplo completo em `exemplos/exemplo-openai-chatgpt-js.html`.
