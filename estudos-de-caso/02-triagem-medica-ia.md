# 🏥 Estudo de Caso: Sistema de Triagem Médica com IA

## Visão Geral
Desenvolvimento de um sistema web para triagem inicial de pacientes em hospitais, utilizando IA para priorizar atendimentos baseado na severidade dos sintomas.

## Problema
- Filas longas em emergências
- Dificuldade para priorizar casos urgentes
- Sobrecarga da equipe médica
- Tempo de espera inadequado para casos críticos

## Solução com IA

### 1. Questionário Inteligente
- **Tecnologia**: Árvore de decisão + NLP
- **Funcionalidade**: Perguntas adaptativas baseadas nas respostas
- **Resultado**: Triagem 3x mais rápida

### 2. Análise de Sintomas
- **Algoritmo**: Classificação multi-label
- **Entrada**: Sintomas, idade, histórico médico
- **Saída**: Nível de prioridade (1-5) e especialidade recomendada

### 3. Predição de Tempo de Espera
- **Modelo**: Regressão temporal
- **Dados**: Histórico de atendimentos, sazonalidade, recursos disponíveis
- **Precisão**: 87% na predição de tempo de espera

## Implementação Técnica

### Frontend (Tablet para Pacientes)
```html
<!-- Interface simplificada para triagem -->
<div class="triagem-container">
  <h2>Sistema de Triagem Inteligente</h2>
  <div id="pergunta-atual"></div>
  <div id="opcoes-resposta"></div>
  <button onclick="proximaPergunta()">Continuar</button>
</div>
```

### Algoritmo de Triagem
```javascript
function calcularPrioridade(sintomas, idade, historico) {
  let score = 0;
  
  // Sintomas críticos
  if (sintomas.includes('dor_peito_intensa')) score += 10;
  if (sintomas.includes('dificuldade_respirar')) score += 8;
  if (sintomas.includes('perda_consciencia')) score += 10;
  
  // Fatores de risco por idade
  if (idade > 65) score += 2;
  if (idade < 2) score += 3;
  
  // Histórico médico
  if (historico.includes('diabetes')) score += 1;
  if (historico.includes('hipertensao')) score += 1;
  
  return classificarPrioridade(score);
}

function classificarPrioridade(score) {
  if (score >= 10) return { nivel: 1, cor: 'vermelho', tempo: '0 min' };
  if (score >= 7) return { nivel: 2, cor: 'laranja', tempo: '10 min' };
  if (score >= 4) return { nivel: 3, cor: 'amarelo', tempo: '30 min' };
  if (score >= 2) return { nivel: 4, cor: 'verde', tempo: '60 min' };
  return { nivel: 5, cor: 'azul', tempo: '120 min' };
}
```

### Backend (Python + Flask)
```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar modelo treinado
modelo_triagem = joblib.load('modelo_triagem.pkl')

@app.route('/triagem', methods=['POST'])
def realizar_triagem():
    dados = request.json
    
    # Preparar dados para o modelo
    features = [
        dados['idade'],
        dados['temperatura'],
        len(dados['sintomas']),
        dados['dor_escala'],
        dados['pressao_sistolica']
    ]
    
    # Predição
    prioridade = modelo_triagem.predict([features])[0]
    probabilidade = modelo_triagem.predict_proba([features])[0].max()
    
    return jsonify({
        'prioridade': int(prioridade),
        'confianca': float(probabilidade),
        'tempo_estimado': calcular_tempo_espera(prioridade),
        'especialidade': sugerir_especialidade(dados['sintomas'])
    })
```

## Resultados Obtidos

### Métricas de Performance
- **Redução do tempo de triagem**: 65%
- **Precisão na classificação**: 91%
- **Redução de casos mal priorizados**: 78%
- **Satisfação dos pacientes**: +42%

### Impacto Operacional
- **Tempo médio de espera crítica**: 15min → 3min
- **Eficiência da equipe**: +35%
- **Redução de reclamações**: 60%
- **Melhor distribuição de recursos**: Otimização de 23%

## Considerações Éticas e Legais

### Responsabilidades
- ✅ IA como **apoio** à decisão médica, não substituto
- ✅ Revisão obrigatória por profissional qualificado
- ✅ Transparência sobre uso de IA para pacientes
- ✅ Conformidade com LGPD e normas médicas

### Limitações
- Sistema não substitui avaliação médica presencial
- Casos raros podem não ser bem classificados
- Necessita atualização constante com novos dados
- Viés pode ser introduzido pelos dados de treinamento

## Expansões Futuras
1. **Integração com IoT**: Sensores para sinais vitais automáticos
2. **IA conversacional**: Chatbot para coleta de sintomas
3. **Visão computacional**: Análise de imagens médicas básicas
4. **Medicina preditiva**: Prever deterioração de pacientes

## Tecnologias Utilizadas
- **Frontend**: React + TypeScript
- **Backend**: Python + Flask + scikit-learn
- **Banco**: PostgreSQL
- **Deploy**: Docker + AWS
- **Monitoramento**: Prometheus + Grafana

Veja implementação detalhada na pasta `exemplos/`
