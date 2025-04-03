
# 🧠 SSA Monte Carlo Agent

Este projeto é um agente inteligente (SSA - Small Specific Agent) que calcula a **probabilidade de um ativo atingir um preço-alvo até uma data futura**, usando simulação de Monte Carlo com dados reais do mercado financeiro.

---

## 🚀 O que o agente faz

- 🔍 Extrai parâmetros da sua pergunta com LLM (OpenAI)
- 📈 Coleta dados reais via Yahoo Finance (`yfinance`)
- 🎲 Realiza 1.000.000 de simulações para projetar preços futuros
- 📊 Exibe a distribuição dos preços simulados (com gráfico interativo)
- 💬 Explica as chances de atingir ou não o preço-alvo

---

## 🧪 Exemplos de perguntas

```txt
Qual a chance da PETR4 bater 45 até o mês que vem?
Qual a chance de VALE3 atingir 82 até 15/09/2025?
```

---

## 🛠️ Como rodar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/ssa-montecarlo-agent.git
cd ssa-montecarlo-agent
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# ou
source venv/bin/activate  # Linux/macOS
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure sua chave da OpenAI

Crie um arquivo `.env`:

```env
OPENAI_API_KEY=sk-sua-chave-aqui
```

### 5. Rode o agente

```bash
python ssa_montecarlo_app.py
```

---

## 🧠 Tecnologias utilizadas

- `openai` (chat-based LLM)
- `yfinance` (preço histórico)
- `numpy` e `matplotlib` (simulações)
- `gradio` (interface web)
- `python-dotenv` (variáveis de ambiente)

---

## 📈 Exemplo de saída

- Probabilidade de **Call** (atingir ou ultrapassar o alvo)
- Probabilidade de **Put** (não atingir o alvo)
- Gráfico com:
  - Linha do preço atual
  - Linha do strike
  - Média dos preços simulados

---

## 📌 Roadmap (ideias futuras)

- [ ] Exportar gráfico para PNG/PDF
- [ ] Adicionar suporte a múltiplos ativos
- [ ] Históricos e logs de simulações
- [ ] Deploy online via HuggingFace ou Render

---

Feito com 💻 e ☕ por [Seu Nome]
