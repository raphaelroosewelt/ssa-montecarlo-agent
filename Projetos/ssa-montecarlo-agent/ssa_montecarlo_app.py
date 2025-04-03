import os
import json
import numpy as np
import yfinance as yf
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
import matplotlib.pyplot as plt
import io

load_dotenv()
client = OpenAI()

def extract_parameters_with_llm(user_question, model="gpt-3.5-turbo"):
    prompt = f"""
Você é um extrator de parâmetros para precificação de opções via simulação de Monte Carlo.

Sua tarefa é analisar a pergunta de um usuário e retornar um JSON com os seguintes campos:
- "symbol": (string) o código do ativo, como PETR4 ou VALE3
- "strike": (float) o valor que o usuário quer saber se será atingido
- "date": (string, formato YYYY-MM-DD) a data até a qual o valor pode ser atingido

Regras:
- Se faltar qualquer um desses 3 parâmetros, responda com: "missing"
- Interprete expressões como "mês que vem", "ano que vem", "fim do ano" e transforme em uma data real
- Não invente dados
- Use ponto como separador decimal
- Retorne apenas o JSON

Exemplo:
Pergunta: "Qual a chance do ITUB4 bater 32 até 20 de agosto de 2025?"
Resposta:
{{
  "symbol": "ITUB4",
  "strike": 32,
  "date": "2025-08-20"
}}

Pergunta: "{user_question}"
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    if content.lower() == "missing":
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None

def adjust_symbol(symbol):
    return symbol + ".SA" if "." not in symbol else symbol

def get_current_price(symbol):
    symbol = adjust_symbol(symbol)
    data = yf.Ticker(symbol).history(period="1d")
    if data.empty:
        raise ValueError("Não foi possível obter o preço atual do ativo.")
    return data['Close'].iloc[-1]

def get_annualized_volatility(symbol):
    symbol = adjust_symbol(symbol)
    data = yf.Ticker(symbol).history(period="2y")
    if data.empty:
        raise ValueError("Não foi possível obter os dados históricos do ativo.")
    data['Return'] = data['Close'].pct_change()
    return data['Return'].std() * np.sqrt(252)

def calculate_probability(symbol, strike, target_date, simulations=1_000_000, r=0.1425):
    S0 = get_current_price(symbol)
    sigma = get_annualized_volatility(symbol)
    delta_days = (target_date - datetime.today()).days
    if delta_days < 1:
        raise ValueError("A data alvo deve estar no futuro.")
    T = delta_days / 252
    np.random.seed(42)
    Z = np.random.standard_normal(simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    p_call = np.mean(ST >= strike) * 100
    p_put = 100 - p_call
    return S0, sigma, T, delta_days, p_call, p_put, ST

def plot_distribution(ST, strike, S0):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ST, bins=100, color="#4f81bd", alpha=0.7, edgecolor="black")
    ax.axvline(strike, color="red", linestyle="--", label=f"Strike ({strike})")
    ax.axvline(S0, color="green", linestyle=":", label=f"Atual ({S0:.2f})")
    ax.axvline(np.mean(ST), color="orange", linestyle="-.", label=f"Média Simulada ({np.mean(ST):.2f})")
    ax.set_title("Distribuição de Preços Simulados")
    ax.set_xlabel("Preço Final do Ativo")
    ax.set_ylabel("Frequência")
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def run_agent_intelligent(query):
    params = extract_parameters_with_llm(query)
    if not params:
        return "❌ Não foi possível entender sua pergunta.", None
    try:
        symbol = params["symbol"]
        strike = float(params["strike"])
        date = datetime.strptime(params["date"], "%Y-%m-%d")
        S0, sigma, T, delta_days, p_call, p_put, ST = calculate_probability(symbol, strike, date)

        resumo = f"""
📈 Ativo: {symbol}
🎯 Preço alvo: {strike}
🗓️ Data alvo: {date.strftime('%d/%m/%Y')}
💰 Preço atual: {S0:.2f}
📊 Volatilidade anualizada: {sigma:.2%}
⏳ Tempo até o vencimento: {delta_days} dias corridos ({T:.2f} anos úteis)
📈 Probabilidade de atingir ou ultrapassar (CALL): {p_call:.2f}%
📉 Probabilidade de não atingir (PUT): {p_put:.2f}%

🧾 Interpretação do gráfico:
- A linha vermelha representa o strike ({strike}) que o ativo deve atingir.
- A linha verde pontilhada mostra o preço atual do ativo.
- A linha laranja tracejada representa o preço médio estimado no vencimento com base nas simulações.
"""
        plot = plot_distribution(ST, strike, S0)
        return resumo, plot
    except Exception as e:
        return f"❌ Erro: {e}", None

if __name__ == "__main__":
    gr.Interface(
        fn=run_agent_intelligent,
        inputs=gr.Textbox(lines=2, placeholder="Ex: Qual a chance da PETR4 bater 45 até o mês que vem?"),
        outputs=["text", "image"],
        title="Agente - Calcula a probabilidade de um ativo atingir o strike usando Monte Carlo",
        description="Digite uma pergunta sobre um ativo, valor e data alvo. O gráfico mostra a distribuição dos preços simulados, com destaque para o strike, preço atual e média prevista."
    ).launch()
