"""
PLATAFORMA COMPLETA B3 - Dashboard Avançado de Ações Brasileiras
Versão Ultra Completa com todas as funcionalidades profissionais
"""

import logging
import math
import warnings
from datetime import datetime, time, timedelta
from io import StringIO
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Logger global
logger = logging.getLogger("b3_dashboard")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Configuração da página
st.set_page_config(
    page_title="B3 Dashboard Pro - Ações Brasileiras",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈",
)

# CSS customizado
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .alert-high { background-color: #d4edda; border-color: #c3e6cb; }
    .alert-low { background-color: #f8d7da; border-color: #f5c6cb; }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #1f77b4;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Listas de ativos
BRAZILIAN_TICKERS = [
    "VALE3",
    "PETR4",
    "PETR3",
    "ITUB4",
    "ITUB3",
    "ABEV3",
    "B3SA3",
    "WEGE3",
    "BBAS3",
    "BBDC4",
    "BBDC3",
    "BBSE3",
    "SUZB3",
    "GGBR4",
    "JBSS3",
    "COGN3",
    "PRIO3",
    "EMBR3",
    "VIVT3",
    "ELET3",
    "ELET6",
    "LREN3",
    "RENT3",
    "MGLU3",
    "BRFS3",
    "CSNA3",
    "CPLE6",
    "BRKM5",
    "EQTL3",
    "TAEE11",
    "VAMO3",
    "NTCO3",
    "RADL3",
    "HAPV3",
    "BRML3",
    "YDUQ3",
    "SANB11",
    "CRFB3",
    "TOTS3",
    "MRVE3",
    "SOMA3",
    "CMIG4",
    "UGPA3",
    "RAIL3",
    "TIMS3",
    "BRAP4",
    "MRFG3",
    "ENEV3",
    "ALPA4",
    "AZUL4",
    "ARZZ3",
    "STBP11",
    "CVCB3",
    "GOAU4",
    "BEEF3",
    "EZTC3",
    "FLRY3",
    "CSAN3",
    "CIEL3",
    "CCRO3",
    "CYRE3",
    "DXCO3",
    "ENGI11",
    "GOLL4",
    "HYPE3",
    "IRBR3",
    "KLBN11",
    "LAME4",
    "MULT3",
    "PCAR3",
    "QUAL3",
    "RDOR3",
    "SBSP3",
    "SLCE3",
    "TRPL4",
    "USIM5",
    "VBBR3",
    "LWSA3",
    "ENBR3",
    "ITSA4",
    "BPAC11",
    "BRSR6",
    "PINE4",
    "BMFB3",
    "BIDI4",
    "EGIE3",
    "LIGT3",
    "NEOE3",
    "OIBR3",
    "OIBR4",
    "RNEW11",
    "AMER3",
    "GUAR3",
    "SMFT3",
    "JHSF3",
    "HBOR3",
    "PDGR3",
    "EVEN3",
    "PNVL3",
    "DASA3",
    "ONCO3",
    "APER3",
    "SEER3",
    "SMTO3",
    "SLC83",
    "PETZ3",
    "MELI34",
]

REAL_ESTATE_FUNDS = [
    "HGLG11",
    "XPML11",
    "VISC11",
    "KNRI11",
    "HGCR11",
    "BTLG11",
    "VGIR11",
    "BCFF11",
    "VILG11",
    "MXRF11",
    "IRDM11",
    "KNCR11",
    "XPIN11",
    "RBRR11",
]

BRAZILIAN_TICKERS = list(dict.fromkeys(ticker.strip().upper() for ticker in BRAZILIAN_TICKERS))
REAL_ESTATE_FUNDS = list(dict.fromkeys(fii.strip().upper() for fii in REAL_ESTATE_FUNDS))


@st.cache_data(ttl=60)
def get_market_overview() -> Dict[str, Dict[str, float]]:
    """Obtém visão geral de índices principais."""
    indices = {"IBOVESPA": "^BVSP", "IFIX": "^IFIX", "Dólar": "USDBRL=X"}
    overview: Dict[str, Dict[str, float]] = {}
    for label, symbol in indices.items():
        try:
            hist = yf.Ticker(symbol).history(period="5d")
            if hist.empty:
                overview[label] = None
                continue
            last_close = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_close
            delta_pct = ((last_close / prev_close) - 1) * 100 if prev_close else 0.0
            overview[label] = {"ultimo": last_close, "variacao_pct": delta_pct}
        except Exception as exc:
            logger.warning("Falha ao obter dados de %s (%s): %s", label, symbol, exc)
            overview[label] = None
    return overview


def is_market_open() -> Tuple[bool, str]:
    """Verifica se a B3 está aberta."""
    try:
        brt = pytz.timezone("America/Sao_Paulo")
        now_brt = datetime.now(brt)

        if now_brt.weekday() >= 5:
            return False, "Mercado fechado (fim de semana)"

        market_open = time(10, 0)
        market_close = time(17, 30)
        current_time = now_brt.time()

        if market_open <= current_time <= market_close:
            return True, f"Mercado aberto • {now_brt.strftime('%H:%M')} BRT"
        if current_time < market_open:
            return False, f"Pré-abertura • Retoma às {market_open.strftime('%H:%M')} BRT"
        return False, "Mercado fechado • Após o horário regular"
    except Exception as exc:
        logger.warning("Não foi possível verificar status do mercado: %s", exc)
        return False, "Status do mercado indisponível"


@st.cache_data(ttl=30, show_spinner=False)
def fetch_enhanced_stock_data(tickers: List[str], include_fiis: bool = False) -> pd.DataFrame:
    """Busca dados completos das ações com análise técnica."""
    base_tickers = [ticker.strip().upper() for ticker in tickers if ticker]
    all_tickers = base_tickers + (REAL_ESTATE_FUNDS if include_fiis else [])
    all_tickers = list(dict.fromkeys(all_tickers))

    if not all_tickers:
        return pd.DataFrame()

    mapped_symbols = {ticker: ticker if "." in ticker else f"{ticker}.SA" for ticker in all_tickers}

    try:
        hist = yf.download(
            list(mapped_symbols.values()),
            period="1y",
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:
        logger.exception("Erro ao baixar dados históricos: %s", exc)
        return pd.DataFrame()

    if hist.empty:
        return pd.DataFrame()

    if not isinstance(hist.columns, pd.MultiIndex):
        first_symbol = next(iter(mapped_symbols.values()))
        hist = pd.concat({first_symbol: hist}, axis=1)

    tickers_obj = yf.Tickers(" ".join(mapped_symbols.values()))
    data_rows = []

    for original, symbol in mapped_symbols.items():
        try:
            ticker_hist = hist.xs(symbol, axis=1, level=0).dropna()
        except KeyError:
            logger.warning("Histórico não encontrado para %s", original)
            continue

        if ticker_hist.empty:
            logger.warning("Histórico vazio para %s", original)
            continue

        close_series = ticker_hist["Close"]
        last_close = float(close_series.iloc[-1])
        prev_close = float(close_series.iloc[-2]) if len(close_series) > 1 else last_close
        price_change_pct = ((last_close / prev_close) - 1) * 100 if prev_close else 0.0

        daily_returns = close_series.pct_change().dropna()
        if not daily_returns.empty:
            recent_window = daily_returns.tail(21)
            volatility_pct = float(recent_window.std() * math.sqrt(252) * 100)
        else:
            volatility_pct = 0.0

        window_52w = close_series.tail(252)
        high_52w = float(window_52w.max()) if not window_52w.empty else last_close
        low_52w = float(window_52w.min()) if not window_52w.empty else last_close

        rsi_value = float(calculate_rsi(close_series, window=14))

        ticker_obj = tickers_obj.tickers.get(symbol)
        short_name = original
        sector = "N/A"
        if ticker_obj:
            for attr in ("fast_info", "info"):
                try:
                    candidate = getattr(ticker_obj, attr)
                except Exception:
                    continue
                if not candidate:
                    continue

                short_name_options = []
                sector_options = []

                if isinstance(candidate, dict):
                    short_name_options.extend(
                        [
                            candidate.get("shortName"),
                            candidate.get("longName"),
                        ]
                    )
                    sector_options.extend(
                        [
                            candidate.get("sector"),
                            candidate.get("industry"),
                        ]
                    )
                else:
                    short_name_options.extend(
                        [
                            getattr(candidate, "shortName", None),
                            getattr(candidate, "longName", None),
                        ]
                    )
                    sector_options.extend(
                        [
                            getattr(candidate, "sector", None),
                            getattr(candidate, "industry", None),
                        ]
                    )
                    if hasattr(candidate, "get"):
                        short_name_options.extend(
                            [candidate.get("shortName"), candidate.get("longName")]
                        )
                        sector_options.extend(
                            [candidate.get("sector"), candidate.get("industry")]
                        )

                short_name_candidate = next((value for value in short_name_options if value), None)
                sector_candidate = next((value for value in sector_options if value), None)

                if short_name_candidate:
                    short_name = short_name_candidate
                if sector_candidate:
                    sector = sector_candidate
                if short_name_candidate or sector_candidate:
                    break

        data_rows.append(
            {
                "Ticker": original,
                "Nome": short_name,
                "Preço Atual (R$)": last_close,
                "Variação (%)": price_change_pct,
                "Variação Numérica": price_change_pct,
                "Volume": int(ticker_hist["Volume"].iloc[-1]) if "Volume" in ticker_hist else 0,
                "RSI": rsi_value,
                "Volatilidade (%)": volatility_pct,
                "Volatilidade Numérica": volatility_pct,
                "Setor": sector or "N/A",
                "Tipo": "FII" if original in REAL_ESTATE_FUNDS else "Ação",
                "52w High": high_52w,
                "52w Low": low_52w,
            }
        )

    df = pd.DataFrame(data_rows)
    if df.empty:
        return df

    return df.sort_values("Ticker").reset_index(drop=True)


def calculate_rsi(prices: pd.Series, window: int = 14, return_series: bool = False):
    """Calcula o RSI (Relative Strength Index)."""
    try:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean().replace(0, pd.NA)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(method="bfill").fillna(method="ffill")
        if return_series:
            return rsi
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    except Exception as exc:
        logger.warning("Falha ao calcular RSI: %s", exc)
        if return_series:
            return pd.Series(dtype=float)
        return 50.0


def create_advanced_chart(ticker: str, period: str = "1d", chart_type: str = "candlestick"):
    """Cria gráfico técnico avançado."""
    try:
        full_ticker = f"{ticker}.SA" if "." not in ticker else ticker
        stock = yf.Ticker(full_ticker)

        if period == "1d":
            hist = stock.history(period="1d", interval="5m")
        elif period == "1w":
            hist = stock.history(period="7d", interval="30m")
        elif period == "1m":
            hist = stock.history(period="1mo", interval="1d")
        else:
            hist = stock.history(period=period, interval="1d")

        if hist.empty:
            return None

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f"{ticker} - Preço", "Volume", "RSI"),
            row_width=[0.2, 0.15, 0.65],
        )

        if chart_type == "candlestick" and len(hist) > 1:
            fig.add_trace(
                go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name="Preço",
                    increasing_line_color="#2ECC71",
                    decreasing_line_color="#E74C3C",
                ),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist["Close"],
                    mode="lines",
                    name="Preço",
                    line=dict(color="#1f77b4", width=2),
                ),
                row=1,
                col=1,
            )

        if len(hist) >= 20:
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist["Close"].rolling(window=20).mean(),
                    mode="lines",
                    name="MM20",
                    line=dict(color="#FF9800", width=1.4),
                ),
                row=1,
                col=1,
            )

        if len(hist) >= 50:
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist["Close"].rolling(window=50).mean(),
                    mode="lines",
                    name="MM50",
                    line=dict(color="#9C27B0", width=1.4),
                ),
                row=1,
                col=1,
            )

        colors = [
            "#E74C3C" if hist["Close"].iloc[i] < hist["Open"].iloc[i] else "#2ECC71"
            for i in range(len(hist))
        ]
        fig.add_trace(
            go.Bar(
                x=hist.index,
                y=hist["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

        if len(hist) >= 14:
            rsi_series = calculate_rsi(hist["Close"], window=14, return_series=True)
            if not rsi_series.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rsi_series.index,
                        y=rsi_series,
                        mode="lines",
                        name="RSI (14)",
                        line=dict(color="#34495E", width=1.5),
                    ),
                    row=3,
                    col=1,
                )

        fig.update_layout(
            title=f"Análise Técnica - {ticker}",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            hovermode="x unified",
        )

        fig.update_yaxes(title_text="Preço (R$)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        fig.update_xaxes(title_text="Data", row=3, col=1)

        return fig

    except Exception as exc:
        logger.exception("Erro ao montar gráfico avançado de %s: %s", ticker, exc)
        return None


def create_comparison_chart(tickers_list: List[str], period: str = "1m"):
    """Cria gráfico de comparação entre múltiplas ações."""
    try:
        fig = go.Figure()

        for ticker in tickers_list:
            full_ticker = f"{ticker}.SA" if "." not in ticker else ticker
            hist = yf.Ticker(full_ticker).history(period=period, interval="1d")
            if hist.empty:
                logger.warning("Sem dados para comparação de %s", ticker)
                continue

            normalized = (hist["Close"] / hist["Close"].iloc[0]) * 100
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=normalized,
                    mode="lines",
                    name=ticker,
                )
            )

        if not fig.data:
            return None

        fig.update_layout(
            title="Comparação de Performance (Base 100)",
            xaxis_title="Data",
            yaxis_title="Performance Normalizada",
            height=500,
            hovermode="x unified",
        )

        return fig

    except Exception as exc:
        logger.exception("Erro ao montar gráfico de comparação: %s", exc)
        return None


def calculate_returns(
    ticker: str,
    investment_amount: float,
    start_date: datetime,
    end_date: datetime,
):
    """Calcula rentabilidade de investimento."""
    try:
        if not all([ticker, investment_amount, start_date, end_date]):
            return None

        full_ticker = f"{ticker}.SA" if "." not in ticker else ticker
        stock = yf.Ticker(full_ticker)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty or len(hist) < 2:
            return None

        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])

        if start_price == 0:
            return None

        shares = investment_amount / start_price
        final_value = shares * end_price
        total_return = final_value - investment_amount
        return_pct = (total_return / investment_amount) * 100 if investment_amount else 0.0

        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        if days <= 0:
            annualized_return = 0.0
        else:
            annualized_return = ((final_value / investment_amount) ** (365 / days)) - 1

        return {
            "investment": investment_amount,
            "final_value": final_value,
            "total_return": total_return,
            "return_pct": return_pct,
            "annualized_return": annualized_return * 100,
            "shares": shares,
            "start_price": start_price,
            "end_price": end_price,
        }
    except Exception as exc:
        logger.exception("Erro ao calcular rentabilidade de %s: %s", ticker, exc)
        return None


def render_main_dashboard(include_fiis: bool = False):
    st.subheader("📊 Visão Geral do Mercado")
    default_selection = ["VALE3", "PETR4", "ITUB4", "B3SA3", "WEGE3"]
    selection = st.multiselect(
        "Selecione os ativos",
        options=BRAZILIAN_TICKERS + (REAL_ESTATE_FUNDS if include_fiis else []),
        default=default_selection,
    )

    if not selection:
        st.info("Selecione pelo menos um ativo para visualizar os dados.")
        return

    data = fetch_enhanced_stock_data(selection, include_fiis=include_fiis)

    if data.empty:
        st.warning("Nenhum dado disponível no momento.")
        return

    sectores = sorted(data["Setor"].dropna().replace({"": "N/A"}).unique())
    sector_filter = st.selectbox("Filtrar por setor", options=["Todos"] + sectores)
    if sector_filter != "Todos":
        data = data[data["Setor"].fillna("N/A") == sector_filter]

    st.dataframe(
        data[
            [
                "Ticker",
                "Nome",
                "Preço Atual (R$)",
                "Variação (%)",
                "Volume",
                "RSI",
                "Volatilidade (%)",
                "52w High",
                "52w Low",
                "Setor",
                "Tipo",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Preço Atual (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
            "Variação (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Volatilidade (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "52w High": st.column_config.NumberColumn(format="R$ %.2f"),
            "52w Low": st.column_config.NumberColumn(format="R$ %.2f"),
        },
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Maior Alta",
        data.loc[data["Variação Numérica"].idxmax(), "Ticker"],
        f"{data['Variação Numérica'].max():+.2f}%",
    )
    col2.metric(
        "Maior Baixa",
        data.loc[data["Variação Numérica"].idxmin(), "Ticker"],
        f"{data['Variação Numérica'].min():+.2f}%",
    )
    col3.metric(
        "Mais Volátil (21d)",
        data.loc[data["Volatilidade Numérica"].idxmax(), "Ticker"],
        f"{data['Volatilidade Numérica'].max():.2f}%",
    )


def render_individual_analysis():
    st.subheader("🔍 Análise Individual")
    ticker = st.selectbox("Selecione o ativo", options=sorted(BRAZILIAN_TICKERS))
    period = st.selectbox(
        "Período",
        options=["1d", "1w", "1m", "3mo", "6mo", "1y", "2y", "5y"],
        index=2,
    )
    chart_type = st.selectbox("Tipo de gráfico", options=["candlestick", "line"], index=0)

    fig = create_advanced_chart(ticker, period=period, chart_type=chart_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Não foi possível carregar o gráfico deste ativo.")


def render_comparison_analysis():
    st.subheader("⚖️ Comparação entre Ativos")
    tickers_list = st.multiselect(
        "Selecione até 5 ativos",
        options=sorted(BRAZILIAN_TICKERS),
        default=["VALE3", "PETR4", "ITUB4"],
        max_selections=5,
    )
    period = st.selectbox("Período", options=["1m", "3mo", "6mo", "1y"], index=0)

    if not tickers_list:
        st.info("Selecione pelo menos dois ativos.")
        return

    fig = create_comparison_chart(tickers_list, period=period)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Sem dados suficientes para montar o gráfico.")


def render_calculator():
    st.subheader("💰 Calculadora de Rentabilidade")

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.selectbox("Ativo", options=sorted(BRAZILIAN_TICKERS))
        investment_amount = st.number_input("Valor investido (R$)", min_value=0.0, value=1000.0, step=100.0)

    with col2:
        start_date = st.date_input("Data inicial", value=datetime.now() - timedelta(days=180))
        end_date = st.date_input("Data final", value=datetime.now())

    if st.button("Calcular"):
        result = calculate_returns(ticker, investment_amount, start_date, end_date)
        if not result:
            st.warning("Não foi possível calcular. Verifique as datas e o ativo selecionado.")
            return

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Valor Final", f"R$ {result['final_value']:.2f}", f"{result['return_pct']:.2f}%")
        col_b.metric("Retorno Total", f"R$ {result['total_return']:.2f}")
        col_c.metric("Retorno Anualizado", f"{result['annualized_return']:.2f}%")

        st.caption(
            f"Preço inicial: R$ {result['start_price']:.2f} • Preço final: R$ {result['end_price']:.2f} "
            f"• Quantidade aproximada de ações: {result['shares']:.2f}"
        )


def render_rankings(include_fiis: bool = False):
    st.subheader("🏆 Rankings de Performance")

    top_universe = BRAZILIAN_TICKERS + (REAL_ESTATE_FUNDS if include_fiis else [])
    data = fetch_enhanced_stock_data(top_universe, include_fiis=include_fiis)
    if data.empty:
        st.warning("Não foi possível gerar rankings no momento.")
        return

    col1, col2, col3 = st.columns(3)
    col1.write("### Maiores Altas (24h)")
    col1.dataframe(
        data.nlargest(5, "Variação Numérica")[["Ticker", "Preço Atual (R$)", "Variação (%)"]],
        hide_index=True,
    )

    col2.write("### Maiores Baixas (24h)")
    col2.dataframe(
        data.nsmallest(5, "Variação Numérica")[["Ticker", "Preço Atual (R$)", "Variação (%)"]],
        hide_index=True,
    )

    col3.write("### Mais Voláteis (21d)")
    col3.dataframe(
        data.nlargest(5, "Volatilidade Numérica")[["Ticker", "Preço Atual (R$)", "Volatilidade (%)"]],
        hide_index=True,
    )


def render_alerts():
    st.subheader("🔔 Alertas de Preço (mock)")

    ticker = st.selectbox("Ativo", options=sorted(BRAZILIAN_TICKERS))
    target_price = st.number_input("Preço alvo (R$)", min_value=0.0, value=50.0, step=1.0)
    direction = st.radio("Condição", ["Acima", "Abaixo"], horizontal=True)
    email = st.text_input("Email para notificação")

    if st.button("Criar alerta"):
        st.success(
            f"Alerta criado: {ticker} {'acima' if direction == 'Acima' else 'abaixo'} de R$ {target_price:.2f}. "
            "Integração de envio ainda não implementada."
        )


def render_reports(include_fiis: bool = False):
    st.subheader("📄 Relatórios e Exportações")

    selection = st.multiselect(
        "Selecione os ativos para exportar",
        options=BRAZILIAN_TICKERS + (REAL_ESTATE_FUNDS if include_fiis else []),
        default=["VALE3", "PETR4", "ITUB4"],
    )

    if not selection:
        st.info("Escolha ao menos um ativo para exportação.")
        return

    data = fetch_enhanced_stock_data(selection, include_fiis=include_fiis)
    if data.empty:
        st.warning("Não há dados para exportar.")
        return

    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    st.download_button(
        "Baixar CSV",
        data=csv_buffer.getvalue(),
        file_name="b3_dashboard_export.csv",
        mime="text/csv",
    )

    st.write("Prévia dos dados exportados:")
    st.dataframe(data, hide_index=True, use_container_width=True)


def main():
    st.markdown('<div class="main-header">B3 Dashboard Pro</div>', unsafe_allow_html=True)

    market_open, market_message = is_market_open()
    if market_open:
        st.success(market_message)
    else:
        st.info(market_message)

    st.sidebar.markdown("## ⚙️ Configurações")
    include_fiis = st.sidebar.checkbox("Incluir FIIs", value=False)
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

    if auto_refresh:
        now = datetime.utcnow()
        last_refresh = st.session_state.get("_last_refresh_ts")
        if last_refresh is None:
            st.session_state["_last_refresh_ts"] = now
        elif (now - last_refresh).total_seconds() >= 30:
            st.session_state["_last_refresh_ts"] = now
            st.experimental_rerun()
    else:
        st.session_state.pop("_last_refresh_ts", None)

    st.sidebar.markdown("### 🌍 Mercado Geral")
    market_data = get_market_overview()
    for name, data in (market_data or {}).items():
        if not data:
            st.sidebar.caption(f"{name}: dados indisponíveis")
            continue
        st.sidebar.metric(name, f"{data['ultimo']:.2f}", f"{data['variacao_pct']:+.2f}%")

    st.sidebar.markdown("### 🧭 Navegação")
    page = st.sidebar.radio(
        "Escolha a seção",
        options=[
            "Dashboard",
            "Análise Individual",
            "Comparação",
            "Calculadora",
            "Rankings",
            "Alertas",
            "Relatórios",
        ],
    )

    if page == "Dashboard":
        render_main_dashboard(include_fiis=include_fiis)
    elif page == "Análise Individual":
        render_individual_analysis()
    elif page == "Comparação":
        render_comparison_analysis()
    elif page == "Calculadora":
        render_calculator()
    elif page == "Rankings":
        render_rankings(include_fiis=include_fiis)
    elif page == "Alertas":
        render_alerts()
    elif page == "Relatórios":
        render_reports(include_fiis=include_fiis)


if __name__ == "__main__":
    main()
