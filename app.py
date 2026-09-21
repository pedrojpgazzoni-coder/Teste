import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Dashboard de Vendas & Machine Learning", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Cabecalho_da_Nota (5).xlsx")
    except Exception as e:
        st.error(f"Erro ao carregar a planilha 'Cabecalho_da_Nota (5).xlsx': {e}. Certifique-se de que o arquivo está na mesma pasta que este script.")
        st.stop()
        
    df.columns = [str(c).strip() for c in df.columns]
    
    col_map = {
        'Descrição (Tipo de Operação)': 'Tipo_Operacao',
        'Descrição (Centro de Resultado)': 'Centro_Resultado',
        'Razão Social': 'Cliente',
        'Vlr. Nota': 'Valor',
        'Qtd. volumes': 'Volumes',
        'Peso bruto': 'Peso'
    }
    df = df.rename(columns=col_map)
    
    if 'Tipo_Operacao' in df.columns:
        df = df[df['Tipo_Operacao'].astype(str).str.strip() != "*"].copy()
    
    def parse_num(val):
        if isinstance(val, (int, float)) and not pd.isna(val):
            return float(val)
        s = str(val).strip()
        if s in ("*", "", "nan", "None"):
            return np.nan
        if "." in s and "," in s:
            s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return np.nan

    for c in ['Valor', 'Volumes', 'Peso']:
        if c in df.columns:
            df[c] = df[c].apply(parse_num)
        
    # Preenche valores vazios em Volumes e Peso para evitar erros no Plotly
    if 'Volumes' in df.columns:
        df['Volumes'] = df['Volumes'].fillna(1)
    if 'Peso' in df.columns:
        df['Peso'] = df['Peso'].fillna(0)

    return df.dropna(subset=['Valor']).copy()

df = load_data()

# Filtros Laterais
st.sidebar.header("🎯 Filtros Globais")
centros_opt = sorted(df['Centro_Resultado'].dropna().unique())
tipos_opt = sorted(df['Tipo_Operacao'].dropna().unique())

centros = st.sidebar.multiselect("Centro de Resultado:", options=centros_opt, default=centros_opt)
tipos = st.sidebar.multiselect("Tipo de Operação:", options=tipos_opt, default=tipos_opt)

df_filtered = df[(df['Centro_Resultado'].isin(centros)) & (df['Tipo_Operacao'].isin(tipos))]

st.title("📊 Dashboard de Vendas, Estatística & Machine Learning")

if df_filtered.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
else:
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faturamento Total", f"R$ {df_filtered['Valor'].sum():,.2f}")
    c2.metric("Total de Operações", f"{len(df_filtered):,}")
    c3.metric("Ticket Médio", f"R$ {df_filtered['Valor'].mean():,.2f}")
    c4.metric("Peso Total", f"{df_filtered['Peso'].sum():,.1f} kg")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📈 Visão Geral & Pareto", "🔬 Análise Estatística", "🤖 Machine Learning"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            top_cr = df_filtered.groupby('Centro_Resultado')['Valor'].sum().reset_index().sort_values('Valor', ascending=True).tail(10)
            st.plotly_chart(px.bar(top_cr, x='Valor', y='Centro_Resultado', orientation='h', title="Top Centros de Resultado", color_discrete_sequence=['#1F4E79']), use_container_width=True)
        with col_b:
            top_cli = df_filtered.groupby('Cliente')['Valor'].sum().reset_index().sort_values('Valor', ascending=False).head(10)
            st.plotly_chart(px.bar(top_cli, x='Cliente', y='Valor', title="Top 10 Clientes por Valor", color_discrete_sequence=['#C55A11']), use_container_width=True)

        # Pareto
        cli_p = df_filtered.groupby('Cliente')['Valor'].sum().sort_values(ascending=False).reset_index()
        cli_p['Cum'] = (cli_p['Valor'].cumsum() / cli_p['Valor'].sum()) * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=cli_p['Cliente'].head(25), y=cli_p['Valor'].head(25), name="Valor (R$)", marker_color="#1F4E79"))
        fig_p.add_trace(go.Scatter(x=cli_p['Cliente'].head(25), y=cli_p['Cum'].head(25), name="% Acumulado", yaxis="y2", line=dict(color="#C55A11", width=2)))
        fig_p.update_layout(title="Curva ABC / Pareto (Top 25 Clientes)", yaxis2=dict(overlaying="y", side="right", range=[0, 105]))
        st.plotly_chart(fig_p, use_container_width=True)

    with tab2:
        st.write("### Estatísticas Descritivas")
        st.dataframe(df_filtered[['Valor', 'Volumes', 'Peso']].describe().T)
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.plotly_chart(px.box(df_filtered, y='Valor', points="outliers", log_y=True, title="Distribuição do Valor (Escala Log)"), use_container_width=True)
        with col_e2:
            # Filtro para ignorar linhas com NaNs restantes nas variáveis do gráfico
            df_scatter = df_filtered.dropna(subset=['Peso', 'Valor', 'Volumes'])
            st.plotly_chart(px.scatter(df_scatter, x='Peso', y='Valor', size='Volumes', color='Centro_Resultado', title="Peso (kg) vs. Valor (R$)"), use_container_width=True)

    with tab3:
        st.subheader("1. Segmentação de Clientes (K-Means Clustering)")
        rfm = df.groupby('Cliente').agg(
            Faturamento=('Valor', 'sum'),
            Frequencia=('Valor', 'count'),
            Peso_Medio=('Peso', 'mean')
        ).fillna(0)
        
        if len(rfm) >= 3:
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)
            kmeans = KMeans(n_clusters=3, random_state=42)
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled).astype(str)
            
            st.plotly_chart(px.scatter_3d(rfm.reset_index(), x='Frequencia', y='Faturamento', z='Peso_Medio', color='Cluster', hover_name='Cliente', title="Agrupamento de Clientes (Clusters)"), use_container_width=True)
        else:
            st.info("Dados insuficientes para gerar os grupos de clustering.")
        
        st.divider()
        st.subheader("2. Simulador Preditivo de Vendas (Random Forest)")
        df_ml = df_filtered.dropna(subset=['Valor', 'Volumes', 'Peso']).copy()
        
        if len(df_ml) > 10:
            X = pd.get_dummies(df_ml[['Centro_Resultado', 'Tipo_Operacao', 'Volumes', 'Peso']], drop_first=True)
            y = df_ml['Valor']
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            c_sim1, c_sim2, c_sim3, c_sim4 = st.columns(4)
            s_cr = c_sim1.selectbox("Centro:", sorted(df['Centro_Resultado'].dropna().unique()))
            s_tipo = c_sim2.selectbox("Operação:", sorted(df['Tipo_Operacao'].dropna().unique()))
            s_vol = c_sim3.number_input("Volumes:", min_value=1.0, value=5.0)
            s_peso = c_sim4.number_input("Peso (kg):", min_value=0.1, value=20.0)
            
            in_df = pd.DataFrame([{'Centro_Resultado': s_cr, 'Tipo_Operacao': s_tipo, 'Volumes': s_vol, 'Peso': s_peso}])
            in_enc = pd.get_dummies(in_df).reindex(columns=X.columns, fill_value=0)
            
            pred = rf.predict(in_enc)[0]
            st.success(f"💰 **Valor Estimado Preditivo da Nota:** R$ {max(0, pred):,.2f}")
        else:
            st.info("Selecione mais dados nos filtros para treinar o modelo preditivo.")