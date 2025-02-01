import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import plotly.graph_objects as go
import os

from ydata_profiling import ProfileReport
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title = 'Default Forecast',
    layout="wide",
    initial_sidebar_state='expanded')

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def carregar_arquivo(uploaded_file):
    if uploaded_file is not None:
        extensao = os.path.splitext(uploaded_file.name)[1].lower()

        funcoes_leitura = {
            '.csv': lambda x: pd.read_csv(x),
            '.xlsx': lambda x: pd.read_excel(x, engine='openpyxl'),
            '.xls': lambda x: pd.read_excel(x, engine='openpyxl'),
            '.ftr': lambda x: pd.read_feather(x)
        }

        if extensao in funcoes_leitura:
            df = funcoes_leitura[extensao](uploaded_file)
            st.success(f"📂 File '{uploaded_file.name}' uploaded successfully!")
            return df
        else:
            st.error(f"❌ File type '{extension}' not supported!")
            return None
    return None

def main():
    st.markdown('-----')

    st.sidebar.title("Upload your file")
    uploaded_file = st.sidebar.file_uploader("Upload a file:", type=["csv", "excel", "ftr"])
    data_file_1 = "./input/credit_scoring.ftr"


    if (data_file_1 is not None):
        df = pd.read_feather(data_file_1)

        st.title("Start and end date")
        data_inicial = st.date_input("Select start date:", df["data_ref"].min().date())
        data_final = st.date_input("Select end date:", df["data_ref"].max().date())
        data_inicial = pd.to_datetime(data_inicial).normalize()
        data_final = pd.to_datetime(data_final).normalize()
        df = df[(df["data_ref"] >= data_inicial) & (df["data_ref"] <= data_final)]

        st.write(f"Displaying data from {data_inicial.date()} until {data_final.date()}")

        metadados = pd.DataFrame({'dtypes': df.dtypes})
        metadados['missing'] = df.isna().sum()
        metadados['perc_missing'] = round((metadados['missing']/df.shape[0])*100)
        metadados['valores_unicos'] = df.nunique()

        col1, col2 = st.columns(2)

        with col1:
            mostrar_df = st.checkbox("Display DataFrame", value=True)
            if mostrar_df:
                st.subheader("DataFrame")
                st.dataframe(df)

        with col2:
            mostrar_metadados = st.checkbox("Display Metadata", value=True)
            if mostrar_metadados:
                st.subheader("Metadata")
                metadados

        st.write('---')
        
        st.markdown("## Pandas profiling analysis")

        profile = ProfileReport(df, explorative=True)

        buffer = io.BytesIO()
        profile.to_file("report.html")

        with open("report.html", "rb") as f:
            report_content = f.read()

        st.download_button(
            label="Download Report",
            data=report_content,
            file_name="profiling_report.html",
            mime="text/html"
        )


        col3, col4 = st.columns(2)
        plt.figure(figsize=(6,6))

        with col3:
            st.subheader("Univariate Analysis")
            var_uni = st.selectbox("Select the variable for univariate analysis:", df.columns.drop(['df_index','data_ref']))
            
            fig_uni, ax_uni = plt.subplots()
            if df[var_uni].dtype == "object":
                sns.countplot(x=var_uni, data=df, ax=ax_uni)
            else:
                sns.histplot(df[var_uni], kde=True, ax=ax_uni)
            st.pyplot(fig_uni)

        with col4:
            st.subheader("Bivariate Analysis")
            var_bi2 = st.selectbox("Select the second variable:", df.columns.drop(['df_index', 'data_ref']))

            fig_bi, ax_bi = plt.subplots()
            if df[var_uni].dtype == "object" or df[var_bi2].dtype == "object":
                sns.boxplot(x=var_uni, y=var_bi2, data=df, ax=ax_bi)
            else:
                sns.scatterplot(x=var_uni, y=var_bi2, data=df, ax=ax_bi)
            st.pyplot(fig_bi)

        st.write("----")
        st.title("Distribution of Variables Over Time:")

        variavel = st.selectbox("Select variable:", df.columns.drop(['df_index','data_ref']))

        if df[variavel].dtypes == "float64" or "int64":
            st.subheader(f"Distribution of the variable '{variavel}' over time")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x="data_ref", y=variavel, data=df, ax=ax)
            ax.set_title(f"Trend of {variavel} over time")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        elif df[variavel].dtypes == "object" or "bool":
            st.subheader(f"Distribution of the variable '{variavel}' over time")
            if df[variavel].dtype == "object":
                df_counts = df.groupby(["data_ref", variavel]).size().reset_index(name="Contagem")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(data=df_counts, x="Data", weights="Contagem", hue=variavel, multiple="stack", ax=ax)
                ax.set_title(f"Distribution of {variavel} over time")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.error("Select a valid qualitative variable.")
                
        st.write("----")

        st.title("Stability Chart")
        variavel = st.selectbox("Choose the variable for the stability graph:", df.columns.drop(['df_index', 'data_ref']))

        if df[variavel].dtype in ["int64", "float64"]: 
            st.subheader(f"Stability Chart for {variavel}")

            media = df[variavel].mean()
            std = df[variavel].std()
            limite_superior = media + 2 * std
            limite_inferior = media - 2 * std

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["data_ref"],
                y=df[variavel],
                mode="lines+markers",
                name="Values",
                line=dict(color="blue"),
                marker=dict(size=5)
            ))

            fig.add_trace(go.Scatter(
                x=df["data_ref"],
                y=[media] * len(df),
                mode="lines",
                name="Average",
                line=dict(color="green", dash="dash")
            ))

            fig.add_trace(go.Scatter(
                x=df["data_ref"],
                y=[limite_superior] * len(df),
                mode="lines",
                name="Upper Limit",
                line=dict(color="red", dash="dot")
            ))
            fig.add_trace(go.Scatter(
                x=df["data_ref"],
                y=[limite_inferior] * len(df),
                mode="lines",
                name="Lower Limit",
                line=dict(color="red", dash="dot")
            ))

            fig.update_layout(
                title=f"Stability Chart: {variavel}",
                xaxis_title="data_ref",
                yaxis_title=variavel,
                template="plotly_white",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif df[variavel].dtype == "object":
            st.subheader(f"Stability Chart for {variavel}")

            df_grouped = df.groupby(["data_ref", variavel]).size().reset_index(name="Count")

            fig = go.Figure()

            for categoria in df[variavel].unique():
                subset = df_grouped[df_grouped[variavel] == categoria]
                fig.add_trace(go.Bar(
                    x=subset["data_ref"],
                    y=subset["Count"],
                    name=str(categoria)
                ))

            fig.update_layout(
                title=f"Distribution of Categories Over Time: {variavel}",
                xaxis_title="data_ref",
                yaxis_title="Count",
                barmode="stack",
                template="plotly_white",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("The selected variable is not valid for this graph. Choose quantitative or categorical.")

        st.write("----")

        st.title("Default Forecast")
    
        st.write("This application performs predictions using a lightgbm model.")

        modelo_carregado = load_model("model_finalizado")
        
        col5, col6 = st.columns(2)

        if uploaded_file is not None:
            df_predicao = carregar_arquivo(uploaded_file)

            with col5:
                mostrar_df = st.checkbox("DataFrame used for model construction", value=True)
                st.dataframe(df)

            with col6:
                mostrar_metadados = st.checkbox("DataFrame for prediction", value=True)
                st.dataframe(df_predicao)
            
            

            if st.sidebar.button("Predict"):
                predicoes = modelo_carregado.predict(df_predicao[['posse_de_veiculo', 'tempo_emprego', 'renda']])
                df_predicao["Prediction"] = predicoes

                st.write("### DataFrame with Predictions:")
                st.dataframe(df_predicao)

                csv_download = df_predicao.to_csv(index=False)
                st.download_button(
                        label="Download Result with Predictions",
                        data=csv_download,
                        file_name="resultado_com_predicoes.csv",
                        mime="text/csv",
                    )
        else:
            st.info("Please upload a CSV file to continue.")

if __name__ == '__main__':
    main()