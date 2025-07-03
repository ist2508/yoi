
import streamlit as st
import pandas as pd
import os
from preprocessing import run_full_preprocessing
from labeling import run_labeling
from balancing import run_balancing
from modeling import run_naive_bayes
from visualization import create_wordcloud, plot_sentiment_distribution, plot_top_words
from utils import read_csv_safely
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import nltk
nltk.download('stopwords')

st.set_page_config(page_title="Analisis Sentimen Menggunakan Naive Bayes", layout="wide")
st.title("📊 Analisis Sentimen Menggunakan Naive Bayes")

upload_tab, preprocess_tab, label_tab, balance_tab, model_tab, visual_tab = st.tabs([
    "📂 Upload Data",
    "🔄 Preprocessing",
    "🏷️ Labeling",
    "⚖️ Balancing",
    "📈 Model Naive Bayes",
    "🖼️ Visualisasi"
])

with upload_tab:
    st.subheader("📂 Unggah File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV Tweet", type="csv")
    if uploaded_file is not None:
        with open("dataMakanSiangGratis.csv", "wb") as f:
            f.write(uploaded_file.read())
        st.session_state['data_uploaded'] = True
        st.success("✅ File berhasil diunggah. Silakan lanjut ke tab berikutnya.")

with preprocess_tab:
    st.subheader("🔄 Tahap Preprocessing")
    if st.button("🚀 Jalankan Preprocessing"):
        if not st.session_state.get('data_uploaded', False):
            st.error("❌ Anda harus mengunggah data terlebih dahulu sebelum menjalankan preprocessing.")
        else:
            with st.spinner("Sedang memproses data..."):
                df_preprocessed = run_full_preprocessing("dataMakanSiangGratis.csv")
                st.session_state.df_preprocessed = df_preprocessed
                os.makedirs("hasil", exist_ok=True)
                df_preprocessed.to_csv("hasil/hasil_preprocessing.csv", index=False)
                st.success("✅ Preprocessing selesai.")

    if 'df_preprocessed' in st.session_state:
        with st.expander("📄 Lihat Hasil Preprocessing"):
            st.dataframe(st.session_state.df_preprocessed.head())
        if os.path.exists("hasil/hasil_preprocessing.csv"):
            with open("hasil/hasil_preprocessing.csv", "rb") as f:
                st.download_button("⬇️ Unduh Hasil Preprocessing", f, file_name="hasil_preprocessing.csv", mime="text/csv")

with label_tab:
    st.subheader("🏷️ Tahap Labeling Sentimen")
    if st.button("🏷️ Jalankan Labeling"):
        if 'df_preprocessed' not in st.session_state:
            st.error("❌ Lakukan preprocessing terlebih dahulu sebelum melakukan labeling.")
        else:
            with st.spinner("Menentukan sentimen berdasarkan lexicon..."):
                df_labelled = run_labeling()
                st.session_state.df_labelled = df_labelled
                os.makedirs("hasil", exist_ok=True)
                df_labelled.to_csv("hasil/hasil_labeling.csv", index=False)
                st.success("✅ Labeling selesai.")

    if 'df_labelled' in st.session_state:
        with st.expander("📄 Lihat Hasil Labeling"):
            st.dataframe(st.session_state.df_labelled.head())
        if os.path.exists("hasil/hasil_labeling.csv"):
            with open("hasil/hasil_labeling.csv", "rb") as f:
                st.download_button("⬇️ Unduh Hasil Labeling", f, file_name="hasil_labeling.csv", mime="text/csv")

with balance_tab:
    st.subheader("⚖️ Tahap Balancing Dataset")
    if st.button("⚖️ Jalankan Balancing"):
        if 'df_labelled' not in st.session_state:
            st.error("❌ Anda harus melakukan labeling sebelum menjalankan balancing.")
        else:
            with st.spinner("Menyeimbangkan jumlah data pada tiap kelas sentimen..."):
                df_balanced, fig = run_balancing()
                st.session_state.df_balanced = df_balanced
                os.makedirs("hasil", exist_ok=True)
                df_balanced.to_csv("hasil/hasil_balancing.csv", index=False)
                st.success("✅ Balancing selesai.")
                st.pyplot(fig)

    if 'df_balanced' in st.session_state:
        with st.expander("📄 Lihat Hasil Balancing"):
            st.dataframe(st.session_state.df_balanced.head())
        if os.path.exists("hasil/hasil_balancing.csv"):
            with open("hasil/hasil_balancing.csv", "rb") as f:
                st.download_button("⬇️ Unduh Hasil Balancing", f, file_name="hasil_balancing.csv", mime="text/csv")

with model_tab:
    st.subheader("📈 Naive Bayes (Multinomial)")
    if st.button("🔍 Jalankan Model Naive Bayes"):
        if 'df_balanced' not in st.session_state:
            st.error("❌ Anda harus melakukan balancing sebelum menjalankan model.")
        else:
            with st.spinner("Melatih dan mengevaluasi model..."):
                accuracy, report, conf_matrix, result_df, *_ , x_train_len, x_test_len = run_naive_bayes()
                st.session_state.accuracy = accuracy
                st.session_state.report = report
                st.session_state.df_pred = result_df
                st.session_state.x_train_len = x_train_len
                st.session_state.x_test_len = x_test_len
                result_df.to_csv("hasil/Hasil_pred_MultinomialNB.csv", index=False)
                st.success(f"✅ Akurasi Model: {accuracy:.2f}")

    if 'df_pred' in st.session_state:
        st.subheader("📊 Hasil Splitting Dataset")
        st.text(f"Jumlah Data Latih: {st.session_state.x_train_len}")
        st.text(f"Jumlah Data Uji: {st.session_state.x_test_len}")

        with st.expander("📊 Laporan Evaluasi"):
            report_dict = classification_report(
                st.session_state.df_pred['Actual'],
                st.session_state.df_pred['Predicted'],
                output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose()
            selected_cols = ['precision', 'recall', 'f1-score', 'support']
            report_df = report_df[selected_cols].round(2)
            st.dataframe(report_df)

        with st.expander("📄 Hasil Prediksi"):
            st.dataframe(st.session_state.df_pred.head())

        st.subheader("📊 Diagram Batang Prediksi Sentimen")
        sentiment_distribution = st.session_state.df_pred['Predicted'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(sentiment_distribution.index, sentiment_distribution.values, color=['green', 'orange', 'red'])
        ax.set_title('Diagram Batang Hasil Analisis Sentimen Menggunakan Naive Bayes')
        ax.set_xlabel('Sentimen Prediksi')
        ax.set_ylabel('Jumlah Tweet')
        ax.set_xticks(range(len(sentiment_distribution.index)))
        ax.set_xticklabels(sentiment_distribution.index)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 5, round(yval, 2), ha='center', va='bottom')

        st.pyplot(fig)

    hasil_file = "hasil/Hasil_pred_MultinomialNB.csv"
    if os.path.exists(hasil_file):
        with open(hasil_file, "rb") as f:
            st.download_button("⬇️ Unduh Hasil Prediksi", f, file_name="hasil_sentimen.csv", mime="text/csv")

with visual_tab:
    st.subheader("🖼️ Visualisasi Sentimen dan Kata Berdasarkan Hasil Prediksi")
    if 'df_pred' not in st.session_state:
        st.error("❌ Silakan jalankan model terlebih dahulu sebelum melihat visualisasi.")
    else:
        df_vis = read_csv_safely("hasil/Hasil_pred_MultinomialNB.csv")
        if df_vis is not None:
            os.makedirs("hasil", exist_ok=True)
            plot_sentiment_distribution(df_vis)

            text_neg = ' '.join(df_vis[df_vis['Predicted'] == 'Negatif']['steming_data'].dropna())
            if text_neg.strip():
                create_wordcloud(text_neg, 'wordcloud_negatif.png')

            text_net = ' '.join(df_vis[df_vis['Predicted'] == 'Netral']['steming_data'].dropna())
            if text_net.strip():
                create_wordcloud(text_net, 'wordcloud_netral.png')

            text_pos = ' '.join(df_vis[df_vis['Predicted'] == 'Positif']['steming_data'].dropna())
            if text_pos.strip():
                create_wordcloud(text_pos, 'wordcloud_positif.png')

            plot_top_words(df_vis, 'Negatif', 'top_words_negatif.png')
            plot_top_words(df_vis, 'Netral', 'top_words_netral.png')
            plot_top_words(df_vis, 'Positif', 'top_words_positif.png')
            st.session_state.show_visual = True

        if st.session_state.get("show_visual"):
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists("hasil/wordcloud_negatif.png"):
                    st.image("hasil/wordcloud_negatif.png", caption="WordCloud Negatif")
                if os.path.exists("hasil/top_words_negatif.png"):
                    st.image("hasil/top_words_negatif.png", caption="Top Words Negatif")
                if os.path.exists("hasil/wordcloud_netral.png"):
                    st.image("hasil/wordcloud_netral.png", caption="WordCloud Netral")
                if os.path.exists("hasil/top_words_netral.png"):
                    st.image("hasil/top_words_netral.png", caption="Top Words Netral")
            with col2:
                if os.path.exists("hasil/wordcloud_positif.png"):
                    st.image("hasil/wordcloud_positif.png", caption="WordCloud Positif")
                if os.path.exists("hasil/top_words_positif.png"):
                    st.image("hasil/top_words_positif.png", caption="Top Words Positif")

            if os.path.exists("hasil/sentimen_distribution.png"):
                st.image("hasil/sentimen_distribution.png", caption="Distribusi Sentimen")
            if os.path.exists("hasil/conf_matrix_mnb.png"):
                st.image("hasil/conf_matrix_mnb.png", caption="Confusion Matrix MultinomialNB")
