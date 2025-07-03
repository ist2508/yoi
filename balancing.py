import pandas as pd
import matplotlib.pyplot as plt

def run_balancing():
    # 1. Baca data hasil labeling
    df = pd.read_csv("Hasil_Labelling_Data.csv")

    # 2. Tentukan jumlah minimum dari semua kelas sentimen
    min_jumlah = df['Sentiment'].value_counts().min()

    # 3. Ambil sampel yang seimbang dari setiap kelas
    df_pos = df[df['Sentiment'] == 'Positif'].sample(min_jumlah, random_state=42)
    df_neg = df[df['Sentiment'] == 'Negatif'].sample(min_jumlah, random_state=42)
    df_net = df[df['Sentiment'] == 'Netral'].sample(min_jumlah, random_state=42)

    # 4. Gabungkan kembali dan acak
    df_balanced = pd.concat([df_pos, df_neg, df_net]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Simpan hasil balancing
    df_balanced.to_csv("Hasil_Labeling_Seimbang.csv", index=False)

    # 6. Buat visualisasi distribusi sebelum dan sesudah balancing
    before = df['Sentiment'].value_counts()
    after = df_balanced['Sentiment'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sebelum balancing
    bars1 = axes[0].bar(before.index, before.values, color=['green', 'gray', 'red'])
    axes[0].set_title("Distribusi Sebelum Balancing")
    axes[0].set_xlabel("Sentimen")
    axes[0].set_ylabel("Jumlah")
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 2, str(int(yval)), ha='center', va='bottom')

    # Sesudah balancing
    bars2 = axes[1].bar(after.index, after.values, color=['green', 'gray', 'red'])
    axes[1].set_title("Distribusi Setelah Balancing")
    axes[1].set_xlabel("Sentimen")
    axes[1].set_ylabel("Jumlah")
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 2, str(int(yval)), ha='center', va='bottom')

    plt.tight_layout()

    # 7. Kembalikan hasil balancing dan figure
    return df_balanced, fig
