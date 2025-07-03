import pandas as pd
import requests

def run_labeling():
    df = pd.read_csv("Hasil_Preprocessing_Data.csv", encoding='latin-1')
    df = df.dropna()
    df = df[['date', 'time', 'steming_data']]

    # Ambil data lexicon
    positive_url = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
    negative_url = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
    positive_lexicon = set(pd.read_csv(positive_url, sep="\t", header=None)[0])
    negative_lexicon = set(pd.read_csv(negative_url, sep="\t", header=None)[0])

    def determine_sentiment(text):
        if isinstance(text, str):
            pos_count = sum(1 for word in text.split() if word in positive_lexicon)
            neg_count = sum(1 for word in text.split() if word in negative_lexicon)
            score = pos_count - neg_count
            if score > 0:
                return score, "Positif"
            elif score < 0:
                return score, "Negatif"
            else:
                return score, "Netral"
        return 0, "Netral"

    df[['Score', 'Sentiment']] = df['steming_data'].apply(lambda x: pd.Series(determine_sentiment(x)))
    df.to_csv("Hasil_Labelling_Data.csv", index=False, encoding='utf8')

    return df 