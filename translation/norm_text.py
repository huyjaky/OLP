import re 
import string 
from spacy.lang.vi import Vietnamese


nlp = Vietnamese()

def text_normalize(
    text,
    remove_urls=True,
    remove_emails=True,
    remove_mentions=True,
    remove_hashtags=True,
    remove_numbers=False,
):
    try:
        text = str(text)
        # basic normalization
        text = text.strip().lower()

        # remove urls
        if remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # remove emails
        if remove_emails:
            text = re.sub(r"\S+@\S+", " ", text)
        # remove html tags
        text = re.sub(r"<.*?>", " ", text)
        # mentions (@username)
        if remove_mentions:
            text = re.sub(r"@\w+", " ", text)
        # hashtags: keep the word but drop the #
        if remove_hashtags:
            text = re.sub(r"#(\w+)", r"\1", text)
        # remove phone numbers (simple patterns)
        text = re.sub(r"\b0\d{8,}\b|\b\d{9,}\b", " ", text)
        # replace common currency symbols with space
        text = re.sub(r"[\$€£¥₫]", " ", text)

        # remove punctuation (translate to space to avoid joining words)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)

        # remove non-printable characters
        text = "".join(ch for ch in text if ch.isprintable())

        # collapse repeated punctuation (e.g., '!!!' -> '!')
        text = re.sub(r"([!?.]){2,}", r"\1", text)
        # reduce elongated characters (>2 repeats -> 2 repeats) e.g. heyyyy -> heyy
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # optionally remove digits entirely
        if remove_numbers:
            text = re.sub(r"\d+", " ", text)

        # normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # tokenize with spaCy (Vietnamese) and filter stopwords/punct/space
        doc = nlp(text)
        cleaned_tokens = []
        for token in doc:
            # Keep tokens that are not stopwords/punctuation/space
            if not token.is_stop and not token.is_punct and not token.is_space:
                # filter out tokens that contain digits (optional already handled above)
                if not any(char.isdigit() for char in token.text):
                    cleaned_tokens.append(token.lower_)

        return " ".join(cleaned_tokens)

    except Exception as e:
        print("Error normalizing text:", e)
        return ""
