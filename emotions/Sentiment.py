from textblob_fr import PatternTagger, PatternAnalyzer
import os
from textblob import TextBlob
import csv
from collections import defaultdict

stopwords = ["à", "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", "bon", "car", "ce", "cela",
             "ces", "ceux", "chaque", "ci", "comme", "comment", "dans", "des", "du", "dedans", "dehors",
             "depuis", "devrait", "doit", "donc", "début", "elle", "elles", "en", "encore", "essai", "est", "et", "eu",
             "fait", "faites", "fois", "font", "hors", "ici", "il", "ils", "je", "juste", "la", "le", "les",
             "leur", "là", "ma", "maintenant", "mais", "mes", "mien", "moins", "mon", "même", "ni", "notre", "nous",
             "ou", "où", "par", "parce", "pas", "peut", "peu", "plupart", "pour", "pourquoi", "quand",
             "que", "quel", "quelle", "quelles", "quels", "qui", "sa", "sans", "ses", "seulement", "si", "sien", "son",
             "sont", "sous", "soyez", "sujet", "sur", "ta", "tandis", "tellement", "tels", "tes", "ton", "tous",
             "tout", "trop", "très", "tu", "voient", "vont", "votre", "vous", "vu", "ça", "étaient", "état", "étions",
             "été", "être"]


def sentiment(phrase):
    textfr = ''
    # print(phrase)
    sentiment = [0, 0]
    for i in range(len(phrase)):
        # print(phrase[i])
        textfr += ' ' + phrase[i].get('lemma', phrase[i].get('text'))  # form
        blob = TextBlob(textfr, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        polarity = blob.sentiment[0]
        subjectivity = blob.sentiment[1]
        sentiment = [polarity, polarity]
        if polarity > 1 or subjectivity < -1:
            sentiment[0] = 0
        if subjectivity > 1 or subjectivity < 0:
            sentiment[1] = 0
    return sentiment


def getLexicon(f_path):
    """
    Entrée : chemin du fichier .tsv avec le lexique
    Sortie : le lexique, sous forme de dicos
    1. dico_cat_terms : {"catégorie" : ["terme1", "terme 2"]}
    2. dico_terms_cat : {"terme1" : "catégorie", "terme2" : "catégorie"}
    :param f_path: str
    :return: tuple
    """

    dico_cat_terms = defaultdict(lambda: list())
    dico_terms_cat = dict()

    with open(f_path, 'r', encoding="utf-8") as in_stream:
        in_stream.readline()  # on évacue la 1ère ligne, celle avec les en-têtes de colonne
        for line in in_stream:  # pour chaque ligne du fichier
            elements = line.strip().split("\t")  # on récupère le terme et la catégorie
            dico_cat_terms[elements[1]].append(
                elements[0])  # ajout du terme dans la liste de termes associés à une catégorie
            dico_terms_cat[elements[0]] = elements[1]  # ajout du couple terme/catégorie

    return dico_cat_terms, dico_terms_cat


def DiffEmotionsList(Lexicon_term_cat):
    EmotionsList = []
    for emotion in Lexicon_term_cat.values():
        if emotion == 'dégoût':
            emotion = 'degout'
        if emotion not in EmotionsList:
            EmotionsList += [emotion]
    return ['neutre'] + EmotionsList


def EmotionMotVector(mot, Lexicon_term_cat):  # Vecteur One hot des émotions du mot
    EmotionList = DiffEmotionsList(Lexicon_term_cat)
    EmotionVector = [0] * len(EmotionList)
    Mot_Emotion = Lexicon_term_cat.get(mot, 'neutre')
    if Mot_Emotion == 'dégoût':
        Mot_Emotion = 'degout'
    EmotionVector[EmotionList.index(Mot_Emotion)] = 1
    return EmotionVector


def EmotionVectorPhrase(phrase, Lexicon_term_cat):
    motsPorteursEmotion = []
    for i in range(len(phrase)):
        mot = phrase[i].get('lemma', phrase[i].get('text'))  # form
        if mot not in stopwords:
            motsPorteursEmotion += [mot]
    EmotionPhraseVector = [0] * len(DiffEmotionsList(Lexicon_term_cat))
    if not motsPorteursEmotion:
        return EmotionPhraseVector
    else:
        for i in range(len(EmotionPhraseVector)):
            sumEmo_i = 0
            for j in range(len(motsPorteursEmotion)):
                CurrentMotEmotion = EmotionMotVector(motsPorteursEmotion[j], Lexicon_term_cat)
                sumEmo_i += CurrentMotEmotion[i]
            EmotionPhraseVector[i] = sumEmo_i / len(motsPorteursEmotion)
        return EmotionPhraseVector


def CsvMakerPhrases(ListOfPhrases):
    print(os.getcwd())
    Lexicon = getLexicon("../tools/Lexique/lexique_emotionnel.tsv")[1]
    ListfeaturesPhrases = ['score_de_polarite', 'score_de_subjectitvite'] + DiffEmotionsList(Lexicon)
    ListfeaturesEX = []
    for i in range(len(ListOfPhrases)):
        phrase = ListOfPhrases[i]
        ListfeaturesEX += [[sentiment(phrase)[0], sentiment(phrase)[1]] + EmotionVectorPhrase(phrase, Lexicon)]
    if 'SentimentFeatures.csv' not in os.listdir(os.getcwd()):
        with open('SentimentFeatures.csv', 'w', newline='', encoding='utf-8') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter="\t")
            spamwriter.writerow(ListfeaturesPhrases)
    with open('SentimentFeatures.csv', 'a', newline='', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter="\t")
        for i in range(len(ListfeaturesEX)):
            spamwriter.writerow(ListfeaturesEX[i])
    return 'SentimentFeatures.csv'

