from stanza.pipeline.processor import Processor, register_processor
from stanza.models.common.doc import Document

processor_name = 'emotions'

lexicon = Sentiment.getLexicon("inria_processors/tools/Lexique/lexique_emotionnel.tsv")[1]
features_list = ['score_de_polarite', 'score_de_subjectitvite'] + Sentiment.DiffEmotionsList(lexicon)


def setFeatures(input):

    #conll format conversion
    if isinstance(input, Document):
        conll_text = input.to_dict()
    else:
        conll_text = [input.to_dict()]

    #flatten list of sentences if input is doc
    if isinstance(conll_text, list):
        conll_text = [item for sublist in conll_text for item in sublist]

    setattr(input, features_list[0], Sentiment.sentiment(conll_text)[0])
    setattr(input, features_list[1], Sentiment.sentiment(conll_text)[1])
    other_features = zip(features_list[2:], Sentiment.EmotionVectorPhrase(conll_text, lexicon))
    for f, s in other_features:
        setattr(input, f, s)    


@register_processor(processor_name)
class EmotionsProcessor(Processor):
    """ Processor that computes emotions features at sentence and text levels """
    _provides = {processor_name}

    def __init__(self, device, config, pipeline):
        if "analysis_level" in pipeline.kwargs:
            self.levels = pipeline.kwargs["analysis_level"].split(",")
        pass

    def _set_up_model(self, *args):
        pass

    def process(self, doc):

        if "text" in self.levels:
            # print("emotion_texte")
            # print(doc)
            # print(features_list)
            processor_export_format.add_config(doc, processor_name+"_texte", "text", "csv", features_list)
            setFeatures(doc)

        if "sentence" in self.levels:
            # print("emotion_sentence")
            # print(doc)
            # print(features_list)
            processor_export_format.add_config(doc, processor_name+"_phrase", "sentence", "csv", features_list)
            for sentence in doc.sentences:
                setFeatures(sentence)

        return doc
