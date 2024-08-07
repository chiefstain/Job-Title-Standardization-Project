from googletrans import Translator

translator = Translator()

def translate_title(title, src_lang='auto', dest_lang='en'):
    translation = translator.translate(title, src=src_lang, dest=dest_lang)
    return translation.text
