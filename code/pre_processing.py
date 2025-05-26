import unicodedata
from typing import Literal, Optional
import re

REMOVE_WHITESPACES_PATTERN = (r"\s+", " ")  # Match all whitespace and replace with single space

# Replace all consecutive occurrences of same punctuation with single occurrence
# except for multiple '.' that are replaced with '...'
REMOVE_PUNCTUATION_PATTERN = (
    r'([!\"#$%&\'()*+,\-/:;<=>?@[\\\]^_`{|}~])\1+|\.{2,}',
    lambda m: m.group(1) if m.group(1) else '...'
)

# dict of common contractions in english and their long form
CONTRACTIONS_DICT = \
    {"'aight": "alright",
     "ain't": "are not",
     "amn't": "am not",
     "aren't": "are not",
     "can't": "can not",
     "'cause": "because",
     "could've": "could have",
     "couldn't": "could not",
     "couldn't've": "could not have",
     "daren't": "dare not",
     "daresn't": "dare not",
     "didn't": "did not",
     "doesn't": "does not",
     "don't": "do not",
     "d'ye": "do you",
     "e'er": "ever",
     "everybody's": "everybody is",
     "everyone's": "everyone is",
     "finna": "going to",
     "g'day": "good day",
     "gimme": "give me",
     "giv'n": "given",
     "gonna": "going to",
     "gotta": "got to",
     "hadn't": "had not",
     "had've": "had have",
     "hasn't": "has not",
     "haven't": "have not",
     "he'd": "he had",
     "he's": "he is",
     "he've": "he have",
     "how'd": "how did",
     "howdy": "how do you do",
     "how'll": "how shall",
     "how're": "how are",
     "how's": "how has",
     "i'd": "i had",
     "i'd've": "i would have",
     "i'll": "i will",
     "i've": "i have",
     "i'd've": "i would have",
     "i'll": "i will",
     "i'm": "i am",
     "i'm'a": "i am about to",
     "i'm'o": "i am going to",
     "innit": "is it not",
     "i've": "i have",
     "isn't": "is not",
     "it'd": "it would",
     "it'll": "it will",
     "it's": "it is",
     "let's": "let us",
     "ma'am": "madam",
     "mayn't": "may not",
     "may've": "may have",
     "methinks": "me thinks",
     "mightn't": "might not",
     "might've": "might have",
     "mustn't": "must not",
     "mustn't've": "must not have",
     "must've": "must have",
     "needn't": "need not",
     "ne'er": "never",
     "o'clock": "of the clock",
     "o'er": "over",
     "ol'": "old",
     "oughtn't": "ought not",
     "'s": "is",
     "shalln't": "shall not",
     "shan't": "shall not",
     "she'd": "she had",
     "she'll": "she will",
     "she's": "she is",
     "should've": "should have",
     "shouldn't": "should not",
     "shouldn't've": "should not have",
     "somebody's": "somebody has",
     "someone's": "someone has",
     "something's": "something has",
     "so're": "so are",
     "that'll": "that shall",
     "that're": "that are",
     "that'd": "that would",
     "there'd": "there had",
     "there'll": "there shall",
     "there're": "there are",
     "there's": "there has",
     "these're": "these are",
     "these've": "these have",
     "they'd": "they had",
     "they'll": "they will",
     "they're": "they are",
     "they've": "they have",
     "this's": "this has",
     "those're": "those are",
     "those've": "those have",
     "'tis": "it is",
     "to've": "to have",
     "'twas": "it was",
     "wanna": "want to",
     "wasn't": "was not",
     "we'd": "we had",
     "we'd've": "we would have",
     "we'll": "we will",
     "we're": "we are",
     "we've": "we have",
     "weren't": "were not",
     "what'd": "what did",
     "what'll": "what shall",
     "what're": "what are",
     "what's": "what has",
     "what've": "what have",
     "when's": "when has",
     "where'd": "where did",
     "where'll": "where shall",
     "where're": "where are",
     "where's": "where has",
     "where've": "where have",
     "which'd": "which had",
     "which'll": "which shall",
     "which're": "which are",
     "which's": "which has",
     "which've": "which have",
     "who'd": "who would",
     "who'd've": "who would have",
     "who'll": "who shall",
     "who're": "who are",
     "who's": "who has",
     "who've": "who have",
     "why'd": "why did",
     "why're": "why are",
     "why's": "why has",
     "won't": "will not",
     "would've": "would have",
     "wouldn't": "would not",
     "wouldn't've": "would not have",
     "y'all": "you all",
     "y'all'd've": "you all would have",
     "y'all're": "you all are",
     "you'd": "you had",
     "you'll": "you will",
     "you're": "you are",
     "you've": "you have",
     "aight": "alright",
     "aint": "are not",
     "amnt": "am not",
     "arent": "are not",
     "cant": "can not",
     "cause": "because",
     "couldve": "could have",
     "couldnt": "could not",
     "couldntve": "could not have",
     "darent": "dare not",
     "daresnt": "dare not",
     "dasnt": "dare not",
     "didnt": "did not",
     "doesnt": "does not",
     "dont": "do not",
     "eer": "ever",
     "everybodys": "everybody is",
     "everyones": "everyone is",
     "finna": "fixing to",
     "gday": "good day",
     "gimme": "give me",
     "givn": "given",
     "gonna": "going to",
     "gonn": "going to",
     "gotta": "got to",
     "hadnt": "had not",
     "hadve": "had have",
     "hasnt": "has not",
     "havent": "have not",
     "hed": "he had",
     "he'll": "he will",
     "hes": "he has",
     "heve": "he have",
     "howd": "how did",
     "howdy": "how do you do",
     "howll": "how will",
     "howre": "how are",
     "hows": "how has",
     "idve": "i would have",
     "ill": "i will",
     "im": "i am",
     "ima": "i am about to",
     "ive": "i have",
     "isnt": "is not",
     "itd": "it would",
     "itll": "it shall",
     "its": "it is",
     "lets": "let us",
     "maam": "madam",
     "maynt": "may not",
     "mayve": "may have",
     "methinks": "me thinks",
     "mightnt": "might not",
     "mightve": "might have",
     "mustnt": "must not",
     "mustntve": "must not have",
     "mustve": "must have",
     "neednt": "need not",
     "neer": "never",
     "oclock": "of the clock",
     "oer": "over",
     "ol": "old",
     "oughtnt": "ought not",
     "shallnt": "shall not",
     "shant": "shall not",
     "shed": "she had",
     "shell": "she will",
     "shes": "she has",
     "shouldve": "should have",
     "shouldnt": "should not",
     "shouldntve": "should not have",
     "somebodys": "somebody has",
     "someones": "someone has",
     "somethings": "something has",
     "thatll": "that shall",
     "thatre": "that are",
     "thatd": "that would",
     "thered": "there had",
     "therell": "there shall",
     "therere": "there are",
     "theres": "there has",
     "thesere": "these are",
     "theseve": "these have",
     "theyd": "they had",
     "theyll": "they shall",
     "theyre": "they are",
     "theyve": "they have",
     "thosere": "those are",
     "thoseve": "those have",
     "tis": "it is",
     "tove": "to have",
     "twas": "it was",
     "wanna": "want to",
     "wasnt": "was not",
     "wed": "we had",
     "wedve": "we would have",
     "were": "we are",
     "weve": "we have",
     "werent": "were not",
     "whatd": "what did",
     "whatll": "what shall",
     "whatre": "what are",
     "whats": "what has",
     "whatve": "what have",
     "whens": "when has",
     "whered": "where did",
     "wherell": "where shall",
     "wherere": "where are",
     "wheres": "where has",
     "whereve": "where have",
     "whichd": "which had",
     "whichll": "which shall",
     "whichre": "which are",
     "whichs": "which has",
     "whichve": "which have",
     "whod": "who would",
     "whodve": "who would have",
     "wholl": "who shall",
     "whore": "who are",
     "whos": "who has",
     "whove": "who have",
     "whyd": "why did",
     "whyre": "why are",
     "whys": "why has",
     "wont": "will not",
     "wouldve": "would have",
     "wouldnt": "would not",
     "wouldntve": "would not have",
     "yall": "you all",
     "yalldve": "you all would have",
     "yallre": "you all are",
     "youd": "you had",
     "youll": "you shall",
     "youre": "you are",
     "youve": "you have",
     "'re": "are",
     "that's": "that is",
     "thats": "that is",
     "ion": "i do not",
     "i'on": "i do not",
     "lemme": "let me",
     "outta": "out of",
     "kinda": "kind of",
     "sorta": "sort of",
     "lotta": "lot of",
     "cuzza": "because of",
     "prolly": "probably",
     "imma": "i am going to",
     "brb": "be right back",
     "bbl": "be back later",
     "b4": "before",
     "bc": "because",
     "bcoz": "because",
     "idk": "i dont know",
     "ikr": "i know right",
     "rn": "right now",
     "smh": "shaking my head",
     "wtf": "what the fuck",
     "tf": "the fuck",
     "lmk": "let me know",
     "nvm": "never mind",
     "ppl": "people",
     "yall": "you all",
     "ya": "you",
     "u": "you",
     "r": "are",
     "ur": "you are",
     "rite": "right",
     "dnt": "do not",
     "pls": "please",
     "plz": "please",
     "tho": "though",
     "bcuz": "because",
     "cuz": "because",
     "sup": "what is up",
     "ftw": "for the win",
     "omg": "oh my god",
     "gg": "good game",
     "gn": "good night",
     "gr8": "great",
     "hmu": "hit me up",
     "omw": "on my way",
     "yolo": "you only live once",
     "nah": "no",
     "yup": "yes",
     "yep": "yes",
     "naw": "no",
     "ok": "okay",
     "fr": "for real",
     "dm": "direct message",
     "fam": "close friend",
     "deadass": "seriously",
     "tho": "though",
     "thx": "thanks",
     "ty": "thank you",
     "np": "no problem",
     "lol": "laughing out loud",
     "lmao": "laughing my ass off",
     "rofl": "rolling on the floor laughing",
     "tbh": "to be honest",
     "atm": "at the moment",
     "btw": "by the way",
     "idc": "i do not care",
     "ily": "i love you",
     "ilysm": "i love you so much",
     "imho": "in my humble opinion",
     "imo": "in my opinion",
     "jk": "just kidding",
     "ttyl": "talk to you later",
     "bff": "best friends forever",
     "wyd": "what are you doing",
     "wbu": "what about you",
     "hbu": "how about you",
     "bruh": "brother",
     "bro": "brother",
     "sis": "sister",
     "cu": "see you",
     "cya": "see you",
     "gtg": "got to go",
     "g2g": "got to go",
     "nope": "no",
     "hecka": "very",
     "hella": "very",
     "obvi": "obviously",
     "tryna": "trying to",
     "wassup": "what is up",
     "whatchu": "what are you",
     "chu": "you",
     "nahh": "no",
     "yaaa": "yes",
     "vibe": "feeling",
     "woulda": "would have",
     "mighta": "might have",
     "hafta": "have to",
     "needa": "need to",
     "coulda": "could have",
     "shoulda": "should have",
     "ya'll": "you all",
     "yall": "you all",
     "dat": "that",
     "dis": "this",
     "dere": "there",
     "doin": "doing",
     "nuthin": "nothing",
     "alotta": "a lot of",
     "kewl": "cool",
     "omfg": "oh my fucking god",
     "lmao": "laughing my ass off",
     "ily": "i love you",
     "nite": "night",
     "nitey": "nighty",
     "bday": "birthday",
     "l8r": "later",
     "2day": "today",
     "2moro": "tomorrow",
     "4eva": "forever",
     "hun": "honey",
     "dunno": "dont know",
     "nt": "not",
     }

HTML_DICT = {
     "&amp;": "&",
     "&lt;": "<",
     "&gt;": ">",
     "&quot;": "\"",
     "&apos;": "'",
     "&nbsp;": " ",
     "&copy;": "©",
     "&reg;": "®",
     "&trade;": "™",
     "&mdash;": "—",
     "&ndash;": "–",
     "&hellip;": "…",
     "&iexcl;": "¡",
     "&iquest;": "¿",
     "&deg;": "°"
    }

# regex for detecting any type of url
URL_RE = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
# url token
URL_TOKEN = "[URL]"


class Normalizer:
    def __init__(self,
                 remove_needless_ws: bool = True,
                 remove_needless_ws_pattern=None,
                 lower_case: bool = True,
                 remove_needless_punctuation: bool = True,
                 remove_needless_punctuation_pattern=None,
                 remove_accents: bool = False,
                 unicode_normalization: Optional[Literal['NFC', 'NFD', 'NFKC', 'NFKD']] = None,
                 expand_contractions: bool = False,
                 replace_urls: bool = False,
                 replace_html_tags: bool = False
                 ):
        """
        Args:
            remove_needless_ws: Weather to replace all whitespaces occurrences with given pattern.
            remove_needless_ws_pattern: pattern for removing whitespaces. Replace all whitespace characters occurrences with single whitespace by default.
            lower_case: Whether to convert text to lowercase.
            remove_needless_punctuation = Whether to replace all punctuation with given pattern.
            remove_needless_punctuation_pattern = pattern for removing punctuation. Replace all consecutive occurrences of same punctuation with single occurrence by default.
            remove_accents: Whether to strip diacritical marks.
            unicode_normalization: Unicode normalization form to apply ('NFC', 'NFD', 'NFKC', 'NFKD').
            expand_contractions: whether to expend common contractions.
            replace_urls: whether to replace urls with special token <URL>
            replace_html_tags: whether to replace html tags with their view form (e.g "&lt;" -> "<")
        """
        self.remove_needless_ws = remove_needless_ws
        self.remove_needless_ws_pattern = remove_needless_ws_pattern \
            if remove_needless_ws_pattern is not None \
            else REMOVE_WHITESPACES_PATTERN
        self.lower_case = lower_case
        self.remove_needless_punctuation = remove_needless_punctuation
        self.remove_needless_punctuation_pattern = remove_needless_punctuation_pattern \
            if remove_needless_punctuation_pattern is not None \
            else REMOVE_PUNCTUATION_PATTERN
        self.remove_accents = remove_accents
        self.unicode_normalization = unicode_normalization
        self.expand_contractions = expand_contractions
        self.replace_urls = replace_urls
        self.replace_html_tags = replace_html_tags

    def normalize_text(self, text: str) -> list:
        if self.replace_html_tags:
            text = _replace_html_tags(text=text)
        if self.replace_urls:
            text = _replace_urls(text=text)
        if self.unicode_normalization:
            text = _unicode_normalize(
                text=text,
                unicode_norm=self.unicode_normalization,
                remove_accents=self.remove_accents
            )
        if self.expand_contractions:
            text = _contractions_expender(text=text)
        if self.lower_case:
            text = _lower_case_text(text=text)
        if self.remove_needless_punctuation:
            text = self._remove_needless_punctuations(text=text)
        if self.remove_needless_ws:
            text = self._remove_needless_whitespace(text=text)
        return text

    def _remove_needless_whitespace(self, text: str) -> str:
        """
        Replace all whitespace characters occurrences with given pattern
        Args:
            text: String of text
        Returns: 'text' string with only one whitespace between words
        """
        return re.sub(*self.remove_needless_ws_pattern, text)

    def _remove_needless_punctuations(self, text: str) -> str:
        """
        Replace all punctuations characters occurrences with given pattern
        Args:
            text: string of text
        Returns: 'text' string with only one whitespace between words
        """
        return re.sub(*self.remove_needless_punctuation_pattern, text)


def _lower_case_text(text: str) -> str:
    """
    Lower case the characters
    Args:
        text: String of text
    Returns: Lower cased 'text' string
    """
    # Escape tokens
    url_escaped_token = re.escape(URL_TOKEN)

    # Replace [URL] temporarily with a placeholder
    url_placeholder = "[url]"
    text = re.sub(url_escaped_token, url_placeholder, text)

    # Lowercase everything else
    text = text.lower()
    # Restore the original [URL] token
    return text.replace(url_placeholder, URL_TOKEN)


def _unicode_normalize(text: str,
                       unicode_norm: Optional[Literal['NFC', 'NFD', 'NFKC', 'NFKD']] = 'NFKD',
                       remove_accents: bool = False) -> str:
    """
    Normalize the text based on given unicode normalization such as  'NFC', 'NFD', 'NFKC', 'NFKD'
    Remove accents/diacritics if stated.
    Args:
        text: String of text
        unicode_norm: One of 'NFC', 'NFD', 'NFKC', 'NFKD'
        remove_accents: Whether to remove accents
    Returns: Normalized text

    """
    text_splited = text.split()
    text = []
    for word in text_splited:
        normalized_chrs = [c for c in unicodedata.normalize(unicode_norm, word)]
        if remove_accents:
            word = ""
            for c in normalized_chrs:
                unicode_category = unicodedata.category(c)
                if not unicode_category.startswith('M'):  # Skip accents/diacritics:
                    word += c
        else:
            word = "".join(normalized_chrs)
        if len(word) > 0:
            text.append(word)
    return " ".join(text)


def _contractions_expender(text: str) -> str:
    """
    Expand common english contractions
    Args:
        text: String of text
    Returns: The text string with any contractions replaced with their longer form
    """

    def _replace_contraction(match):
        return CONTRACTIONS_DICT[match.group(0)]

    pattern = r'\b(?:' + '|'.join(map(re.escape, CONTRACTIONS_DICT.keys())) + r')\b'
    contractions_re = re.compile(pattern)

    return contractions_re.sub(_replace_contraction, text)


def _replace_urls(text: str) -> str:
    """
    Replace any occurrence of an url with url token <URL>
    Args:
        text: String of text
    Returns: The text string with urls replaced with <URL>

    """
    text = re.sub(URL_RE, URL_TOKEN, text)
    return text

def _replace_html_tags(text: str) -> str:
    """
    Replace all HTML tags with their browser viewed form (e.g '&lt;' -> '<')
    Args:
        text: String of text
    Returns: text with html tags replaced with their browser viewed form
    """
    def _replace_entity(match):
        return HTML_DICT[match.group(0)]

    # Escape the keys for safe regex matching
    pattern = '|'.join(map(re.escape, HTML_DICT.keys()))
    html_entity_re = re.compile(pattern)

    return html_entity_re.sub(_replace_entity, text)


text = "Mrs.CJBaran&lt;3 OMG???!!!! what the hell he's very intereting...... Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 ﬃ é ain't The   very name strikes Dr. Smith fear and awe into the hearts of programmers worldwide.   We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception. http://bit.ly/Q3o2N  just listen to it Demi is amaizing"
text = "good things happen to those who wait. - soï¿½iï¿½m waiting. good things coming up aheadï¿½bring it.  iï¿½m excited!... http://tumblr.com/xlt1ub13n"
print(text)
normalizer = Normalizer(unicode_normalization='NFKD',
                        remove_accents=True,
                        expand_contractions=True,
                        replace_urls=True,
                        replace_html_tags=True)
print("---")
text = normalizer.normalize_text(text)
print(text)
print("---")
print(text.split())
