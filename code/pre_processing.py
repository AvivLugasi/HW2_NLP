import unicodedata
from typing import Literal, Optional
import re
from patterns_and_dicts import *


class Normalizer:
    def __init__(self,
                 remove_needless_ws: bool = True,
                 remove_needless_ws_pattern=None,
                 lower_case: Optional[Literal[
                     'ALL', 'STOP WORDS ONLY', 'TITLE CASE + STOP WORDS', 'TITLE CASE ONLY', 'LEAVE ACRONYM']] = None,
                 remove_needless_punctuation: bool = True,
                 remove_needless_punctuation_pattern=None,
                 remove_accents: bool = False,
                 unicode_normalization: Optional[Literal['NFC', 'NFD', 'NFKC', 'NFKD']] = None,
                 expand_contractions: bool = False,
                 replace_urls: bool = False,
                 replace_usernames: bool = False,
                 replace_html_tags: bool = False,
                 replace_hashtag: bool = False
                 ):
        """
        Args: remove_needless_ws: Weather to replace all whitespaces occurrences with given pattern.
        remove_needless_ws_pattern: pattern for removing whitespaces. Replace all whitespace characters occurrences
        with single whitespace by default. lower_case: Mode for convert text to lowercase, if None then nothing is
        converted. remove_needless_punctuation = Whether to replace all punctuation with given pattern.
        remove_needless_punctuation_pattern = pattern for removing punctuation. Replace all consecutive occurrences
        of same punctuation with single occurrence by default. remove_accents: Whether to strip diacritical marks.
        Works only if unicode_normalization=True unicode_normalization: Unicode normalization form to apply ('NFC',
        'NFD', 'NFKC', 'NFKD'). expand_contractions: whether to expend common contractions. replace_urls: whether to
        replace urls with special token [URL] replace_usernames: whether to replace usernames with spacial token [
        USER] replace_html_tags: whether to replace html tags with their view form (e.g "&lt;" -> "<")
        replace_hashtag: whether to replace hashtag tags with special token [Hashtag] and seperate it from the
        following entitiy
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
        self.replace_usernames = replace_usernames
        self.replace_html_tags = replace_html_tags
        self.replace_hashtag = replace_hashtag

    def normalize_text(self, text: str) -> list:
        if self.replace_html_tags:
            text = _replace_html_tags(text=text)
        if self.replace_urls:
            text = _replace_urls(text=text)
        if self.replace_usernames:
            text = _replace_usernames(text=text)
        if self.replace_hashtag:
            text = _replace_hashtags(text=text)
        if self.unicode_normalization:
            text = _unicode_normalize(
                text=text,
                unicode_norm=self.unicode_normalization,
                remove_accents=self.remove_accents
            )
        if self.lower_case:
            text = _lower_case_text(text=text, mode=self.lower_case)
        if self.expand_contractions:
            text = _contractions_expender(text=text)
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


def _lower_case_text(text: str, mode: str) -> str:
    """
    Lower case the characters
    Args:
        text: String of text
        mode: mode for lower casing.
            values: ['ALL', 'STOP WORDS ONLY', 'TITLE CASE + STOP WORDS', 'TITLE CASE ONLY', 'LEAVE ACRONYM']
                ALL - lowercase all words.
                STOP WORDS ONLY - lower case only stop words from defined list.(e.g The -> the)
                TITLE CASE ONLY - title lower case.(e.g hoUse -> House).
                TITLE CASE + STOP WORDS - stop words and title lower case.
                LEAVE ACRONYM - lowercasing everything beside words that are all capital(e.g NATO).
    Returns: Lower cased 'text' string
    """
    def to_title_case(match):
        word = match.group(0)
        return word.capitalize()  # First letter uppercase, rest lowercase

    # Escape tokens
    url_escaped_token = re.escape(URL_TOKEN)
    usr_escaped_token = re.escape(USER_TOKEN)
    hashtag_escaped_token = re.escape(HASHTAG_TOKEN)

    # Replace special tokens temporarily with a placeholder
    url_placeholder = URL_TOKEN.lower()
    usr_placeholder = USER_TOKEN.lower()
    hashtag_placeholder = HASHTAG_TOKEN.lower()
    text = re.sub(url_escaped_token, url_placeholder, text)
    text = re.sub(usr_escaped_token, usr_placeholder, text)
    text = re.sub(hashtag_escaped_token, hashtag_placeholder, text)

    # Lowercase text
    if mode == "ALL":
        pattern = r'[A-Z]'
    elif mode == "STOP WORDS ONLY":
        pattern = r'\b(?:' + '|'.join(
            f"{re.escape(word.capitalize())}|{re.escape(word.upper())}" for word in STOP_WORDS
        ) + r')\b'
    elif mode == "TITLE CASE ONLY":
        pattern = r'\b(?![A-Z][a-z]+\b)(?=\w*[a-z])(?=\w*[A-Z])\w+\b'
        text = re.sub(pattern, to_title_case, text)
    elif mode == "TITLE CASE + STOP WORDS":
        pattern = r'\b(?![A-Z][a-z]+\b)(?=\w*[a-z])(?=\w*[A-Z])\w+\b'
        text = re.sub(pattern, to_title_case, text)
        pattern = r'\b(?:' + '|'.join(
            f"{re.escape(word.capitalize())}|{re.escape(word.upper())}" for word in STOP_WORDS
        ) + r')\b'
    elif mode == "LEAVE ACRONYM":
        pattern = r'\b(?! (?:[A-Z0-9]\.?){2,}\b)\w+\b'
    text = re.sub(pattern, lambda m: m.group(0).lower(), text)

    # Restore the original special tokens
    text = text.replace(url_placeholder, URL_TOKEN)
    text = text.replace(usr_placeholder, USER_TOKEN)
    text = text.replace(hashtag_placeholder, HASHTAG_TOKEN)
    return text


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
        original = match.group(0)
        lower = original.lower()
        if lower in CONTRACTIONS_DICT:
            expanded = CONTRACTIONS_DICT[lower]
            # Preserve casing
            if original.isupper():
                return expanded.upper()
            elif original[0].isupper():
                return expanded.capitalize()
            else:
                return expanded
        return original  # fallback in case no match

    pattern = r'\b(?:' + '|'.join(map(re.escape, CONTRACTIONS_DICT.keys())) + r')\b'
    contractions_re = re.compile(pattern, flags=re.IGNORECASE)

    return contractions_re.sub(_replace_contraction, text)


def _replace_urls(text: str) -> str:
    """
    Replace any occurrence of an url with url token [URL]
    Args:
        text: String of text
    Returns: The text string with urls replaced with [URL]

    """
    return re.sub(URL_RE, URL_TOKEN, text)


def _replace_usernames(text: str) -> str:
    """
    Replace any occurrence of a username with username token [USER]
    Args:
        text: String of text
    Returns: The text string with urls replaced with [USER]
    """
    return re.sub(USER_RE, USER_TOKEN, text)


def _replace_html_tags(text: str) -> str:
    """
    Replace all HTML tags with their browser viewed form (e.g '&lt;' -> '<')
    Args:
        text: String of text
    Returns: text with html tags replaced with their browser viewed form
    """

    def _replace_html_tag(match):
        return HTML_DICT[match.group(0)]

    pattern = '|'.join(map(re.escape, HTML_DICT.keys()))
    html_entity_re = re.compile(pattern)

    return html_entity_re.sub(_replace_html_tag, text)


def _replace_hashtags(text: str) -> str:
    """
    Replace all # tags with special token [HASHTAG] split the following entity from it.
    e.g: #AllStars -> [HASHTAG] AllStars
    Args:
        text: String of text
    Returns: text with # tags replaced with '[HASHTAG] '.
    """
    def _hashtag_replacer(match):
        return HASHTAG_TOKEN + " " + match.group(0)[1:]

    hashtag_pattern = re.compile(HASHTAG_RE)
    return hashtag_pattern.sub(_hashtag_replacer, text)


text_file = []
with open("../data/domain_1_dev.txt", 'r', encoding='utf-8') as f:
    text_file = f.readlines()

# text = "Mrs.CJBaran&lt;3 OMG???!!!! what the hell he's very intereting...... Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 ﬃ é ain't The   very name strikes Dr. Smith fear and awe into the hearts of programmers worldwide.   We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception. http://bit.ly/Q3o2N  just listen to it Demi is amaizing"
# text = "good things happen to those who wait. - soï¿½iï¿½m waiting. good things coming up aheadï¿½bring it.  iï¿½m excited!... http://tumblr.com/xlt1ub13n"
# text = "@A_True_Diamond.  comeeee already. Sheesh. I Might go to l a in 2 weeks but idk. Its a guy situation thing. Kinda dnt wanna be obligated"
# text = "@igrigorik Oh, nice, #London it's a profile of you  Nice job on getting in the National Post  http://tinyurl.com/czewbw"
# text = "@danaschurer me toooo. so jealous of @nicoledegroot !! miss your face. hope you have a fun night. i'm off to get druuuuunk"
normalizer = Normalizer(unicode_normalization='NFKD',
                        lower_case="TITLE CASE + STOP WORDS",
                        remove_accents=True,
                        expand_contractions=True,
                        replace_urls=True,
                        replace_usernames=True,
                        replace_hashtag=True,
                        replace_html_tags=True)
# print(normalizer.normalize_text(text))
for text in text_file:
    print("--- raw text")
    print(text)
    text = normalizer.normalize_text(text)
    print("--- post normalization")
    print(text)
    print("---")
    print(text.split())
