import re
import spacy
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.tokenizer import Tokenizer
import inspect
from emoji import EMOJI_DATA

import progressbar


def get_progressbar(N, name=""):
    return progressbar.ProgressBar(
        maxval=N,
        widgets=[progressbar.Bar('#', '[', ']'),
                 name,
                 progressbar.Percentage()])


class CustomToken:
    def __init__(self, text, lex=None, is_stop=False, is_sy=False) -> None:
        self.text = text
        self.is_stop = is_stop
        # self.is_date = False
        self.is_symbol = is_sy
        self.lemma = lex

    @staticmethod
    def cluster_list(list):
        clusters, token = {}, CustomToken('')
        for name, method in inspect.getmembers(token, predicate=inspect.ismethod):
            if name.startswith('_') or name == 'cluster_list':
                continue

            clusters[name] = []
            for word in list:
                token.text = word.text
                if method():
                    clusters[name].append(word)

        return clusters

    def unknown(self):
        result = True
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('_') or name == 'cluster_list' or 'unknown' == name:
                continue

            result = result and not method()
        return result

    def space(self):
        return re.fullmatch(r'\s+', self.text) != None

    def is_emoji(self):
        return self.text in EMOJI_DATA

    def is_digit(self):
        return re.fullmatch(r'[0-9\%ª]+', self.text) != None

    def is_hashtag(self):
        return re.fullmatch(r'\#.*', self.text) != None

    def is_user_tag(self):
        return re.fullmatch('@.*', self.text) != None

    def is_url(self):
        return re.fullmatch(r'''^https?://.*''', self.text) != None

    def is_date(self):
        return re.fullmatch(r'[(0-9)+\/(pm)(am)\s\-]+', self.text) != None and not self.space() and not self.is_digit()

    def natural_word(self):
        return re.fullmatch(r'[a-zA-ZáéíóúñÁÉÍÓÚüUÜÑ]+', self.text) != None

    def combined_word(self):
        return re.fullmatch(r'[a-zA-ZáéíóúñÁÉÍÓÚüUÜÑ\-\_]+', self.text) != None and not self.natural_word()

    def contract_word(self):
        return re.fullmatch(r'[a-zA-ZáéíóúñÁÉÍÓÚüUÜÑ\-\_\'’`]+', self.text) != None and not self.combined_word() and not self.natural_word()

    def numeral_word(self):
        return (
            re.fullmatch(r'[a-zA-ZáéíóúñÁÉÍÓÚüUÜÑ\-\_\_0-9\@]+', self.text) != None and
            not self.combined_word() and
            not self.natural_word() and
            not self.is_user_tag() and
            not self.is_digit()
        )

    def __eq__(self, __o: object) -> bool:
        return __o.text == self.text

    def __hash__(self) -> int:
        return hash(self.text)


class SpacyCustomTokenizer:
    basic_regex = [r'[\(\)\[\]\{\},;\!\?\+\*\"¬¨\.¿¡:“”|\$\/=]+', r'\s+']
    sp_fix_regex = [r'[\'\-\_\\/″]+']
    prefix_regex = []
    delete_prefix_regex = ['#']
    suffix_regex = []
    delete_suffix_regex = []
    infix_regex = []
    delete_infix_regex = []

    def __init__(self, special_cases={}) -> None:
        self.nlp = spacy.load("es_core_news_sm")

        emoji = [str(key) for key in EMOJI_DATA.keys()]
        emoji = ''.join(emoji)
        emoji = f'[{emoji}]+'
        self.sp_fix_regex.append(emoji)

        prefixes = list(self.nlp.Defaults.prefixes)
        for reg in self.delete_prefix_regex:
            prefixes.remove(reg)

        self.prefix_regex = compile_prefix_regex(
            prefixes + self.prefix_regex + self.basic_regex + self.sp_fix_regex)

        suffixes = list(self.nlp.Defaults.suffixes)
        for reg in self.delete_suffix_regex:
            suffixes.remove(reg)

        self.suffix_regex = compile_suffix_regex(
            suffixes + self.suffix_regex + self.basic_regex + self.sp_fix_regex)

        infixes = list(self.nlp.Defaults.infixes)
        for reg in self.delete_infix_regex:
            infixes.remove(reg)

        self.infix_regex = compile_infix_regex(
            infixes + self.infix_regex + self.basic_regex)

        simple_url_re = re.compile(r'''^https?://.*''')

        self.nlp.tokenizer = Tokenizer(self.nlp.vocab, rules=special_cases,
                                       prefix_search=self.prefix_regex.search,
                                       suffix_search=self.suffix_regex.search,
                                       infix_finditer=self.infix_regex.finditer,
                                       url_match=simple_url_re.match
                                       )

    def __ents__(self, text):
        return self.nlp(text).ents

    def __call__(self, text):
        for token in self.nlp(text):
            for t in self.__check_token__(token.text, CustomToken(token.text, is_stop=token.is_stop, is_sy=token.is_punct, lex=token.lemma_)):
                yield t

    def __check_token__(self, text, h_token=None):
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('_'):
                continue
            tokens = method(text)
            if any(tokens):
                for t in tokens:
                    yield t
                break
        else:
            yield h_token if not h_token is None else CustomToken(text)

    def prefix_re_check(self, text):
        m = self.prefix_regex.search(text)
        if m is None:
            return []
        if m.start() == 0 and m.end != len(text):
            token = CustomToken(text[0: m.end()], None, True, True)
            return [token] + list(self.__check_token__(text[m.end():]))
        return []

    def suffix_re_check(self, text):
        m = self.suffix_regex.search(text)
        if m is None:
            return []

        if m.start() != 0 and m.end == len(text):
            token = CustomToken(text[m.start():], None, True, True)
            return list(self.__check_token__(text[0: m.start():])) + [[token]]
        return []

    def emoji_check(self, text):
        if text in EMOJI_DATA:
            return []
        result = []
        previous_text = ''
        for c in text:
            if c in EMOJI_DATA:
                if previous_text != '':
                    result += list(self.__check_token__(previous_text,
                                   CustomToken(previous_text)))
                    previous_text = ''
                token = CustomToken(c)
                result.append(token)
            else:
                previous_text += c
        if previous_text == text:
            return []
        return result

    def space_check(self, text):
        l = text.split(' ')
        if len(l) > 1:
            return [self.__check_token__(t) for t in l]

        return []
