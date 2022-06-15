import re
import spacy
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.tokenizer import Tokenizer
import inspect
from emoji import EMOJI_DATA
import json
import progressbar
import ast


def get_progressbar(N, name=""):
    return progressbar.ProgressBar(
        maxval=N,
        widgets=[progressbar.Bar('#', '[', ']'),
                 name,
                 progressbar.Percentage()])


class CustomToken:
    def __init__(self, text, lex=None, is_stop=False,
                 is_sy=False, is_title=False, is_end=False,
                 pos='', tag='', vector=None, dep='', sent='') -> None:
        self.text = text
        self.is_stop = is_stop
        # self.is_date = False
        self.is_symbol = is_sy
        self.lemma = lex
        self.syntax = (is_title, pos, tag, dep, is_end)
        self.vector = None if vector is None else tuple(vector)
        self.sent = sent

    def clone(self):
        token = CustomToken(self.text)
        token.__dict__ = self.__dict__
        return token

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
        self.memory = {}
        self.embedding = {}
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

    def __save__(self, path='token_text.json'):
        with open(path, 'w+') as f:
            f.write(str(self.memory))
            f.close()

    def __load__(self, path='token_text.json'):
        try:
            with open(path, 'r') as f:
                text = f.read()
                if not any(text):
                    return
                self.memory = ast.literal_eval(text)
                f.close()
        except:
            pass

    def __ents__(self, text):
        return self.nlp(text).ents

    def __transform__(self, token):
        return CustomToken(token.text, is_stop=token.is_stop, is_sy=token.is_punct or token.is_left_punct,
                           lex=token.lemma_, is_title=token.is_title or token.is_sent_start, is_end=token.is_sent_end,
                           pos=token.pos_, tag=token.tag_, dep=token.dep_,
                           vector=token.vector, sent=token.sent.text)

    def __call__(self, text):
        hsh = str(hash(text))
        if hsh in self.memory:
            for obj in self.memory[hsh]:
                t = CustomToken('')
                t.text = obj["text"]
                t.is_stop = obj["is_stop"]
                t.is_symbol = obj["is_symbol"]
                t.lemma = obj["lemma"]
                t.syntax = obj["syntax"]
                # t.vector = obj["vector"]
                t.sent = obj["sent"]

                yield t
        else:
            self.memory[hsh] = []
            for token in self.nlp(text):
                for t in self.__check_token__(token.text, self.__transform__(token)):
                    t.sent = token.sent.text
                    self.memory[hsh].append({
                        "text": t.text,
                        "is_stop": t.is_stop,
                        "is_symbol": t.is_symbol,
                        "lemma": t.lemma,
                        "syntax": t.syntax,
                        # "vector": t.vector,
                        "sent": t.sent
                    })

                    self.embedding[t.text] = t.vector
                    yield t
            self.memory[hsh] = tuple(self.memory[hsh])

    def __check_token__(self, text, h_token):
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith('_'):
                continue
            tokens = method(text, h_token)
            if any(tokens):
                for t in tokens:
                    yield t
                break
        else:
            h_token.text = text
            yield h_token

    def prefix_re_check(self, text, h_token):
        m = self.prefix_regex.search(text)
        if m is None:
            return []
        if m.start() == 0 and m.end != len(text):
            token = self.nlp(text[0: m.end()])[0]
            return [self.__transform__(token)] + list(self.__check_token__(text[m.end():], h_token))
        return []

    def suffix_re_check(self, text, h_token):
        m = self.suffix_regex.search(text)
        if m is None:
            return []

        if m.start() != 0 and m.end == len(text):
            token = self.nlp(text[m.start():])[0]
            return list(self.__check_token__(text[0: m.start():], h_token)) + [self.__transform__(token)]
        return []

    def emoji_check(self, text, h_token):
        if text in EMOJI_DATA:
            return []
        result = []
        previous_text = ''
        for c in text:
            if c in EMOJI_DATA:
                if previous_text != '':
                    result += list(self.__check_token__(previous_text, h_token))
                    previous_text = ''
                token = self.__transform__(self.nlp(c)[0])
                result.append(token)
            else:
                previous_text += c
        if previous_text == text:
            return []
        return result

    def space_check(self, text, h_token):
        l = text.split(' ')
        if len(l) > 1:
            return [self.__check_token__(t, h_token) for t in l]

        return []
