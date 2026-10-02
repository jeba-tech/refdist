"""
T1-T6. Each has one declared direction, and none generates text: every edit
swaps one human form for another from a fixed, auditable rule list.

  T1  expand contractions            don't -> do not
  T2  rarer -> commoner word        utilize -> use    (curated plain-English list)
  T3  split at comma + conjunction   ", and it" -> ". And it"
  T4  insert additive discourse marker at sentence starts
  T5  American -> British spelling   color -> colour
  T6  insert optional complementiser "think he" -> "think that he"
"""

from __future__ import annotations

import re

from refdist.transforms.base import Site, Transform, match_case

_APOS = "['’]"


# ── T1 contractions ──────────────────────────────────────────────────────────

_NT = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not", "can't": "cannot",
    "won't": "will not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
    "mustn't": "must not", "needn't": "need not",
}
_PRON = {
    "i'm": "I am", "you're": "you are", "we're": "we are", "they're": "they are",
    "i've": "I have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "I will", "you'll": "you will", "we'll": "we will", "they'll": "they will",
    "he'll": "he will", "she'll": "she will", "it'll": "it will",
}
# 's is "is" unless a past participle follows ("it's been" = "it has been").
_S_HOSTS = ("it", "that", "there", "what", "he", "she", "here", "who")
_HAS_NEXT = {"been", "got", "gotten", "had", "done", "seen", "made", "taken", "become"}


class Contractions(Transform):
    key, name, direction = "T1", "contractions", "expand"
    _re = re.compile(rf"\b([A-Za-z]+{_APOS}(?:t|m|re|ve|ll|s))\b(\s+(\w+))?")

    def sites(self, text):
        out = []
        for m in self._re.finditer(text):
            tok = m.group(1)
            norm = re.sub(_APOS, "'", tok).lower()
            nxt = (m.group(3) or "").lower()
            if norm in _NT:
                rep = _NT[norm]
            elif norm in _PRON:
                rep = _PRON[norm]
            elif norm.endswith("'s") and norm[:-2] in _S_HOSTS:
                rep = f"{norm[:-2]} {'has' if nxt in _HAS_NEXT else 'is'}"
            else:
                continue
            out.append(Site(m.start(1), m.end(1), tok, match_case(tok, rep)))
        return out


# ── T2 lexical frequency ─────────────────────────────────────────────────────

# Curated plain-English substitutions, rarer -> commoner. Automatic WordNet
# synonymy was tried and rejected: its sense ordering maps "category" to
# "family" and "sunny" to "gay", so perplexity changes would measure semantic
# damage rather than typicality. Every pair below is safe in nearly all
# contexts; forms that are also nouns or adjectives in a way that breaks the
# swap ("an attempt" -> "an try", "provided that") are deliberately omitted.
_PLAIN = {
    "utilize": "use", "utilizes": "uses", "utilized": "used", "utilizing": "using",
    "utilise": "use", "utilises": "uses", "utilised": "used", "utilising": "using",
    "commence": "start", "commences": "starts", "commenced": "started", "commencing": "starting",
    "assist": "help", "assisted": "helped", "assisting": "helping", "assistance": "help",
    "obtain": "get", "obtains": "gets", "obtained": "got", "obtaining": "getting",
    "acquire": "get", "acquired": "got", "acquiring": "getting",
    "require": "need", "requires": "needs",
    "sufficient": "enough", "adequate": "enough", "numerous": "many",
    "approximately": "about", "demonstrate": "show", "demonstrates": "shows",
    "demonstrated": "showed", "demonstrating": "showing", "individuals": "people",
    "attempted": "tried", "attempting": "trying", "inquire": "ask", "inquired": "asked",
    "enquire": "ask", "enquired": "asked", "residence": "home", "construct": "build",
    "constructed": "built", "constructing": "building", "possess": "have",
    "possesses": "has", "possessed": "had", "indicates": "shows", "regarding": "about",
    "subsequently": "later", "frequently": "often", "terminate": "end", "terminated": "ended",
    "facilitate": "help", "facilitates": "helps", "facilitated": "helped",
    "initiate": "start", "initiated": "started", "modify": "change", "modifies": "changes",
    "modified": "changed", "modifying": "changing", "reside": "live", "resides": "lives",
    "resided": "lived", "transmit": "send", "transmitted": "sent", "retain": "keep",
    "retained": "kept", "consequently": "so", "permitted": "allowed", "ascertain": "find out",
    "comprehend": "understand", "beverage": "drink", "beverages": "drinks",
    "automobile": "car", "automobiles": "cars", "physician": "doctor", "physicians": "doctors",
    "magnitude": "size", "optimal": "best", "additionally": "also", "whilst": "while",
    "amongst": "among", "upon": "on", "remainder": "rest", "component": "part",
    "components": "parts", "alteration": "change", "alterations": "changes",
    "inform": "tell", "requested": "asked", "provide": "give", "provides": "gives",
    "providing": "giving", "receive": "get", "receives": "gets", "received": "got",
    "inexpensive": "cheap", "presently": "now", "initially": "at first", "ensure": "make sure",
    "ensures": "makes sure", "prior to": "before", "in order to": "to",
    "a great deal of": "a lot of", "the majority of": "most of", "in the event that": "if",
    "due to the fact that": "because", "at this point in time": "now",
    "in close proximity to": "near",
}


def _check_direction() -> None:
    from wordfreq import zipf_frequency
    for rare, common in _PLAIN.items():
        if " " in rare or " " in common:
            continue
        if zipf_frequency(common, "en") <= zipf_frequency(rare, "en"):
            raise AssertionError(f"T2 pair {rare!r} -> {common!r} is not rarer -> commoner")


class LexicalFrequency(Transform):
    key, name, direction = "T2", "lexfreq", "rarer->commoner"
    _re = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(_PLAIN, key=len, reverse=True))
                     + r")\b", re.I)

    def __init__(self) -> None:
        _check_direction()

    def sites(self, text):
        return [Site(m.start(), m.end(), m.group(), match_case(m.group(), _PLAIN[m.group().lower()]))
                for m in self._re.finditer(text)]


# ── T3 sentence split ────────────────────────────────────────────────────────

class SentenceSplit(Transform):
    key, name, direction = "T3", "sentence_split", "split"
    _re = re.compile(r", (and|but|so|yet) (?=[a-z])")

    def sites(self, text):
        return [Site(m.start(), m.end(), m.group(), f". {m.group(1).capitalize()} ")
                for m in self._re.finditer(text)]


# ── T4 discourse markers ─────────────────────────────────────────────────────

_MARKERS = ["Moreover, ", "Furthermore, ", "In addition, ", "Additionally, ", "Also, "]
_HAS_MARKER = re.compile(
    r"^(?:moreover|furthermore|in addition|additionally|also|however|therefore|thus|"
    r"hence|besides|but|and|so|yet|or)\b", re.I)


class DiscourseMarkers(Transform):
    key, name, direction = "T4", "discourse", "insert"
    _start = re.compile(r"(?<=[.!?] )([A-Z][a-z]+)")

    def sites(self, text):
        out = []
        for i, m in enumerate(self._start.finditer(text)):
            w = m.group(1)
            if _HAS_MARKER.match(w) or w in {"I"}:
                continue
            first = w if w in {"I"} else w[0].lower() + w[1:]
            marker = _MARKERS[i % len(_MARKERS)]
            out.append(Site(m.start(1), m.end(1), w, marker + first))
        return out


# ── T5 orthography ───────────────────────────────────────────────────────────

_US_UK = {
    "color": "colour", "colors": "colours", "colored": "coloured", "favorite": "favourite",
    "favorites": "favourites", "behavior": "behaviour", "behaviors": "behaviours",
    "honor": "honour", "labor": "labour", "neighbor": "neighbour", "neighbors": "neighbours",
    "flavor": "flavour", "flavors": "flavours", "humor": "humour", "rumor": "rumour",
    "harbor": "harbour", "favor": "favour", "center": "centre", "centers": "centres",
    "theater": "theatre", "meter": "metre", "meters": "metres", "liter": "litre",
    "liters": "litres", "fiber": "fibre", "organize": "organise", "organized": "organised",
    "organization": "organisation", "organizations": "organisations", "realize": "realise",
    "realized": "realised", "recognize": "recognise", "recognized": "recognised",
    "apologize": "apologise", "emphasize": "emphasise", "criticize": "criticise",
    "summarize": "summarise", "prioritize": "prioritise", "minimize": "minimise",
    "maximize": "maximise", "optimize": "optimise", "optimized": "optimised",
    "customize": "customise", "standardize": "standardise", "analyze": "analyse",
    "analyzed": "analysed", "paralyze": "paralyse", "defense": "defence", "offense": "offence",
    "license": "licence", "traveled": "travelled", "traveling": "travelling",
    "traveler": "traveller", "travelers": "travellers", "canceled": "cancelled",
    "canceling": "cancelling", "modeling": "modelling", "labeled": "labelled",
    "fueled": "fuelled", "catalog": "catalogue", "dialog": "dialogue", "analog": "analogue",
    "gray": "grey", "program": "programme", "programs": "programmes", "check": "cheque",
    "jewelry": "jewellery", "mold": "mould", "plow": "plough", "tire": "tyre", "tires": "tyres",
    "aluminum": "aluminium", "judgment": "judgement", "acknowledgment": "acknowledgement",
    "skeptical": "sceptical", "pajamas": "pyjamas", "cozy": "cosy", "donut": "doughnut",
    "percent": "per cent", "math": "maths", "fulfill": "fulfil", "enroll": "enrol",
    "installment": "instalment", "practicing": "practising", "specialty": "speciality",
}
# Ambiguous in context ("check" a box, "program" in computing, "tire" of) -- excluded.
for _amb in ("check", "program", "programs", "tire", "tires", "mold", "license", "percent", "math"):
    _US_UK.pop(_amb)


_SUFFIXES = (("izations", "isations"), ("ization", "isation"), ("izing", "ising"), ("ized", "ised"),
             ("izes", "ises"), ("ize", "ise"), ("yzing", "ysing"), ("yzed", "ysed"), ("yze", "yse"))
_NOT_IZE = {"size", "seize", "prize", "capsize", "baize", "maize", "sizes", "sized", "sizing",
            "prizes", "prized", "seized", "seizes", "seizing"}


def _ize_pairs() -> dict[str, str]:
    """-ize -> -ise for the 80k most frequent words, kept only where the British
    form is itself attested (zipf >= 1.5). Deterministic given the pinned wordfreq."""
    from wordfreq import top_n_list, zipf_frequency
    out = {}
    for w in top_n_list("en", 80000):
        if w in _NOT_IZE or not w.isalpha():
            continue
        for us, uk in _SUFFIXES:
            if w.endswith(us) and len(w) > len(us) + 2:
                cand = w[: -len(us)] + uk
                if zipf_frequency(cand, "en") >= 1.5:
                    out[w] = cand
                break
    return out


class Orthography(Transform):
    key, name, direction = "T5", "orthography", "US->UK"

    def __init__(self) -> None:
        self.map = {**_ize_pairs(), **_US_UK}  # hand-curated entries win
        words = sorted(self.map, key=len, reverse=True)
        self._re = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)

    def sites(self, text):
        return [Site(m.start(), m.end(), m.group(), match_case(m.group(), self.map[m.group().lower()]))
                for m in self._re.finditer(text)]


# ── T6 optional complementiser ───────────────────────────────────────────────

_VERBS = ("think|thought|believe|believed|say|said|says|know|knew|suppose|guess|hope|feel|"
          "felt|realize|realized|mean|meant|assume|notice|noticed|claim|claimed|found|find|"
          "suggest|suggests|argue|argued|agree|expect|admit|doubt|imagine|understand|see|saw")
# A subject pronoun right after the verb marks a clause, so "that" is optional
# there. Determiners are excluded: "I know the answer" is an object, not a clause.
_SUBJ = "I|you|he|she|it|we|they|there|someone|people"


class FunctionWords(Transform):
    # Insertion, not deletion: forum writers usually omit "that" already, so the
    # deletion direction found sites in only 11% of S6 documents.
    key, name, direction = "T6", "function_words", "insert that"
    _re = re.compile(r"\b(?:" + _VERBS + r") (?=(?:" + _SUBJ + r")\b)", re.I)

    def sites(self, text):
        return [Site(m.end(), m.end(), "", "that ") for m in self._re.finditer(text)]


TRANSFORMS: dict[str, Transform] = {
    t.key: t for t in (Contractions(), LexicalFrequency(), SentenceSplit(),
                       DiscourseMarkers(), Orthography(), FunctionWords())
}
