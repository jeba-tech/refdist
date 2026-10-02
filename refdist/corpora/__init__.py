from refdist.corpora import arxiv, gutenberg, learner, social, stackexchange

ADAPTERS = {
    "gutenberg": gutenberg.candidates,
    "arxiv": arxiv.candidates,
    "stackexchange": stackexchange.candidates,
    "pelic": learner.pelic,
    "icnale": learner.icnale,
    "asap": learner.asap,
    "twitteraae": social.twitteraae,
    "legal": social.legal,
    "spoken": social.spoken,
}
