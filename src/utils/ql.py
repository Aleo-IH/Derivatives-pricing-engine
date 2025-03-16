import QuantLib as ql


def preprocess_quotes(params: dict = {}):
    return {key: ql.SimpleQuote(v) for key, v in params.items()}
