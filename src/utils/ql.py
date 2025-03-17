import QuantLib as ql
from pandas import Timestamp


def preprocess_quotes(params: dict = {}):
    result = {}
    for key, val in params.items():
        
        if isinstance(val, (float, int)) and key != 'k':
            result[key] = ql.SimpleQuote(val)

        elif isinstance(val, Timestamp):
            result[key] = ql.Date(val.day, val.month, val.year)

        elif key == 'option_type':
            if val == 'CALL':
                result[key] = ql.Option.Call
            else:
                result[key] = ql.Option.Put

        else:
            result[key] = val
    
    return result



        
