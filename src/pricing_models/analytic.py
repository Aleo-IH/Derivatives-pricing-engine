import QuantLib as ql
from ..utils.ql import preprocess_quotes


def BSM_EuroVanilla(u_params: dict = None, o_params: dict = None):
    q_params = preprocess_quotes(params=u_params)

    calendar = ql.TARGET()
    day_count = ql.Actual365Fixed()

    div_term = ql.FlatForward(0, calendar, ql.QuoteHandle(q_params["div"]), day_count)
    rf_term = ql.FlatForward(0, calendar, ql.QuoteHandle(q_params["r"]), day_count)
    vol_term = ql.BlackConstantVol(
        0, calendar, ql.QuoteHandle(q_params["sigma"]), day_count
    )

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(q_params["u"]),
        ql.YieldTermStructureHandle(div_term),
        ql.YieldTermStructureHandle(rf_term),
        ql.BlackVolTermStructureHandle(vol_term),
    )

    option = ql.EuropeanOption(
        ql.PlainVanillaPayoff(ql.Option.Call, o_params["k"]),
        ql.EuropeanExercise(o_params["exercise_date"]),
    )

    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option
