import QuantLib as ql
from ..utils.ql import preprocess_quotes


def Tree_USVanilla(u_params, steps: int = 200):
    today = ql.Settings.instance().evaluationDate

    calendar = ql.TARGET()
    count = ql.Actual365Fixed()

    q_params = preprocess_quotes(u_params)

    div_term = ql.FlatForward(0, calendar, ql.QuoteHandle(q_params["div"]), count)
    r_term = ql.FlatForward(0, calendar, ql.QuoteHandle(q_params["r"]), count)
    vol_term = ql.BlackConstantVol(
        0, calendar, ql.QuoteHandle(q_params["sigma"]), count
    )

    bsm_process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(q_params["u"]),
        ql.YieldTermStructureHandle(div_term),
        ql.YieldTermStructureHandle(r_term),
        ql.BlackVolTermStructureHandle(vol_term),
    )

    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(q_params["option_type"], q_params["k"]),
        ql.AmericanExercise(today, q_params["exercise_date"]),
    )

    option.setPricingEngine(ql.BinomialVanillaEngine(bsm_process, "crr", steps=steps))
    return option, bsm_process
