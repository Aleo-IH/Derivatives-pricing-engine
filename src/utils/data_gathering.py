import yfinance as yf
import pandas as pd
import numpy as np
from typing import Union, List
from dateutil.relativedelta import relativedelta


def to_continuous(rate: float) -> float:
    """
    Converts an annual nominal rate to a continuously compounded rate.

    Args:
        rate (float): Nominal annual rate (expressed as a fraction, e.g., 0.015 for 1.5%)

    Returns:
        float: Continuously compounded rate, or None in case of an error.
    """
    try:
        return np.log(1 + rate)
    except Exception as e:
        print(f"Warning: Error converting rate {rate} to continuous: {e}")
        return None


def validate_option_type(option_type: str) -> str:
    """
    Validates the option type.

    Args:
        option_type (str): Option type to validate - 'call', 'put', or 'both'

    Returns:
        str: Validated option type in lowercase

    Raises:
        ValueError: If option_type is not valid.
    """
    valid_types = ["call", "put", "both"]
    option_type = option_type.lower()
    if option_type not in valid_types:
        raise ValueError(f"option_type must be one of {valid_types}")
    return option_type


def get_risk_free_rate() -> float:
    """
    Retrieves the current risk-free rate using the ticker "^IRX" and converts it
    to an annual continuously compounded rate.

    Returns:
        float: Continuously compounded risk-free rate, or None in case of an error.
    """
    try:
        rf_ticker = yf.Ticker("^IRX")
        # Divide by 100 to convert percentage to fraction
        risk_free_rate = rf_ticker.info.get("regularMarketPrice") / 100
        if risk_free_rate is not None:
            risk_free_rate = to_continuous(risk_free_rate)
    except Exception as e:
        print(f"Warning: Error retrieving risk free rate: {str(e)}")
        risk_free_rate = None
    return risk_free_rate


def get_current_price(stock: yf.Ticker) -> float:
    """
    Retrieves the current price of the underlying asset.

    Args:
        stock (yf.Ticker): Ticker object of the underlying asset.

    Returns:
        float: The last closing price or None in case of an error.
    """
    try:
        hist_data = stock.history(period="1d")
        if hist_data.empty:
            print(
                f"Warning: No price data found for {stock.info.get('symbol', 'unknown')}"
            )
            return None
        return hist_data["Close"].iloc[-1]
    except Exception as e:
        print(f"Warning: Error retrieving price data: {str(e)}")
        return None


def get_dividend_yield(stock: yf.Ticker) -> float:
    """
    Retrieves the dividend yield of the underlying asset and converts it
    to a continuously compounded yield.

    Args:
        stock (yf.Ticker): Ticker object of the underlying asset.

    Returns:
        float: Continuously compounded dividend yield, or None in case of an error.
    """
    try:
        info = stock.info
        # Divide by 100 to convert percentage to fraction
        dividend_yield = info.get("dividendYield")
        if dividend_yield is not None:
            dividend_yield = dividend_yield / 100
            dividend_yield = to_continuous(dividend_yield)
    except Exception as e:
        print(f"Warning: Error retrieving dividend yield: {str(e)}")
        dividend_yield = None
    return dividend_yield


def get_expiration_dates(
    stock: yf.Ticker,
    expiry_date: str = None,
    expiry_start_date: pd.Timestamp = None,
    expiry_end_date: pd.Timestamp = None,
) -> List[pd.Timestamp]:
    """
    Retrieves and filters expiration dates for a given ticker.

    Args:
        stock (yf.Ticker): Ticker object.
        expiry_date (str, optional): Specific expiration date.
        expiry_start_date (pd.Timestamp, optional): Start date for filtering.
        expiry_end_date (pd.Timestamp, optional): End date for filtering.

    Returns:
        List[str]: List of expiration dates.
    """
    if expiry_date:
        return [pd.Timestamp(expiry_date)]

    expirations = stock.options
    if expiry_start_date or expiry_end_date:
        expirations = [
            pd.Timestamp(exp)
            for exp in expirations
            if (
                (not expiry_start_date or pd.Timestamp(exp) >= expiry_start_date)
                and (not expiry_end_date or pd.Timestamp(exp) <= expiry_end_date)
            )
        ]
    return expirations


def process_option_data(
    opt_data: pd.DataFrame,
    symbol: str,
    exp_date: pd.Timestamp,
    current_date: pd.Timestamp,
    option_type: str,
    min_strike: float = None,
    max_strike: float = None,
    underlying_price: float = None,
    dividend_yield: float = None,
    risk_free_rate: float = None,
) -> pd.DataFrame:
    """
    Processes the options data by adding metadata and applying filters.

    Args:
        opt_data (pd.DataFrame): Raw options data.
        symbol (str): Ticker symbol.
        exp_date (str): Expiration date.
        current_date (pd.Timestamp): Date/time of data retrieval.
        option_type (str): Option type ('CALL' or 'PUT').
        min_strike (float, optional): Minimum strike filter.
        max_strike (float, optional): Maximum strike filter.
        underlying_price (float, optional): Underlying asset price.
        dividend_yield (float, optional): Continuously compounded dividend yield.
        risk_free_rate (float, optional): Continuously compounded risk-free rate.

    Returns:
        pd.DataFrame: Processed options data.
    """
    opt_data["Option Type"] = option_type
    opt_data["Ticker"] = symbol
    opt_data["expiration"] = exp_date
    opt_data["date"] = current_date
    opt_data["time_to_maturity"] = (exp_date - current_date).days / 365.25
    opt_data["underlying_price"] = underlying_price
    opt_data["dividend_yield"] = dividend_yield
    opt_data["risk_free_rate"] = risk_free_rate

    if min_strike is not None:
        opt_data = opt_data[opt_data["strike"] >= min_strike]
    if max_strike is not None:
        opt_data = opt_data[opt_data["strike"] <= max_strike]

    return opt_data


def options_gathering(
    ticker: Union[str, List[str]],
    expiry_date: str = None,
    expiry_start_date: str = None,
    expiry_end_date: str = None,
    min_strike: float = None,
    max_strike: float = None,
    option_type: str = "both",
) -> pd.DataFrame:
    """
    Retrieves the current options data for one or more tickers using the yfinance API,
    with filtering capabilities.

    Args:
        ticker (Union[str, List[str]]): Single ticker symbol or a list of symbols.
        expiry_date (str, optional): Specific expiration date ('YYYY-MM-DD').
        expiry_start_date (str, optional): Start date for filtering expiration dates ('YYYY-MM-DD').
        expiry_end_date (str, optional): End date for filtering expiration dates ('YYYY-MM-DD').
        min_strike (float, optional): Minimum strike to include.
        max_strike (float, optional): Maximum strike to include.
        option_type (str, optional): Option type to return - 'call', 'put', or 'both'.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered current options data.
    """
    option_type = validate_option_type(option_type)

    # Process expiration dates if provided
    if expiry_start_date:
        expiry_start_date = pd.to_datetime(expiry_start_date)
    if expiry_end_date:
        expiry_end_date = pd.to_datetime(expiry_end_date)

    # Retrieve the risk-free rate using the dedicated function
    risk_free_rate = get_risk_free_rate()

    tickers = [ticker] if isinstance(ticker, str) else ticker
    current_date = pd.Timestamp.now()
    all_options_data = []

    for symbol in tickers:
        symbol = symbol.strip()
        stock = yf.Ticker(symbol)
        expirations = get_expiration_dates(
            stock, expiry_date, expiry_start_date, expiry_end_date
        )

        # Retrieve the current underlying price using the dedicated function
        underlying_price = get_current_price(stock)

        # Retrieve the dividend yield using the dedicated function
        dividend_yield = get_dividend_yield(stock)

        for exp_date in expirations:
            try:
                opt_chain = stock.option_chain(exp_date.strftime("%Y-%m-%d"))
                if option_type in ["call", "both"]:
                    calls = process_option_data(
                        opt_chain.calls,
                        symbol,
                        exp_date,
                        current_date,
                        "CALL",
                        min_strike,
                        max_strike,
                        underlying_price,
                        dividend_yield,
                        risk_free_rate,
                    )
                    all_options_data.append(calls)

                if option_type in ["put", "both"]:
                    puts = process_option_data(
                        opt_chain.puts,
                        symbol,
                        exp_date,
                        current_date,
                        "PUT",
                        min_strike,
                        max_strike,
                        underlying_price,
                        dividend_yield,
                        risk_free_rate,
                    )
                    all_options_data.append(puts)

            except Exception as e:
                print(
                    f"Error gathering data for {symbol} expiring {exp_date}: {str(e)}"
                )
                continue

    if not all_options_data:
        return pd.DataFrame()

    all_options_data = pd.concat(all_options_data, axis=0, ignore_index=True)
    all_options_data["expiration"] = pd.to_datetime(all_options_data["expiration"])
    all_options_data["date"] = pd.to_datetime(all_options_data["date"])

    return all_options_data
