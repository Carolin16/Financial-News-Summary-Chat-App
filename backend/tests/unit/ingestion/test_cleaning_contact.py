"""Contact-detail rules: emails and phone numbers, with financial-figure false positives."""

import pytest


@pytest.fixture(scope="module")
def phone(cleaning_config):
    return next(r for r in cleaning_config.noise_rules if r.name == "contact_phone").regex


@pytest.fixture(scope="module")
def email(cleaning_config):
    return next(r for r in cleaning_config.noise_rules if r.name == "contact_email").regex


@pytest.mark.parametrize(
    "text",
    ["800-767-3771 ext. 9339", "call 800-767-3771.", "(212) 555-0199", "+1-212-555-0199"],
)
def test_phone_numbers_match(phone, text):
    assert phone.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "1,024.05",
        "$614.7 billion",
        "2024-2034",
        "from 2024-2032 at a CAGR",
        "13.09%",
        "7-9% target yield",
        "Q4 2024",
        "$22.03",
        "301.63 million",
        "4,700 from 4,600",
        "Feb. 19, 2025",
        "02:34",
        "NT$2,132.3 billion",
    ],
)
def test_phone_pattern_never_matches_financial_figures(phone, text):
    assert not phone.search(text)


@pytest.mark.parametrize("text", ["dhowley@yahoofinance.com", "laura.bratton@yahooinc.com"])
def test_emails_match(email, text):
    assert email.fullmatch(text)


@pytest.mark.parametrize("text", ["AT&T", "S&P 500", "@ 5%", "Data Brokers Market@"])
def test_email_pattern_ignores_non_addresses(email, text):
    assert not email.search(text)
