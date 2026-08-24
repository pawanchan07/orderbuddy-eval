"""
The classification prompt, shared by the cost estimator and the bake-off.

Defined once so that the tokens counted during budgeting are the same tokens
sent during the run. Every tier receives a byte-identical system prompt and
user message; the only thing that varies between tiers is the model string and
the effort setting (Haiku 4.5 does not accept `effort`).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The taxonomy as derived in step 2 and confirmed against ground truth. The
# model is given the label set explicitly — this is a classification task
# against a fixed taxonomy, not open-ended labelling.
INTENTS = [
    "order_status",
    "delivery_delay",
    "cancel_order",
    "return_request",
    "refund_status",
    "exchange_item",
    "change_shipping_address",
    "change_payment_method",
    "damaged_item",
    "wrong_item_received",
    "missing_item",
    "invoice_receipt_request",
    "discount_promo_code",
    "account_login_issue",
    "product_availability",
    "subscription_manage",
]

INTENT_GLOSS = {
    "order_status": "asking where an order is or what stage it is at",
    "delivery_delay": "the order is late or the delivery date has slipped",
    "cancel_order": "wants to cancel an order that has not been received",
    "return_request": "wants to send a received item back",
    "refund_status": "chasing money owed for an already-agreed return or cancellation",
    "exchange_item": "wants to swap an item for a different size, colour or model",
    "change_shipping_address": "wants the delivery address changed or redirected",
    "change_payment_method": "wants to change the card or payment method on an order",
    "damaged_item": "the item arrived broken or damaged",
    "wrong_item_received": "the item delivered is not the item ordered",
    "missing_item": "part of the order did not arrive",
    "invoice_receipt_request": "wants a receipt, invoice or VAT document",
    "discount_promo_code": "a promo code, voucher or discount did not apply",
    "account_login_issue": "cannot log in, reset a password, or access the account",
    "product_availability": "asking about stock, restocking, or availability",
    "subscription_manage": "wants to pause, cancel, skip or change a recurring order",
}

SYSTEM_PROMPT = """You classify OrderBuddy customer support messages into exactly one intent.

TAXONOMY — choose exactly one label from this list:
{taxonomy}

RULES
1. Return the single best-fitting label. If the message genuinely fits none of \
them, or is too ambiguous to choose between two, return "abstain".
2. Quote your evidence verbatim from the customer message. The evidence must be \
a contiguous span copied exactly from the message, not a paraphrase and not a \
restatement of the label.
3. State nothing about the order that the message does not say. Do not infer \
order numbers, dates, items, prices or delivery status.
4. Do not repeat order references, email addresses, phone numbers or postal \
addresses in your output.{rule4a}
5. Do not promise any outcome — no refunds, replacements, cancellations, \
delivery dates or compensation. You are labelling the message, not answering it.

Distinctions that matter:
- cancel_order is about stopping an order not yet received; return_request is \
about sending back something already received.
- refund_status is about money already owed; return_request is about starting \
the return itself.
- exchange_item is a swap; return_request is a send-back for refund.
- delivery_delay is about lateness; order_status is a neutral "where is it".

Respond with JSON only, no preamble:
{{"intent": "<label or abstain>", "evidence": "<verbatim span>", "confidence": "high|medium|low"}}"""


# Prompt versions.
#
# v1 states rules 2 and 4 as independent requirements: quote verbatim, and do
# not repeat identifiers. On a message where the identifier sits inside the
# most natural evidence span, those two rules compete, and v1 leaves the model
# to notice the tension unaided. The v1 run showed that is exactly where the
# cheaper tiers fail (48/80 and 49/80 A2 failures vs 5/80 for the premium
# tier).
#
# v2 adds one sentence naming the conflict and saying which rule wins. Nothing
# else changes, so the difference between the two runs is attributable to that
# sentence alone.
RULE_4A_V2 = (
    " If the most natural answer span contains an order reference or any other "
    "identifier, quote a different span that does not — never repeat an "
    "identifier, even inside a verbatim quote. Rule 4 outranks rule 2 when "
    "they conflict."
)

PROMPT_VERSIONS = {"v1": "", "v2": RULE_4A_V2}
DEFAULT_PROMPT_VERSION = "v2"


def build_system_prompt(version: str = DEFAULT_PROMPT_VERSION) -> str:
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version {version!r}")
    taxonomy = "\n".join(f"- {name}: {INTENT_GLOSS[name]}" for name in INTENTS)
    return SYSTEM_PROMPT.format(taxonomy=taxonomy, rule4a=PROMPT_VERSIONS[version])


def build_user_message(text: str) -> str:
    return f"Customer message:\n{text}"


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS + ["abstain"]},
        "evidence": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["intent", "evidence", "confidence"],
    "additionalProperties": False,
}


def load_prices() -> dict:
    return json.loads((ROOT / "config" / "prices.json").read_text(encoding="utf-8"))
