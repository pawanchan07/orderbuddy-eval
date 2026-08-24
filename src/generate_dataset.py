"""
Step 1 — Synthetic customer-support dataset (orders domain).

Generates a deterministic, seeded corpus of customer messages spanning a
latent intent taxonomy. The generator is compositional (opener + core ask +
detail + closer, with surface noise applied afterwards) so that messages
carry real lexical variety instead of a handful of repeated templates.

The `intent` column is ground truth. It exists so later stages can be
*scored*; the intent-discovery stage (step 2) never reads it.

Usage:
    py src/generate_dataset.py
Writes:
    data/raw/support_messages.csv   (10,000 rows)
    data/raw/dataset_manifest.json  (seed, counts, checksum)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260824
N_ROWS = 10_000
ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "raw" / "support_messages.csv"
OUT_MANIFEST = ROOT / "data" / "raw" / "dataset_manifest.json"

# --------------------------------------------------------------------------
# Surface vocabulary shared across intents
# --------------------------------------------------------------------------

OPENERS = [
    "", "", "", "",  # weight the empty opener — many messages start cold
    "Hi, ", "Hello, ", "Hi there, ", "Hey, ", "Good morning, ",
    "Hi team, ", "Hello support, ", "Hi OrderBuddy, ",
    "Sorry to bother you but ", "Quick question - ",
    "I need some help please. ", "Not sure who to ask about this but ",
    "This is the third time I'm writing in. ",
    "I've been waiting on hold so I'm trying here instead. ",
]

CLOSERS = [
    "", "", "", "", "",  # most messages just stop
    " Thanks.", " Thanks!", " Thank you.", " Thanks in advance.",
    " Please advise.", " Can you help?", " Any help appreciated.",
    " Let me know.", " Please get back to me asap.",
    " I'd really appreciate a quick response.",
    " Regards.", " Cheers.",
]

PRODUCTS = [
    "the blue running shoes", "the coffee grinder", "a pair of headphones",
    "the desk lamp", "my winter jacket", "the kitchen scales",
    "two phone cases", "the yoga mat", "the espresso machine",
    "a set of bed sheets", "the wireless keyboard", "my new backpack",
    "the standing desk", "a bluetooth speaker", "the air fryer",
    "the running watch", "some replacement filters", "the office chair",
]

TIMEFRAMES = [
    "last Tuesday", "on the 3rd", "two weeks ago", "last month",
    "over a week ago", "on Friday", "yesterday", "at the weekend",
    "nearly ten days ago", "on the 14th", "earlier this month",
    "three days ago", "back in June",
]


def _order_ref(rng: np.random.Generator) -> str:
    """Order references appear in several formats, as they would in the wild."""
    style = rng.integers(0, 5)
    if style == 0:
        return f"#{rng.integers(100000, 999999)}"
    if style == 1:
        return f"OB-{rng.integers(10000, 99999)}"
    if style == 2:
        return f"order {rng.integers(100000, 999999)}"
    if style == 3:
        return f"#OB{rng.integers(1000, 9999)}"
    return f"{rng.integers(10000000, 99999999)}"


# --------------------------------------------------------------------------
# Intent definitions
#
# Each intent supplies:
#   core    - the main ask, several distinct phrasings
#   detail  - optional supporting sentence, several phrasings
# Placeholders: {order} {product} {when}
# --------------------------------------------------------------------------

INTENTS: dict[str, dict[str, list[str]]] = {
    "order_status": {
        "core": [
            "where is my order {order}?",
            "can you tell me what's happening with {order}?",
            "I placed an order {when} and I still have no idea where it is.",
            "any update on {order}?",
            "I want to check the status of my order.",
            "what's the current status of {order}?",
            "could you look up {order} for me and tell me where it's at?",
            "I ordered {product} {when} and I've heard nothing since.",
            "is my order actually on its way or not?",
            "checking in on {order} - has it shipped yet?",
        ],
        "detail": [
            "", "", "",
            " The tracking page hasn't updated in days.",
            " I never got a shipping confirmation email.",
            " The app just says processing.",
            " I ordered {product}.",
            " It was supposed to be a gift so I'm getting nervous.",
            " The tracking number you sent doesn't work.",
        ],
    },
    "delivery_delay": {
        "core": [
            "my order is late.",
            "{order} was due {when} and it still hasn't turned up.",
            "the delivery date has been pushed back twice now.",
            "I paid for next day delivery and it's been {when}.",
            "my parcel is way past its estimated delivery date.",
            "why is my delivery taking so long?",
            "the courier keeps saying delayed with no explanation.",
            "you promised delivery {when} and nothing has arrived.",
            "this order is now a week overdue.",
        ],
        "detail": [
            "", "",
            " I need {product} before the weekend.",
            " Can you tell me when it will actually arrive?",
            " If it can't get here soon I'd rather cancel.",
            " This is really frustrating.",
            " The tracking has said out for delivery for two days.",
            " I've taken time off work waiting for this.",
        ],
    },
    "cancel_order": {
        "core": [
            "I want to cancel {order}.",
            "please cancel my order.",
            "can I still cancel {order} before it ships?",
            "I need to cancel the order I placed {when}.",
            "I've changed my mind, please cancel.",
            "how do I cancel an order?",
            "cancel {order} please, I ordered by mistake.",
            "I'd like to call off my order for {product}.",
            "is it too late to cancel {order}?",
        ],
        "detail": [
            "", "",
            " I ordered the wrong size.",
            " I found it cheaper elsewhere.",
            " I don't need it any more.",
            " It hasn't shipped yet as far as I can tell.",
            " Please confirm the cancellation in writing.",
            " I ordered twice by accident.",
        ],
    },
    "return_request": {
        "core": [
            "I want to return {product}.",
            "how do I return an item?",
            "I'd like to send back {order}.",
            "can I return {product} that I bought {when}?",
            "I need to arrange a return.",
            "what's your returns process?",
            "please send me a returns label for {order}.",
            "I want to send this back, it isn't what I expected.",
            "starting a return for {product}.",
        ],
        "detail": [
            "", "",
            " It doesn't fit.",
            " The colour is nothing like the photos.",
            " It's still unopened in the original packaging.",
            " I'm within the 30 day window I think.",
            " Do I have to pay for return postage?",
            " Where do I send it?",
        ],
    },
    "refund_status": {
        "core": [
            "where is my refund?",
            "I returned {product} {when} and still no refund.",
            "when will I get my money back for {order}?",
            "my refund hasn't come through yet.",
            "you confirmed my return but I haven't been refunded.",
            "chasing the refund on {order}.",
            "how long do refunds take?",
            "I was told I'd be refunded {when} and nothing has arrived.",
            "still waiting on the refund for {product}.",
        ],
        "detail": [
            "", "",
            " The tracking shows you received it.",
            " It's been over two weeks now.",
            " Nothing has shown up on my bank statement.",
            " I have the return receipt if you need it.",
            " Can you check which card it went back to?",
        ],
    },
    "exchange_item": {
        "core": [
            "can I exchange {product} for a different size?",
            "I'd like to swap this for another colour.",
            "instead of a refund can I just exchange it?",
            "I need to exchange {order} for the correct item.",
            "how do exchanges work?",
            "please exchange {product} for the larger size.",
            "I want to trade this in for a different model.",
        ],
        "detail": [
            "", "",
            " I ordered a medium but I need a large.",
            " I'd rather have the black one.",
            " Do I need to return the original first?",
            " Happy to pay any difference in price.",
        ],
    },
    "change_shipping_address": {
        "core": [
            "I need to change the delivery address on {order}.",
            "can you send my order to a different address?",
            "I've moved house, please update my shipping address.",
            "wrong address on {order}, can you fix it?",
            "please redirect my parcel to my work address.",
            "how do I update the delivery address?",
            "I entered the old address by mistake on {order}.",
            "can the delivery be changed to a neighbour?",
        ],
        "detail": [
            "", "",
            " It hasn't shipped yet so hopefully it's not too late.",
            " I'll send the new address once you confirm.",
            " The postcode is wrong on the order.",
            " I put my old flat number in.",
        ],
    },
    "change_payment_method": {
        "core": [
            "I need to change the card on {order}.",
            "can I pay with a different card?",
            "my payment failed, how do I update it?",
            "please update the payment method for my order.",
            "the card I used has expired, what do I do?",
            "can I switch to paypal for {order}?",
            "I want to update my billing details.",
        ],
        "detail": [
            "", "",
            " My bank flagged the transaction.",
            " The card was cancelled after fraud on my account.",
            " I don't want the order held up over this.",
            " The new card is already on my account.",
        ],
    },
    "damaged_item": {
        "core": [
            "{product} arrived damaged.",
            "my order turned up broken.",
            "the item in {order} is cracked.",
            "I opened the box and {product} was smashed.",
            "received {product} {when} and it's clearly damaged.",
            "the packaging was crushed and the contents are ruined.",
            "the screen was already broken when it arrived.",
        ],
        "detail": [
            "", "",
            " I've taken photos if you need them.",
            " The box looked fine from the outside.",
            " I want a replacement, not a refund.",
            " This is unacceptable for the price.",
            " It doesn't power on at all.",
        ],
    },
    "wrong_item_received": {
        "core": [
            "I received the wrong item.",
            "you sent me the wrong thing in {order}.",
            "I ordered {product} but got something completely different.",
            "the item in the box isn't what I ordered.",
            "wrong product delivered for {order}.",
            "this isn't the size I ordered.",
            "somebody else's order arrived at my house.",
        ],
        "detail": [
            "", "",
            " The packing slip has someone else's name on it.",
            " I ordered a medium and received an extra small.",
            " Do I need to send this one back?",
            " Please send the correct item as soon as possible.",
        ],
    },
    "missing_item": {
        "core": [
            "part of my order is missing.",
            "{order} arrived but one item wasn't in the box.",
            "I ordered three things and only two arrived.",
            "{product} is missing from my delivery.",
            "my parcel came half empty.",
            "one of the items on my invoice never turned up.",
            "the box arrived but it was missing an item.",
        ],
        "detail": [
            "", "",
            " The packing slip lists it but it isn't there.",
            " I was charged for the full order.",
            " Is it being sent separately?",
            " The box didn't look tampered with.",
        ],
    },
    "invoice_receipt_request": {
        "core": [
            "can I get a receipt for {order}?",
            "please send me a VAT invoice.",
            "I need an invoice for my records.",
            "where do I download my receipt?",
            "can you email me the invoice for the order I placed {when}?",
            "I need a proper invoice for expenses.",
            "could you resend the receipt for {order}?",
        ],
        "detail": [
            "", "",
            " It needs to have my company name on it.",
            " I can't find it in my account.",
            " My accountant needs it for the quarter.",
            " The confirmation email doesn't count as an invoice.",
        ],
    },
    "discount_promo_code": {
        "core": [
            "my promo code didn't work.",
            "the discount wasn't applied to {order}.",
            "I have a 20% off code but it's being rejected.",
            "can you apply my discount retroactively?",
            "why was I charged full price when I used a voucher?",
            "the code from your newsletter isn't valid at checkout.",
            "do you have any offers on {product}?",
        ],
        "detail": [
            "", "",
            " The code was WELCOME20.",
            " It said code expired but it shouldn't be.",
            " I only ordered because of the offer.",
            " Can you refund the difference?",
        ],
    },
    "account_login_issue": {
        "core": [
            "I can't log into my account.",
            "my password reset email never arrives.",
            "I'm locked out of my account.",
            "the site won't accept my password.",
            "how do I change the email on my account?",
            "I can't see my order history since the update.",
            "my account seems to have been deleted.",
        ],
        "detail": [
            "", "",
            " I've tried resetting three times.",
            " Nothing in my spam folder either.",
            " I need to get in to track {order}.",
            " It keeps saying account not found.",
        ],
    },
    "product_availability": {
        "core": [
            "when will {product} be back in stock?",
            "do you have {product} in stock?",
            "is {product} being restocked?",
            "the item I want is showing as sold out.",
            "can you tell me if {product} will return?",
            "will you get more of {product} in?",
            "is there a waiting list for out of stock items?",
        ],
        "detail": [
            "", "",
            " I've been checking the site every day.",
            " I need it in a size 10.",
            " Can you notify me when it's available?",
            " Is it discontinued?",
        ],
    },
    "subscription_manage": {
        "core": [
            "I want to cancel my subscription.",
            "how do I pause my monthly delivery?",
            "can I change my subscription frequency?",
            "please stop my recurring order.",
            "I'm being charged monthly and I want it to stop.",
            "can I skip next month's delivery?",
            "I need to update what's in my subscription box.",
        ],
        "detail": [
            "", "",
            " I've got too much stock at home already.",
            " I'd like it every two months instead.",
            " I don't remember signing up for this.",
            " Please confirm nothing further will be charged.",
        ],
    },
}

CHANNELS = ["email", "chat", "web_form"]
CHANNEL_P = [0.45, 0.35, 0.20]

# --------------------------------------------------------------------------
# Surface noise
# --------------------------------------------------------------------------

TYPO_MAP = {
    "order": "oder", "delivery": "delivary", "received": "recieved",
    "please": "pls", "because": "becuase", "the": "teh",
    "cancel": "cancell", "refund": "refud", "address": "adress",
    "account": "acount", "subscription": "subscripton",
}


def _apply_noise(text: str, rng: np.random.Generator) -> str:
    """Introduce realistic surface variation: typos, casing, punctuation."""
    r = rng.random()

    # ~8% of messages carry a typo
    if r < 0.08:
        for correct, typo in TYPO_MAP.items():
            if correct in text.lower():
                pattern = re.compile(re.escape(correct), re.IGNORECASE)
                text = pattern.sub(typo, text, count=1)
                break

    r2 = rng.random()
    if r2 < 0.03:            # shouty
        text = text.upper()
    elif r2 < 0.09:          # all lowercase, no capitalisation
        text = text.lower()

    if rng.random() < 0.05:  # dropped terminal punctuation
        text = text.rstrip(".!?")

    if rng.random() < 0.04:  # doubled punctuation
        text = text.replace("?", "??", 1)

    return text


def _fill(template: str, rng: np.random.Generator) -> str:
    if "{order}" in template:
        template = template.replace("{order}", _order_ref(rng))
    if "{product}" in template:
        template = template.replace("{product}", PRODUCTS[rng.integers(len(PRODUCTS))])
    if "{when}" in template:
        template = template.replace("{when}", TIMEFRAMES[rng.integers(len(TIMEFRAMES))])
    return template


def build_dataset(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    intent_names = list(INTENTS.keys())

    # Deliberately uneven intent distribution — real support queues are
    # dominated by a few intents, and HDBSCAN must cope with that.
    weights = np.array([
        1.00,  # order_status
        0.85,  # delivery_delay
        0.70,  # cancel_order
        0.75,  # return_request
        0.80,  # refund_status
        0.45,  # exchange_item
        0.55,  # change_shipping_address
        0.35,  # change_payment_method
        0.50,  # damaged_item
        0.50,  # wrong_item_received
        0.45,  # missing_item
        0.35,  # invoice_receipt_request
        0.40,  # discount_promo_code
        0.40,  # account_login_issue
        0.35,  # product_availability
        0.30,  # subscription_manage
    ])
    probs = weights / weights.sum()

    rows = []
    for i in range(n_rows):
        intent = intent_names[rng.choice(len(intent_names), p=probs)]
        spec = INTENTS[intent]

        core = spec["core"][rng.integers(len(spec["core"]))]
        detail = spec["detail"][rng.integers(len(spec["detail"]))]
        opener = OPENERS[rng.integers(len(OPENERS))]
        closer = CLOSERS[rng.integers(len(CLOSERS))]

        body = _fill(core, rng) + _fill(detail, rng)
        # Capitalise the first letter of the core when there is no opener
        if not opener and body and body[0].islower():
            body = body[0].upper() + body[1:]

        text = _apply_noise(f"{opener}{body}{closer}".strip(), rng)

        rows.append({
            "message_id": f"msg_{i:05d}",
            "text": text,
            "intent": intent,
            "channel": CHANNELS[rng.choice(len(CHANNELS), p=CHANNEL_P)],
            "char_len": len(text),
        })

    return pd.DataFrame(rows)


def main() -> None:
    df = build_dataset()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    checksum = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()
    manifest = {
        "seed": SEED,
        "n_rows": int(len(df)),
        "n_intents": int(df["intent"].nunique()),
        "intent_counts": df["intent"].value_counts().to_dict(),
        "unique_texts": int(df["text"].nunique()),
        "duplicate_rate": round(1 - df["text"].nunique() / len(df), 4),
        "mean_char_len": round(float(df["char_len"].mean()), 1),
        "sha256": checksum,
        "generator": "src/generate_dataset.py",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(df):,} rows -> {OUT_CSV}")
    print(f"Intents: {df['intent'].nunique()}  "
          f"unique texts: {df['text'].nunique():,} "
          f"({manifest['duplicate_rate']:.1%} duplicates)")
    print(f"Mean length: {manifest['mean_char_len']} chars")
    print(f"sha256: {checksum[:16]}...")


if __name__ == "__main__":
    main()
