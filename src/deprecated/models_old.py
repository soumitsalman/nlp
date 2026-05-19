import json
from itertools import chain
from typing import Optional

from pydantic import BaseModel, Field

from .utils import *

M_GIST = "# GIST"
M_CATEGORIES = "# DOMAINS"
M_ENTITIES = "# ENTITIES"
M_TOPIC = "# TOPIC"
M_REGIONS = "# REGIONS"
M_SUMMARY = "# SUMMARY"
M_KEYPOINTS = "# KEY POINTS"
M_KEYEVENTS = "# KEY EVENTS"
M_DATAPOINTS = "# KEY POINTS"
M_INSIGHT = "# ACTIONABLE INSIGHT"
M_FIELDS = [
    M_GIST,
    M_CATEGORIES,
    M_ENTITIES,
    M_TOPIC,
    M_REGIONS,
    M_SUMMARY,
    M_KEYPOINTS,
    M_KEYEVENTS,
    M_DATAPOINTS,
    M_INSIGHT,
]
M_START = "```markdown"
M_END = "```"
MARKDOWN_HEADERS = ["# ", "## ", "### ", "#### ", "**"]

C_KEYPOINTS = "P:"
C_KEYEVENTS = "E:"
C_DATAPOINTS = "D:"
C_REGIONS = "R:"
C_ENTITIES = "N:"
C_CATEGORIES = "C:"
C_SENTIMENTS = "S:"
COMPRESSED_FIELDS = [
    C_KEYPOINTS,
    C_KEYEVENTS,
    C_DATAPOINTS,
    C_REGIONS,
    C_ENTITIES,
    C_CATEGORIES,
    C_SENTIMENTS,
]
UNDETERMINED = ["n/a", "none", "undetermined", "not specified", "not mentioned"]

_MAX_NAME_LEN = 40
_MAX_STOCK_TICKER_LEN = 5
strip_nonalphanum = lambda text: re.sub(r"^[\W_]+|[\W_]+$", "", text)

def valid_unique(items: list[str], additional_filter=lambda x: True):
    if not items:
        return items
    return list({item.lower(): item for item in map(strip_nonalphanum, items) if item and additional_filter(item)}.values())

def valid_names(items: list[str]):
    return valid_unique(items, additional_filter=lambda x: len(x) <= _MAX_NAME_LEN)

def valid_stock_tickers(items: list[str]):
    return valid_unique(items, additional_filter=lambda x: len(x) <= _MAX_STOCK_TICKER_LEN)

def valid_regions(items: list[str]):
    return list(map(lambda x: x.replace(".", ""), valid_names(items)))

_VALIDATE_FUNC = {
    "keypoints": valid_unique,
    "keyevents": valid_unique,
    "datapoints": valid_unique,
    "entities": valid_names,
    "people": valid_names,
    "regions": valid_regions,
    "products": valid_unique,
    "organizations": valid_names,
    "stock_tickers": valid_stock_tickers,    
}

class Digest(BaseModel):
    raw: str
    keypoints: Optional[list[str]] = Field(default=None)
    keyevents: Optional[list[str]] = Field(default=None)
    datapoints: Optional[list[str]] = Field(default=None)
    categories: Optional[list[str]] = Field(default=None)
    sentiments: Optional[list[str]] = Field(default=None)

    entities: Optional[list[str]] = Field(default=None)
    people: Optional[list[str]] = Field(default=None)
    regions: Optional[list[str]] = Field(default=None)
    products: Optional[list[str]] = Field(default=None)
    organizations: Optional[list[str]] = Field(default=None)
    stock_tickers: Optional[list[str]] = Field(default=None)

    def model_post_init(self, __context):
        for field, validate_func in _VALIDATE_FUNC.items():
            if getattr(self, field):
                setattr(self, field, validate_func(getattr(self, field)))

    @classmethod
    def parse_json(cls, response: str):
        try:
            kvpair = json.loads(response[response.find("{") : response.rfind("}") + 1])
            return Digest(**kvpair, raw=response)
        except json.JSONDecodeError:
            return

    @classmethod
    def parse_markdown(cls, response: str):
        digest = Digest(raw=response)
        response = response.strip().removeprefix(M_START).removesuffix(M_END).strip()
        last = None
        for line in response.splitlines():
            line = line.strip()
            if not line or line == UNDETERMINED:
                continue

            if any(field in line for field in M_FIELDS):
                last = line
            elif M_GIST in last:
                digest.gist = line
            elif M_CATEGORIES in last:
                digest.categories = split_parts(line)
            elif C_ENTITIES in last:
                digest.entities = split_parts(line)
            elif M_TOPIC in last:
                digest.topic = line
            elif C_REGIONS in last:
                digest.regions = split_parts(line)
            elif M_SUMMARY in last:
                digest.summary = (
                    (digest.summary + "\n" + line) if digest.summary else line
                )
            elif C_KEYPOINTS in last:
                if not digest.keypoints:
                    digest.keypoints = []
                digest.keypoints.append(line.removeprefix("- ").removeprefix("* "))
            elif M_INSIGHT in last:
                digest.insight = line

        return digest

    @classmethod
    def parse_compressed(cls, response: str):
        if not response:
            return response

        results = {"P:": [], "E:": [], "D:": [], "N:": [], "R:": []}
        current_pos = 0
        while current_pos < len(response):
            key = response[current_pos : current_pos + 2]
            if key in results:
                next_key_pos = [
                    response.find(";" + next_key, current_pos + 2)
                    for next_key in results.keys()
                ]
                end = min(
                    [pos for pos in next_key_pos if pos > -1], default=len(response)
                )
                if response[end - 1] == ";":
                    ext = response[current_pos + 2 : end - 1]
                else:
                    ext = response[current_pos + 2 : end]

                results[key].extend(
                    chain(*(item.strip().split(";") for item in ext.strip().split("|")))
                )
                current_pos = end

            current_pos += 1

        response = ""
        for key, value in results.items():
            if not value:
                continue
            response += key + "|".join(v.strip() for v in value) + ";"

        return Digest(
            raw=response,
            keypoints=results.get("P:") or None,
            keyevents=results.get("E:") or None,
            datapoints=results.get("D:") or None,
            entities=results.get("N:") or None,
            regions=results.get("R:") or None,
        )


_THSTART = "<think>"
_THEND = "</think>"
M_TITLE_PREFIX = ["Topic:", "Topics:", "Intelligence Briefing:", "News Recap:"]
M_TITLE = ["## Title"]
M_INTRODUCTION = ["## Introduction"]
M_ANALYSIS = ["## Analysis"]
M_INSIGHTS = [
    "## Key Datapoints",
    "## Key Takeaways",
    "## Key Trends & Insights",
    "## Datapoints",
    "## Takeaways",
]
M_VERDICT = ["## Verdict", "## Conclusion"]
M_PREDICTION = ["## Prediction", "## Predictions"]
M_KEYWORDS = ["## Keywords"]
A_FIELDS = list(
    chain(
        *[
            M_TITLE,
            M_INTRODUCTION,
            M_ANALYSIS,
            M_INSIGHTS,
            M_VERDICT,
            M_PREDICTION,
            M_KEYWORDS,
        ]
    )
)


class Metadata(BaseModel):
    headline: str = Field(description="Headline for the article. Length <= 20 Words")
    question: Optional[str] = Field(
        default=None,
        description="Specific question that the article addresses. Length <= 20 Words",
    )
    highlights: list[str] = Field(
        description="List of highlights and data points. each highlight length <= 20 Words"
    )
    keywords: list[str] = Field(
        description="List of keywords and names. each keyword length <= 3 Words"
    )
    banner_prompt: Optional[str] = Field(
        default=None,
        description="text-to-image LLM Prompt for generating article banner",
    )

    # deprecated fields
    raw: Optional[str] = Field(default=None)
    intro: Optional[str] = Field(default=None)
    insights: Optional[list[str]] = Field(default=[])
    summary: Optional[str] = Field(default=None)
    predictions: Optional[list[str]] = Field(default=[])

    def parse_json(text: str):
        text = text.strip()
        # text = remove_before(text, _THEND).strip()
        text = text.removeprefix("```json").removesuffix("```").strip()

        data = json.loads(text)
        return Metadata(
            raw=text,
            headline=data.get("headline"),
            intro=data.get("introduction"),
            highlights=data.get("analysis") or data.get("highlights"),
            insights=data.get("datapoints") or data.get("takeaways"),
            summary=data.get("summary"),
            predictions=data.get("predictions"),
            keywords=data.get("keywords"),
        )

    def parse_markdown(text: str):
        text = text.strip()
        # text = remove_before(text, _THEND).strip()
        text = text.removeprefix("```markdown").removesuffix("```").strip()
        if not text:
            return

        fields = {k: [] for k in A_FIELDS}
        add_to = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line in A_FIELDS:
                add_to = line
            elif add_to:
                fields[add_to].append(line)

        split_keywords = lambda line: [
            kw.strip().removesuffix(".") for kw in line.split(",") if len(kw) <= 30
        ]
        chain_lines = lambda fnames: filter(
            lambda line: bool(line), chain(*(fields.get(fname) for fname in fnames))
        )
        try:
            return Metadata(
                raw=text,
                headline=next(chain_lines(M_TITLE), ""),
                intro="\n".join(chain_lines(M_INTRODUCTION)),
                highlights=chain_lines(M_ANALYSIS),
                insights=chain_lines(M_INSIGHTS),
                summary="\n".join(chain_lines(M_VERDICT)),
                predictions=chain_lines(M_PREDICTION),
                keywords=split_keywords(next(chain_lines(M_KEYWORDS), "")),
            )
        except:
            print(text)


def cleanup_markdown(text: str) -> str:
    text = remove_before(text, M_START)
    text = remove_after(text, M_END)
    # remove all \t with
    text = text.replace("\t", "")
    # Replace "\n(any number of spaces)\n" with "\n\n"
    text = re.sub(r"\n\s*\n", "\n\n", text)

    # removing the first line if it looks like a header
    text = text.strip()
    if any(text.startswith(tag) for tag in MARKDOWN_HEADERS):
        text = remove_before(text, "\n")

    # replace remaining headers with "**"
    text = re.sub(r"(#+ )(.*?)(\n|$)", _replace_header_tag, text)

    # # Remove any space after "\n"
    # text = re.sub(r"\n\s+", "\n", text)
    # Replace "\n\n\n" with "\n\n"
    # text = re.sub(r"\n\n\n", "\n\n", text)
    # # remove > right after \n
    # text = re.sub(r"\n>", "\n", text)
    # # replace every single \n with \n\n
    # text = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', text)
    # # Add a space after every "+" if there is no space
    # text = re.sub(r'\+(?!\s)', '+ ', text)

    return text.strip()


def _replace_header_tag(match):
    header_content = match.group(2).strip()  # The content after "# " or "## "
    newline = match.group(3)  # Preserve the newline or end of string
    return f"\n**{header_content}**{newline}"
