"""A small bundled seed pool of clean QA facts.

Real, readable examples used as the "control group" for the no-download path: the
injection harness only corrupts an example the model answers correctly with gold
context, so an injected label is trustworthy. Real datasets (NQ, HotpotQA,
TruthfulQA, BEIR, RAGTruth) are loaded via :mod:`tokentrace.data.loaders` when the
``data`` extra is installed; this pool keeps the framework fully exercisable
offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fact:
    question: str
    answer: str
    gold_fact: str                       # a sentence that contains/entails the answer
    wrong_answer: str                    # a plausible distractor answer
    entity: str                          # the disambiguating entity (stripped for ambiguity)
    ambiguous_question: str              # question with the entity replaced by a pronoun
    parametric: bool = False             # is this widely-known (model may recall parametrically)?
    distractors: list[str] = field(default_factory=list)  # topical hard-negative sentences


@dataclass
class MultiHopFact:
    question: str
    answer: str
    facts: list[str]                     # the supporting sentences (>=2 hops)
    wrong_answer: str
    distractors: list[str] = field(default_factory=list)


FACTS: list[Fact] = [
    Fact("When was the Eiffel Tower completed?", "1889",
         "The Eiffel Tower in Paris was completed in 1889.", "1920",
         "Eiffel Tower", "When was it completed?", parametric=True,
         distractors=["The Eiffel Tower is made of wrought iron.",
                      "Paris is the capital of France.",
                      "The tower was designed by Gustave Eiffel's company."]),
    Fact("What is the chemical symbol for gold?", "Au",
         "Gold has the chemical symbol Au on the periodic table.", "Go",
         "gold", "What is its chemical symbol?", parametric=True,
         distractors=["Gold is a dense, soft, yellow metal.",
                      "Silver has the symbol Ag.",
                      "Gold is widely used in jewelry and electronics."]),
    Fact("Who painted the Mona Lisa?", "Leonardo da Vinci",
         "The Mona Lisa was painted by Leonardo da Vinci.", "Michelangelo",
         "the Mona Lisa", "Who painted it?", parametric=True,
         distractors=["The Mona Lisa hangs in the Louvre museum.",
                      "It is painted in oil on a poplar panel.",
                      "The painting is famous for the subject's smile."]),
    Fact("What is the capital of Australia?", "Canberra",
         "Canberra is the capital city of Australia.", "Sydney",
         "Australia", "What is its capital?", parametric=True,
         distractors=["Sydney is the largest city in Australia.",
                      "Australia is both a country and a continent.",
                      "Melbourne is a major cultural centre."]),
    Fact("What is the boiling point of water at sea level in Celsius?", "100",
         "At sea level, water boils at 100 degrees Celsius.", "90",
         "water", "What is its boiling point at sea level in Celsius?", parametric=True,
         distractors=["Water freezes at 0 degrees Celsius.",
                      "Water is composed of hydrogen and oxygen.",
                      "Boiling point decreases at higher altitude."]),
    Fact("Which planet is known as the Red Planet?", "Mars",
         "Mars is known as the Red Planet because of its reddish appearance.", "Jupiter",
         "the Red Planet", "Which planet is it?", parametric=True,
         distractors=["Jupiter is the largest planet in the solar system.",
                      "Mars has two small moons, Phobos and Deimos.",
                      "The reddish colour comes from iron oxide."]),
    Fact("Who wrote the play Romeo and Juliet?", "William Shakespeare",
         "Romeo and Juliet was written by William Shakespeare.", "Christopher Marlowe",
         "Romeo and Juliet", "Who wrote it?", parametric=True,
         distractors=["Romeo and Juliet is a tragedy set in Verona.",
                      "Shakespeare also wrote Hamlet and Macbeth.",
                      "The play was written in the 1590s."]),
    Fact("What gas do plants primarily absorb during photosynthesis?", "carbon dioxide",
         "During photosynthesis, plants primarily absorb carbon dioxide.", "oxygen",
         "plants", "What gas do they primarily absorb during photosynthesis?", parametric=True,
         distractors=["Plants release oxygen as a by-product.",
                      "Photosynthesis occurs in the chloroplasts.",
                      "Sunlight provides the energy for the reaction."]),
    Fact("In what year did the Apollo 11 mission land on the Moon?", "1969",
         "The Apollo 11 mission landed on the Moon in 1969.", "1972",
         "Apollo 11", "In what year did it land on the Moon?", parametric=True,
         distractors=["Neil Armstrong was the first to walk on the Moon.",
                      "The mission was launched by a Saturn V rocket.",
                      "Apollo 11 returned safely to Earth."]),
    Fact("What is the largest ocean on Earth?", "Pacific Ocean",
         "The Pacific Ocean is the largest ocean on Earth.", "Atlantic Ocean",
         "ocean", "What is the largest one on Earth?", parametric=True,
         distractors=["The Atlantic Ocean is the second largest.",
                      "Oceans cover about 71 percent of Earth's surface.",
                      "The Pacific contains the Mariana Trench."]),
    Fact("Who developed the theory of general relativity?", "Albert Einstein",
         "The theory of general relativity was developed by Albert Einstein.", "Isaac Newton",
         "general relativity", "Who developed it?", parametric=True,
         distractors=["General relativity describes gravity as spacetime curvature.",
                      "Newton formulated the law of universal gravitation.",
                      "The theory was published in 1915."]),
    Fact("What is the currency of Japan?", "yen",
         "The official currency of Japan is the yen.", "won",
         "Japan", "What is its currency?", parametric=True,
         distractors=["Tokyo is the capital of Japan.",
                      "The South Korean currency is the won.",
                      "Japan is an island nation in East Asia."]),
    Fact("How many bones are in the adult human body?", "206",
         "An adult human body has 206 bones.", "300",
         "the adult human body", "How many bones are in it?", parametric=False,
         distractors=["Babies are born with around 300 bones.",
                      "Bones are connected at joints.",
                      "The femur is the longest bone in the body."]),
    Fact("What is the hardest known natural material?", "diamond",
         "Diamond is the hardest known natural material.", "quartz",
         "natural material", "What is the hardest known one?", parametric=False,
         distractors=["Diamond is a form of carbon.",
                      "Graphite is a soft form of carbon.",
                      "Hardness is measured on the Mohs scale."]),
    Fact("Which vitamin is produced when skin is exposed to sunlight?", "vitamin D",
         "The skin produces vitamin D when exposed to sunlight.", "vitamin C",
         "skin", "Which vitamin does it produce when exposed to sunlight?", parametric=False,
         distractors=["Vitamin C is found in citrus fruits.",
                      "Vitamin D helps the body absorb calcium.",
                      "Excessive sun exposure can damage skin."]),
    Fact("What is the smallest prime number?", "2",
         "The smallest prime number is 2.", "1",
         "prime number", "What is the smallest one?", parametric=False,
         distractors=["A prime number has exactly two divisors.",
                      "1 is not considered a prime number.",
                      "2 is the only even prime number."]),
]


MULTIHOP: list[MultiHopFact] = [
    MultiHopFact("Who was born first, Marie Curie or Pierre Curie?", "Pierre Curie",
                 ["Marie Curie was born in 1867.", "Pierre Curie was born in 1859."],
                 "Marie Curie",
                 ["The Curies shared a Nobel Prize in Physics.",
                  "Marie Curie discovered polonium and radium."]),
    MultiHopFact("Which is taller, the Eiffel Tower or the Statue of Liberty?", "Eiffel Tower",
                 ["The Eiffel Tower is about 330 metres tall.",
                  "The Statue of Liberty is about 93 metres tall."],
                 "Statue of Liberty",
                 ["Both are famous landmarks.",
                  "The Statue of Liberty stands in New York Harbor."]),
    MultiHopFact("Which country has a larger population, Canada or Australia?", "Canada",
                 ["Canada has a population of about 39 million.",
                  "Australia has a population of about 26 million."],
                 "Australia",
                 ["Both are large countries by land area.",
                  "Canada borders the United States."]),
    MultiHopFact("Which river is longer, the Nile or the Thames?", "Nile",
                 ["The Nile is about 6,650 kilometres long.",
                  "The Thames is about 346 kilometres long."],
                 "Thames",
                 ["The Nile flows through northeastern Africa.",
                  "The Thames runs through London."]),
    MultiHopFact("Who came first, Isaac Newton or Albert Einstein?", "Isaac Newton",
                 ["Isaac Newton was born in 1643.", "Albert Einstein was born in 1879."],
                 "Albert Einstein",
                 ["Both made major contributions to physics.",
                  "Newton formulated the laws of motion."]),
    MultiHopFact("Which is larger, Jupiter or Saturn?", "Jupiter",
                 ["Jupiter has a diameter of about 143,000 km.",
                  "Saturn has a diameter of about 120,500 km."],
                 "Saturn",
                 ["Both are gas giants.", "Saturn is famous for its rings."]),
]


def distractor_pool() -> list[str]:
    """All topical filler sentences, used as hard negatives / padding."""
    pool: list[str] = []
    for f in FACTS:
        pool.extend(f.distractors)
    for mh in MULTIHOP:
        pool.extend(mh.distractors)
    return pool
