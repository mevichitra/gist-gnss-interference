# GIST explained like I'm 5

## ELI5

Planes use GPS (signals from satellites) to know where they are.
They shout their position, height and a "how sure am I" score (NIC) over radio, and hobbyists' receivers record it and share it free (adsb.lol).
Jamming is someone blasting noise so the plane loses GPS, and its "how sure" score drops.
Spoofing is someone sending fake GPS, so the plane is confidently wrong and its score stays high.
Think of a classroom where the teacher asks "where are we?" and one kid whispers a wrong answer to the girl next to him.
She says it loudly and proudly, so just listening to how confident she sounds tells you nothing.
GIST instead checks her answer against things the whisperer can't fake: her air-pressure height, the wind, and what her neighbours say.
A deep-learning model (a Transformer plus planes "talking" to nearby planes) reads the last 10 minutes of each plane and flags the ones that disagree with the crowd.
No humans labelled anything: 10 rule-of-thumb checks vote, and a Dawid-Skene model combines the votes, plus fake jammers and spoofers are added to real traffic so we know the exact answers.

## Say this during the demo

- "Each hexagon is a patch of sky; orange means likely jammed, purple means likely spoofed."
- "The NIC baseline is what the simple confidence-score map sees, and it misses spoofing."
- "The pink X is where the model thinks the jammer is."
- "The bars at the bottom are the 24 hours of the day; watch it play hour by hour."
- "Interim result: spoofing caught 0.80 of the time at 1% false alarms, versus 0.01 for the simple map."

## If asked

**How accurate is the jammer location?**
Median error is about 16.5 km on simulated jammers (seed 0, validation days).

**Does it work on real jammers?**
Not validated yet. We plan to check real-world locating with Denver 2022 data via OpenSky.

**What is not finished?**
The final 3-seed results; today's numbers are interim, from one seed.
