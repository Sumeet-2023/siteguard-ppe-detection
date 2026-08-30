# "Own photos" substitute test set

**This is not what the project spec asked for.** The spec wants 60-80 photos shot on your own
phone — genuinely novel scenes with ground truth you personally verified, guaranteeing no overlap
with any training data. No camera/site access was available in this environment, so this is a
much smaller, weaker substitute: 9 CC-licensed photos pulled from Wikimedia Commons, chosen to
avoid Pexels specifically (SH17 is already Pexels-sourced) to reduce distribution overlap risk —
though it can't be ruled out entirely for web-sourced images.

## Labeling methodology (important limitation)

Ground truth was **model-assisted**, not independent from-scratch annotation:
1. Ran `yolo11s` over each image at conf=0.15 to get draft boxes.
2. Visually reviewed every box against the source image, removed false positives (a fan, a glove,
   both misfired as "helmet" at low confidence), and corrected one misclassification (a soft cloth
   cap called "helmet" — corrected to `head`).
3. **Did not** hunt for heads the model missed entirely and add new boxes for them, except by
   general visual check.

This means **recall/mAP on this set is likely optimistic** — any head the model missed outright
simply isn't in the ground truth, so it can't be scored as a miss. Precision-side findings (the
false positives caught, the cap misclassification) are the trustworthy part of this exercise;
don't read the headline mAP50 as a clean apples-to-apples number against the honest SHWD test
split or the SH17 cross-dataset eval, both of which used real, independently-sourced ground truth.

## Images and licenses

| File | Source title | License | Attribution |
|---|---|---|---|
| commons_000.jpg | Construction workers not wearing fall protection equipment | Public domain | NIOSH |
| commons_001.jpg | Deux artisans électriciens avec leurs apprentis sur un chantier (1969) | CC BY-SA 3.0 | Jean Bazard |
| commons_003.jpg | Fall arrest system (9256417786) | Public domain | NIOSH |
| commons_009.jpg | Roofing fall arrest system (9253634235) | Public domain | NIOSH |
| commons_010.jpg | Roofing workers fall prevention (9253637735) | Public domain | NIOSH |
| commons_016.jpg | A worker wears a helmet and visor at a Hong Kong construction site during a heatwave | CC BY 4.0 | Hong Kong Free Press |
| commons_017.jpg | Construction workers... canal street Nieuwe Herengracht, Amsterdam | CC0 | Fons Heijnsbroek |
| commons_018.jpg | FEMA - 24340 - Photograph by Marvin Nauman, Louisiana | Public domain | FEMA/Marvin Nauman |
| commons_076.jpg | WomanFactory1940s | Public domain | — |

All sourced from commons.wikimedia.org. commons_001.jpg (CC BY-SA 3.0) and commons_016.jpg
(CC BY 4.0) require attribution if redistributed — credited above. Raw images and labels live in
`data/own_photos/{raw,images/test,labels/test}/` — gitignored like every other dataset in this
repo (never commit the dataset), regenerable by re-running `scripts/prepare_own_photos.py` (fetch)
+ the manual correction step documented above. This file is the durable record of what they are,
where they came from, and the labeling caveat.

## Result

19 ground-truth boxes (16 helmet, 3 head) across 9 images. `yolo11s`: mAP50 0.854, precision
0.960, recall 0.833 — see `reports/benchmark.md` for the full table and the caveat above on why
this number should be read cautiously.
