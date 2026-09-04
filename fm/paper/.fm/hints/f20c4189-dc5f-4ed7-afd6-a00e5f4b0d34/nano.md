Create a clean flat-design educational infographic comparing three self-supervised learning loss functions. Aspect ratio 16:9, landscape.

Title at the top center, large bold: "Three Losses, Three Ways" with a smaller subtitle underneath: "DINO vs MoCo-v2 vs BYOL".

Layout: a three-column comparison grid. Each column is a rounded card with its own accent color. Left column accent teal, middle column accent orange, right column accent purple. Above the columns, a narrow row of left-aligned row labels on the far left edge, and the grid has four horizontal bands so the eye reads left-to-right across each band. Thin light gray divider lines between the bands.

Column headers (top of each card, bold, with a small colored pill badge):
- Left: "DINO" / badge text "CE"
- Middle: "MoCo-v2" / badge text "InfoNCE"
- Right: "BYOL" / badge text "MSE"

Band 1, row label on far left: "What is matched?" (Korean gloss underneath in small text: 무엇을 비교하나)
- Left cell: a small diagram of two side-by-side bar-chart histograms, each with 5 spiky bars, one histogram labeled "teacher" and the other "student", connected by a short double-headed arrow. Caption under it: "Distribution vs distribution".
- Middle cell: a small diagram with one blue dot labeled "q" in the center, one green dot labeled "k+" pulled toward it by a bold attracting arrow, and four small red dots around the edge labeled "negatives" pushed outward by repelling arrows. Caption: "One positive, many negatives".
- Right cell: a small diagram of a light circle (unit sphere) with two arrows from the center pointing in nearly the same direction, one arrow labeled "prediction", the other labeled "target", with a small angle marker between them. Caption: "Vector vs vector".

Band 2, row label: "Negatives needed?" (Korean gloss: negative 필요?)
- Left cell: a large red X mark and the word "No".
- Middle cell: a large green check mark and the words "Yes, from queue".
- Right cell: a large red X mark and the word "No".

Band 3, row label: "Anti-collapse device" (Korean gloss: 붕괴 방지 장치)
- Left cell: two small stacked chips reading "centering" and "sharpening", joined by a plus sign.
- Middle cell: one chip reading "negatives push apart".
- Right cell: one chip reading "predictor + stop-grad".

Band 4, row label: "Formula" (Korean gloss: 손실 수식)
- Left cell, math typeset cleanly: -Σ Pt log Ps
- Middle cell: -log( exp(q·k+/τ) / Σ exp(q·kj/τ) )
- Right cell: || q(zs) - zt ||²  with a small note underneath: "L2-normalized  =  2 - 2cos"

At the bottom, a full-width horizontal highlight strip with a light yellow background, a small warning icon, and the text: "Sharpening is for probabilities only — never with MSE".

Style: clean flat design, educational infographic, generous white space, off-white background, soft rounded corners, subtle drop shadows on the cards, crisp sans-serif typography, muted pastel accent colors, high legibility. No photographic elements, no clutter, no extra text beyond the labels specified.
