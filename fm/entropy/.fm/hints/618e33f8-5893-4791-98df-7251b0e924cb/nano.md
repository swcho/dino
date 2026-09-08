A clean flat-design educational infographic, 16:9 landscape, titled "DINO Teacher Target Pipeline" at the top center in bold dark sans-serif.

LAYOUT: One wide horizontal band across the upper two thirds is the TEACHER PATH, reading strictly left to right. A narrower band across the bottom third is the STUDENT PATH, also left to right, drawn in a muted gray-blue to contrast with the vivid teacher band. A thin dashed vertical guide separates title area from diagram.

TEACHER PATH (upper band), five boxes connected by bold right-pointing arrows:
1. A rounded box labeled "Teacher logits z" containing a small bar chart thumbnail with 8 uneven bars, moderately spread.
2. A rounded orange box labeled "Subtract center" with the formula "z - c" beneath it. Under the box, a small bar chart thumbnail showing the same 8 bars but rebalanced, the tallest bar pulled down.
3. A rounded purple box labeled "Divide by teacher temp" with "/ 0.04" beneath it. Under the box, a small bar chart thumbnail with exaggerated gaps between bars.
4. A rounded blue box labeled "Softmax" with "dim = -1" beneath it. Under the box, a small bar chart thumbnail showing one tall spike and tiny bars, labeled "sharp target q".
5. A rounded gray box labeled "Detach" with a small scissors icon cutting a dashed arrow, and the caption "stop gradient".

TOP CALLOUT LABELS, above the teacher band, each in a small colored pill with a short curved arrow pointing down to its box:
- Orange pill "prevents one-hot collapse" pointing to box 2 "Subtract center".
- Purple pill "prevents uniform collapse" pointing to box 3 "Divide by teacher temp".
Place a tiny two-headed horizontal arrow between the two pills labeled "opposite forces, balanced".

STUDENT PATH (bottom band), two boxes connected by an arrow, visually parallel to the teacher path and offset slightly right:
1. A rounded gray-blue box labeled "Student logits" then an arrow to
2. A box labeled "Divide by student temp" with "/ 0.1" beneath, then an arrow to
3. A box labeled "Log softmax".
Under this band write in small text "no centering, gradients flow".

At the far right, both bands converge with two arrows into a single dark box labeled "Cross entropy loss" showing the formula "- sum q log p".

Small footer strip under the teacher band, three tiny tags in a row: "centering before softmax", "temp scales the center too", "center updated from raw logits".

STYLE: clean flat design, educational infographic, generous white space, soft off-white background, thick rounded arrows, consistent 2px outlines, restrained palette of orange, purple, blue, gray-blue on white, crisp modern sans-serif, no gradients, no shadows, no photographic elements. All labels short and legible. Aspect ratio 16:9.
