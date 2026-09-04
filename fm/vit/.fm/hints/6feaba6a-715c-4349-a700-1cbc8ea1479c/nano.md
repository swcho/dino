Create a clean flat-design educational infographic, landscape 16:9 aspect ratio, titled at the top center: "Post-Norm vs Pre-Norm".

Layout: two equal vertical panels side by side, separated by a thin vertical divider line. A small subtitle under the main title reads "Where you put LayerNorm". Reading flow: title at top, then eyes move left panel to right panel, with a horizontal contrast; inside each panel the data flow reads bottom-to-top.

LEFT PANEL — header label "Post-Norm" in a muted grey-blue header bar, with the short formula label beneath it: "x <- LN(x + sub(x))".
Visual: a vertical main path drawn as a thick grey pipe running from bottom to top through three stacked identical blocks. In each block, the main path splits into a side branch containing a rounded box labeled "Sublayer", the branch rejoins the main path at a circled plus sign, and then a prominent solid grey-blue rounded box labeled "LayerNorm" SITS DIRECTLY ON THE MAIN PATH just above the plus sign, so the path must pass through it. Emphasize with a small icon that the main path is interrupted three times.
Gradient visualization: to the left of the stack, an upward-pointing arrow representing gradient flow, drawn thick and vivid red-orange at the top and progressively thinner, paler and more faded/dashed toward the bottom until it almost vanishes. Small caption beside it: "Gradient fades per layer".
Bottom caption box: "Needs LR warmup".

RIGHT PANEL — header label "Pre-Norm" in a vivid teal header bar, with the short formula label beneath it: "x <- x + sub(LN(x))".
Visual: a vertical main path drawn as a bold, unbroken, glossy teal highway running straight from bottom to top through three stacked identical blocks, with NO box ever placed on it. In each block, the main path splits into a side branch where a small rounded box labeled "LayerNorm" appears FIRST, feeding a rounded box labeled "Sublayer", and the branch rejoins the main path at a circled plus sign. Highlight the clean addition-only main line, e.g. with subtle lane markings like a road.
Gradient visualization: to the left of the stack, one thick, uniformly vivid, sharp red-orange arrow running unbroken from the bottom all the way through to the top, same weight along its whole length. Small caption beside it: "Gradient flows undamped".
Bottom caption box: "Trains without warmup".

Bottom strip spanning both panels: a light band containing, on the left half, the label "LN jacobians multiply" with a small stacked-multiplication glyph, and on the right half the key equation label "x_L = x_0 + sum f_l" next to "dx_L/dx_0 = I + sum df" and the short tag "identity path, gain 1". Add one small footnote-style label at the far right: "Cost: final LayerNorm needed".

Style: clean flat vector design, educational infographic, generous white space, off-white background, limited palette (muted grey-blue for Post-Norm, vivid teal for Pre-Norm, red-orange for gradient arrows, dark charcoal text), thin consistent line weights, rounded rectangles, crisp sans-serif labels, no photorealism, no clutter, all text short and in English, spelled exactly as given.
