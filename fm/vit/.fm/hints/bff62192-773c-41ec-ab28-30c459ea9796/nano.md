A clean flat-design educational infographic, 16:9 landscape, titled "Three Forward Paths of VisionTransformer".

Layout and eye flow: a single shared vertical stem on the LEFT THIRD of the canvas flowing top to bottom, which then splits into three horizontal branches fanning out to the RIGHT into three stacked panels (top, middle, bottom). Draw the split as three thick curved arrows leaving one clearly marked fork point, so the branching relationship is unmistakable. Muted off-white background, generous whitespace, thin rounded-rectangle cards, one accent color per branch (blue for top, orange for middle, green for bottom); the shared stem in neutral dark gray.

SHARED STEM (left, four stacked nodes connected by a single downward arrow), labeled top to bottom:
1. a small photo icon of a dog, label "Input image"
2. a grid of small squares, label "Patch embed"
3. a row of tokens with one highlighted first token, label "Add CLS + pos embed"
4. a stack of three identical layer blocks, label "Transformer blocks"
Below node 4 place a small diamond fork marker labeled "Shared prepare_tokens".

BRANCH 1 (top right, blue): a single tall thin vertical bar representing one vector. Labels: method name "forward", shape tag "(B, D)", use tag "k-NN / retrieval".

BRANCH 2 (middle right, orange): a 2x3 grid of six small square heatmaps in warm colors, each square showing a blurry blob. Labels: method name "get_last_selfattention", shape tag "(B, heads, N, N)", use tag "Attention visualization".

BRANCH 3 (bottom right, green): three overlapping rectangular token-grid sheets drawn in perspective as a stack. Labels: method name "get_intermediate_layers", shape tag "n x (B, N, D)", use tag "Linear probe".

Typography: clean sans-serif. Method names in monospace style, bold. Shape tags inside small rounded pill badges. Use tags in smaller italic text. Keep every label short — no sentences, no paragraphs of text anywhere. No extra legend, no watermark.
