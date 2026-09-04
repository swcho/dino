Create a clean flat-design educational infographic, 16:9 landscape, titled "PatchEmbed: Shape Pipeline" at the top center.

Layout: four equal panels in a single left-to-right row, connected by three thick horizontal arrows. Reading flow is strictly left to right. Each panel is a rounded white card on a light gray background, with a small step number badge (1, 2, 3, 4) in its top-left corner, a simple diagram in the middle, and a bold monospace tensor shape label underneath.

Panel 1: a photo-like square image tile drawn as three stacked RGB color planes (red, green, blue offset squares). Label below in bold monospace: "(B, 3, 224, 224)". Small caption under it: "Input image".

Arrow 1 between panel 1 and 2, labeled above the arrow: "Conv2d k=16 s=16".

Panel 2: a square grid of 14 by 14 small cells, each cell a tiny colored square, drawn as a shallow 3D stack of feature-map slices to suggest D channels of depth. Label below in bold monospace: "(B, D, 14, 14)". Small caption: "196 patches grid".

Arrow 2, labeled above: "flatten(2)".

Panel 3: the grid cells unrolled into one long horizontal strip of 196 small squares, shown as a tall stack of D such strips. Label below in bold monospace: "(B, D, 196)". Small caption: "Channels first".

Arrow 3, labeled above: "transpose(1, 2)".

Panel 4: 196 vertical thin bars standing side by side, each bar a token vector of length D, with a bracket marking bar height as "D". Label below in bold monospace: "(B, 196, D)". Small caption: "Token sequence 토큰 시퀀스".

Bottom strip across the full width: a single thin banner with the short text "Conv k=s=P == patch flatten + Linear".

Style: clean flat vector infographic, educational textbook look, generous white space, thin outlines, muted blue and teal accent palette with orange arrows, crisp sans-serif for captions and monospace for tensor shapes, no photorealism, no gradients, no clutter. All labels must be short and spelled exactly as given.
