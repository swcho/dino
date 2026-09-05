A clean flat-design educational infographic, 16:9 landscape, titled "DINOHead: MLP to Unit Sphere to Prototypes" in bold dark navy at the top center, with a small Korean subtitle underneath reading "3층 MLP → L2 정규화 → weight-norm".

Layout: one continuous left-to-right pipeline across four connected panels, joined by thick light-blue arrows so the eye flows strictly left to right. Numbered circular badges 1, 2, 3, 4 sit above each panel.

Panel 1 (far left, narrow): a single vertical teal rounded bar labeled "CLS token 384". Below it a tiny caption "ViT backbone output".

Panel 2: three stacked rounded rectangles forming a vertical MLP stack, drawn largest to smallest to suggest narrowing. Labels inside them, top to bottom: "Linear 384 to 2048", "Linear 2048 to 2048", "Linear 2048 to 256". Two small orange pill badges labeled "GELU" clipped to the gaps between the boxes. A slim bracket to the right of the stack labeled "bottleneck 256".

Panel 3 (visual centerpiece, largest): a large 3D wireframe sphere in soft blue with faint latitude and longitude grid lines, labeled below as "Unit sphere, L2 normalized". A short bold red arrow points from the sphere center to one bright red dot on the surface, labeled "u-tilde, norm = 1". From the same center, five thin dark-blue arrows fan outward to the surface, each ending in a small square marker; a shared label near them reads "K prototypes". A small dashed arc drawn between the red arrow and the nearest blue arrow is annotated with a bold theta symbol and the short label "cosine angle".

Panel 4 (far right): a small horizontal bar chart of five bars, some extending right and some left of a vertical zero line, headed "Logits = cosine". The axis under it is marked only "-1", "0", "+1". The tallest positive bar is highlighted red to match the red dot in Panel 3.

Along the very bottom, spanning the full width, a single pale-yellow banner strip containing the formula in clean serif math type: "z_k = cos(v_k, u-tilde), in [-1, 1]", and to its right a short bold note "weight_g = 1, frozen".

Style: clean flat vector design, educational infographic, generous white space, off-white background, restrained palette of teal, soft blue, dark navy, one accent red and one accent orange, thin uniform line weights, crisp sans-serif labels. Every label short and legible. No photographic texture, no clutter, no extra paragraphs of text.
