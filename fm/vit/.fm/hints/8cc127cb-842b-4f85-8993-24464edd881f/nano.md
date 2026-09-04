Create a clean flat-design educational infographic, 16:9 landscape aspect ratio, titled "prepare_tokens: Image to Tokens" in large bold sans-serif at the top.

Layout: three equal-width vertical panels arranged left to right, connected by two large bold arrows between them, so the eye flows left to right as one pipeline. Below the three panels, a single full-width footer strip.

Panel 1, label "Patch Embedding" at the top of the panel: a square photo-like image thumbnail on the left, overlaid with a 4x4 grid of dividing lines, its 16 cells tinted in soft blue shades. An arrow points right to a vertical stack of 16 small blue rounded rectangles representing a token sequence. Small caption under the panel: "Conv2d, kernel = stride = P". A shape tag in monospace: "(B, N, D)".

Panel 2, label "Prepend CLS Token" at the top: the same vertical stack of blue token rectangles, with one distinctly larger orange rounded rectangle marked "CLS" inserted at the very top of the stack, with a small curved arrow showing it being pushed in at the front. Small caption under the panel: "expand, no memory copy". A shape tag in monospace: "(B, N+1, D)".

Panel 3, label "Add Positional Embedding" at the top: the blue-plus-orange token stack on the left, a big bold plus sign in the middle, and a second stack of green rounded rectangles labeled "pos_embed" on the right, each pair joined by a thin horizontal line to show elementwise addition. Above the green stack, a small side note: "bicubic if size differs". Small caption under the panel: "attention is order-blind". A shape tag in monospace: "(B, N+1, D)".

Footer strip: a thin horizontal pipeline bar with four connected chips reading, left to right, "patch_embed", "cat CLS", "+ pos encoding", "pos_drop", the last chip drawn in faded gray with the tiny note "dropout = 0.0".

Style: clean flat design, educational infographic, off-white background, restrained palette of blue, orange, green and dark charcoal text, thin outlines, generous white space, no gradients, no 3D effects, no photographic realism. All labels short and in English. Text must be crisp and correctly spelled.
