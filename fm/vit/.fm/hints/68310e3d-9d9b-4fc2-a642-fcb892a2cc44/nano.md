A clean flat-design educational infographic, portrait 3:4 aspect ratio, titled "ViT Assembly Stack — Bottom Up (아래에서 위로)" at the very top.

Layout: one single vertical bottom-up architecture stack centered on the page. Data flows UPWARD from the bottom of the page to the top. Draw seven stacked rounded rectangular layer blocks connected by thick upward-pointing arrows between them. A thin vertical guide rail runs behind the stack. On the RIGHT side of each block, place a small monospace tensor-shape tag in a light grey pill. On the LEFT side of the stack, a slim vertical progress ribbon labeled bottom-to-top: "1" through "7".

From bottom to top:

1. Bottom block (blue): a small photo thumbnail of a dog. Label "Input Image 224x224". Right tag "(B, 3, 224, 224)".

2. Block (teal): a 14x14 grid of tiny squares with one square highlighted, plus a small conv kernel icon. Labels "PatchEmbed" and "Conv 16x16, stride 16". Right tag "(B, 196, D)".

3. Block (teal): a horizontal row of small token squares with one distinct star-marked token at the far left, and a row of small "+" plus signs beneath. Labels "Add CLS Token" and "+ Positional Embedding". Right tag "(B, 197, D)".

4. Tall repeated block (purple, drawn only once) with a bold badge in its top-right corner reading "x12". Title inside "Transformer Block". Inside it, two stacked sub-rows, each drawn as a small residual loop with a curved skip-connection arrow bypassing the sub-modules:
   - lower sub-row: three small chips in sequence "LayerNorm", "Multi-Head Attention", "DropPath", with the skip arrow rejoining at a circled plus sign.
   - upper sub-row: three small chips in sequence "LayerNorm", "MLP (GELU)", "DropPath", with the skip arrow rejoining at a circled plus sign.
   Right tag "(B, 197, D)".

5. Block (orange): a simple normalization wave icon. Label "Final LayerNorm". Right tag "(B, 197, D)".

6. Block (orange): the token row again with only the leftmost star-marked token highlighted and all other tokens greyed out, with a narrowing funnel arrow. Labels "Take CLS Token" and "x[:, 0]". Right tag "(B, D)".

7. Top block, drawn inside a DASHED border box with a small tag reading "Training only (DINO 학습 시)" (tinted pink): label "DINOHead", a small chip row "MLP", "L2 Normalize", "Prototypes", and an output label "K Prototype Logits". Right tag "(B, K)".

Style: clean flat vector educational infographic, soft pastel palette (blue, teal, purple, orange, pink) on an off-white background, thin dark outlines, generous whitespace, crisp sans-serif labels, monospace font only for the tensor shape tags. No photorealistic shading, no 3D, no clutter. All text in short English phrases, spelled exactly as given, rendered sharp and legible.
