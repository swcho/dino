Create a clean flat-design educational infographic, 16:9 landscape aspect ratio, titled "DINOHead: MLP to Prototype Logits" in a bold sans-serif header across the top.

Overall layout: one dominant left-to-right horizontal pipeline occupying the central 70% of the canvas, reading strictly left to right, with each stage drawn as a vertical rounded-rectangle "slab" whose WIDTH is directly proportional to its vector dimension, so the eye sees the shape widen and then sharply narrow into a bottleneck. Thin arrows connect consecutive slabs. A slim caption strip runs along the bottom.

Pipeline stages, left to right, each labeled directly beneath its slab in short bold text:

1. A narrow slab, light gray-blue, labeled "CLS vector 384". Draw a small vision-transformer token grid icon feeding into it from the far left.
2. Arrow, then a WIDE slab in medium blue labeled "Linear 2048". It is visibly about five times wider than the first slab.
3. A small circular badge with a smooth S-shaped activation curve inside, labeled "GELU".
4. Another WIDE slab in medium blue of identical width, labeled "Linear 2048".
5. A second identical circular badge labeled "GELU".
6. Arrow, then a clearly NARROW slab in deep teal, dramatically thinner than the 2048 slabs, labeled "Linear 256 bottleneck". Draw the connecting arrows here as a converging funnel or wedge shape so the narrowing is unmistakable.
7. A prominent sphere icon: a wireframe unit sphere with latitude and longitude lines, drawn larger than the slabs. A short arrow from the bottleneck points at the sphere, and a single small bright dot sits ON the sphere's surface with a thin radius line from the center to that dot. Label underneath: "L2 normalize to sphere". A tiny side note in small type: "norm equals one".
8. Arrow, then a slab in orange labeled "Weight-normed Linear". Inside it, draw a few thin unit-length ray arrows radiating from a common origin to suggest prototype directions. Small note beneath: "K unit prototypes".
9. Final panel on the right: a fan of horizontal bars spreading from a shared vertical zero axis, some bars extending right (positive) and some left (negative), varying in length, colored on a teal-to-orange scale. Label: "K cosine logits". Add a small horizontal scale under the fan marked at its two ends with "-1" and "+1".

Bottom caption strip: three short evenly spaced notes in small type with tiny icons — "Unit sphere bounds logits", "Prototype norms fixed to 1", "Head discarded after training".

Style: clean flat design, educational infographic, generous white space on an off-white background, restrained palette of slate gray, medium blue, deep teal and one orange accent, crisp thin outlines, no gradients except the subtle sphere shading, no photorealism, no clutter. All text short, in English, correctly spelled, high contrast and legible. Do not add any text beyond the labels specified above.
